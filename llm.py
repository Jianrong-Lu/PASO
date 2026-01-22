import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, disable_caching
from modelscope import snapshot_download
from modelscope import AutoTokenizer, AutoModelForCausalLM
# from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time
import argparse
from datetime import timedelta
import numpy as np
import copy
import os
import json
import csv
from datetime import datetime
from transformers import get_scheduler

# Try importing DeepSpeed
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not installed. train_loop_deepspeed will fail if selected.")

# ==========================
# Helper Function: Count Real Tokens
# ==========================
def count_real_tokens(batch, pad_token_id):
    """
    Counts non-padding tokens in a batch.
    If pad_token_id is None, returns total elements.
    """
    input_ids = batch['input_ids']
    if pad_token_id is None:
        return input_ids.numel()
    
    # Count tokens that are NOT padding
    real_tokens = (input_ids != pad_token_id).sum().item()
    return real_tokens

# ==========================
# CSV Logger Class
# ==========================
class CSVLogger:
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.csv"
        
        self.filename = filename
        self.fieldnames = set()
        self.rows = []
        directory = os.path.dirname(self.filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
    def log(self, data):
        for key in data.keys():
            if key not in self.fieldnames:
                self.fieldnames.add(key)
        self.rows.append(data)
        
    def save(self):
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(self.fieldnames))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
                
    def finish(self):
        self.save()

# ==========================
# Configuration Class
# ==========================
class Config:
    """Training configuration parameters."""
    batch_size = 8
    num_epochs = 1
    max_steps = 1000
    learning_rate = 6e-5
    momentum = 0.9

    # Acceleration configuration
    P = 4
    threshold = 1e-5
    ema_decay = 0.999
    adaptivity_type = 'mean'
    stage = 1
    
    # System configuration
    seed = 42
    device_count = torch.cuda.device_count()
    
    # Model and dataset configuration
    modelscope_name = 'Llama-3.2-1B' 
    local_model_dir = "./model/Llama-3.2-1B"
    dataset_name = 'local_wikitext'    
    local_dataset_dir = "./dataset/wikitext/wikitext-2"

    max_length = 512
    optimizer_type = 'adam'
    training_mode = 'parallel' 

    
    # Global pad token id
    pad_token_id = None 

    # Validation features
    val_interval = 1001 
    checkpoint_dir = './checkpoint'

    # Scheduler
    lr_scheduler_type = 'cosine'
    warmup_steps = 100

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

# ==========================
# Utility Functions
# ==========================
def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

# ==========================
# Model Definition
# ==========================
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def clone(self, device, other=None):
        if other is None:
            new_model_instance = copy.deepcopy(self.model)
            other = ModelWrapper(new_model_instance).to(device)
        else:
            with torch.no_grad():
                for param_to, param_from in zip(other.parameters(), self.parameters()):
                    param_to.data = param_from.data.clone()
        return other

    def get_params(self):
        return list(self.parameters())

    def set_params(self, params, clone=True):
        my_params = self.get_params()
        for p, q in zip(my_params, params):
            if clone:
                p.data = q.data.clone().to(p.device)
            else:
                p.data = q.to(p.device)

    def set_grads_from_grads(self, grads):
        my_params = self.get_params()
        for p, grad in zip(my_params, grads):
            if grad is not None:
                p.grad = grad.to(p.device)

    def compute_error_from_model(self, other):
        my_params = self.get_params()
        other_params = other.get_params()
        with torch.no_grad():
            error, total_num = 0.0, 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p - q).pow(2).item()
                total_num += np.prod(list(q.shape))
            
            return error / total_num * 1e6

class ModelFactory:
    @staticmethod
    def create_model(config):
        if config.local_model_dir and os.path.exists(config.local_model_dir):
            model_dir = config.local_model_dir
            print(f"Loading model from local path: {model_dir}")
        else:
            print(f"Downloading model '{config.modelscope_name}' from ModelScope...")
            model_dir = snapshot_download(config.modelscope_name, revision='master',cache_dir="./model")
            print(f"Model downloaded to: {model_dir}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        return ModelWrapper(model)

def load_wikitext_from_local(path):
    if not os.path.exists(path):
        raise ValueError(f"Dataset path {path} does not exist.")
    data_files = {
        "train": os.path.join(path, "wiki.train.tokens"),
        "validation": os.path.join(path, "wiki.valid.tokens"),
        "test": os.path.join(path, "wiki.test.tokens")
    }
    existing_files = {split: file for split, file in data_files.items() if os.path.exists(file)}
    if not existing_files:
        raise FileNotFoundError(f"No valid data files found in {path}.")
    print(f"Loading dataset from files: {existing_files}")
    dataset = load_dataset("text", data_files=existing_files)
    return dataset

def get_dataloaders(config, rank=None, world_size=None):
    # 1. Load Tokenizer
    if config.local_model_dir and os.path.exists(config.local_model_dir):
        model_dir = config.local_model_dir
    else:
        model_dir = snapshot_download(config.modelscope_name, revision='master', cache_dir="./model")
    
    if rank == 0 or rank is None:
        print(f"Loading tokenizer from {model_dir}...")
        
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config.pad_token_id = tokenizer.pad_token_id

    # 2. Load Dataset
    if rank == 0 or rank is None:
        print(f"Loading dataset: {config.dataset_name}...")
    
    disable_caching()

    if config.dataset_name == 'local_wikitext':
        dataset = load_wikitext_from_local(config.local_dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")

    # 3. Preprocessing
    def tokenize_function(examples, tokenizer, max_length):
        tokenized_inputs = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        labels = tokenized_inputs["labels"]
        attention_mask = tokenized_inputs["attention_mask"]
        labels[attention_mask == 0] = -100 
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def preprocess(examples):
        return tokenize_function(examples, tokenizer, config.max_length)
        
    if rank == 0 or rank is None:
        print("Tokenizing dataset...")

    column_names = dataset["train"].column_names
    tokenized_datasets = dataset.map(
        preprocess,
        batched=True,
        remove_columns=column_names,
    )
    tokenized_datasets.set_format("torch")
    
    # 4. Sampler Configuration
    use_distributed_sampler = False
    if config.training_mode.lower() in ['ddp', 'deepspeed', 'naive_dp']:
        if rank is not None and world_size is not None:
            use_distributed_sampler = True
            
    if use_distributed_sampler:
        train_sampler = DistributedSampler(
            tokenized_datasets["train"],
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_shuffle = False 
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = torch.utils.data.DataLoader(
        tokenized_datasets["train"],
        batch_size=config.batch_size if not use_distributed_sampler else config.batch_size // world_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_split = 'validation' if 'validation' in tokenized_datasets else 'test'
    
    if use_distributed_sampler:
        test_sampler = DistributedSampler(
            tokenized_datasets[val_split],
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        tokenized_datasets[val_split],
        batch_size=config.batch_size if not use_distributed_sampler else config.batch_size // world_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return train_loader, test_loader

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_perplexity += torch.exp(loss).item()
            total_batches += 1
    
    if total_batches == 0: return 0,0
    
    avg_loss = total_loss / total_batches
    avg_perplexity = total_perplexity / total_batches 
    model.train()
    return avg_loss, avg_perplexity

# =================================================================
# PASO (Parallel) and Worker Logic 
# =================================================================
def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())

def take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset, pad_token_id):
    np.random.seed(step + seed_offset)
    torch.manual_seed(step + seed_offset)
    torch.cuda.manual_seed(step + seed_offset)
    
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader) 
        batch = next(data_iter)
        
    batch = to_device(batch, device)
    num_tokens_in_batch = count_real_tokens(batch, pad_token_id)

    if 'labels' not in batch:
        batch['labels'] = batch['input_ids'].clone()      

    optimizer.zero_grad()
    outputs = model(**batch)
    if outputs.loss is None:
        raise ValueError("Model did not return loss value.")

    loss = outputs.loss
    loss.backward()
    
    perplexity = torch.exp(loss.detach()) if loss is not None else torch.tensor(float('inf'))
        
    return {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'data_iter': data_iter,
        'batch_size': batch['input_ids'].size(0),
        'num_tokens': num_tokens_in_batch
    }

def run_worker(model, optimizer, dataloader, queues, device, seed_offset, pad_token_id, rank):
    data_iter = iter(dataloader)
    
    # Reset peak memory stats at start
    torch.cuda.reset_peak_memory_stats(device)
    
    while True:
        ret = queues[0].get()
        if ret is None:
            # Capture peak memory before exiting
            peak_memory = torch.cuda.max_memory_allocated(device)
            queues[0].put((rank, peak_memory))
            return
            
        params, step = ret
        model.set_params(params, clone=False)
        res = take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset, pad_token_id)
        data_iter = res['data_iter']
        grads = [p.grad for p in model.get_params()]
        queues[1].put((grads, step, {'loss': res['loss'], 
                                    'perplexity': res['perplexity'],
                                    'batch_size': res['batch_size'],
                                    'num_tokens': res['num_tokens']})
                                    )

def train_loop_parallel(config, model, train_loader, test_loader, csv_logger):
    if config.device_count <= 1:
        print("Only one GPU found, falling back to serial training.")
        return train_loop_serial(config, model, train_loader, test_loader, csv_logger)

    mp.set_start_method('spawn', force=True)
    # Added 3rd queue for memory stats
    queues = (mp.Queue(), mp.Queue())
    processes = []

    for rank in range(1, config.device_count):
        worker_device = torch.device(f"cuda:{rank}")
        model = model.to(worker_device)

        if config.optimizer_type.lower() == 'sgd':
            worker_optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer_type.lower() == 'adam':
            worker_optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            worker_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Pass rank to worker
        p = mp.Process(target=run_worker, args=(model, worker_optimizer, train_loader, queues, worker_device, config.seed, config.pad_token_id, rank))
        p.start()
        processes.append(p)
    del worker_optimizer
    device = torch.device("cuda:0")
    model = model.to(device)
    
    # Reset Master Peak Memory
    torch.cuda.reset_peak_memory_stats(device)
    
    T = config.max_steps
    P = min(config.P, T)
    
    models = [None] * (T + 1)
    optimizers = [None] * (T + 1)
    
    # Initialize variables
    begin_idx = 0
    end_idx = P
    total_iters = 0
    total_Grad = 0
    running_loss = 0.0
    running_perplexity = 0.0
    total_consumed_tokens = 0
    thresh = config.threshold
    best_ppl = float('inf') 
    
    for step in range(P + 1):
        models[step] = model.clone(device)
        if config.optimizer_type.lower() == 'sgd':
            optimizers[step] = optim.SGD(models[step].parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer_type.lower() == 'adam':
            optimizers[step] = optim.Adam(models[step].parameters(), lr=config.learning_rate)
        else:
            optimizers[step] = optim.AdamW(models[step].parameters(), lr=config.learning_rate)

    main_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizers[begin_idx],
        num_warmup_steps=config.warmup_steps,
        num_training_steps=T
    )

    pbar = tqdm(total=T, initial=begin_idx, desc="Starting parallel training...")
    start_time = time.time()

    while begin_idx < T:
        parallel_len = end_idx - begin_idx
        pred_f = [None] * parallel_len
        metrics = [None] * parallel_len
        
        for i in range(parallel_len):
            step = begin_idx + i
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))

        for _ in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _grads
            metrics[_i] = _metrics

        rollout_model = models[begin_idx]
        rollout_optimizer = optimizers[begin_idx]
        ind, errors_all = None, 0

        for i in range(parallel_len):
            step = begin_idx + i

            current_lr = main_scheduler.get_last_lr()[0]
            for param_group in rollout_optimizer.param_groups:
                param_group['lr'] = current_lr

            rollout_model.set_grads_from_grads(pred_f[i])
            rollout_optimizer.step()
            main_scheduler.step()
            rollout_optimizer.zero_grad()
            
            error = rollout_model.compute_error_from_model(models[step + 1])
            
            if config.adaptivity_type == 'median' and i == parallel_len // 2:
                errors_all = error
            elif config.adaptivity_type == 'mean':
                errors_all += error / parallel_len

            if ind is None and (error > thresh or i == parallel_len - 1):
                ind = step + 1
                optimizer_state_clone(rollout_optimizer, optimizers[step + 1])

            if ind is None or step < ind:
                _metrics = metrics[i]
                cur_loss = _metrics['loss']
                cur_perplexity = _metrics['perplexity']
                running_loss += _metrics['loss']
                running_perplexity += _metrics['perplexity']
            total_consumed_tokens += _metrics['num_tokens']
            
            if ind is not None:
                models[step + 1] = rollout_model.clone(device, models[step + 1])
            
            total_Grad += 1

        thresh = thresh * config.ema_decay + errors_all * (1 - config.ema_decay)
        progress = ind - begin_idx
        pbar.update(progress)
        total_iters += 1
        
        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)

        for step in range(end_idx + 1, new_end_idx + 1):
            prev_model_idx = step - parallel_len - 1
            models[step] = rollout_model.clone(device, models[prev_model_idx] if prev_model_idx >= 0 else None)
            optimizers[step] = optimizers[prev_model_idx] if prev_model_idx >= 0 else None

        begin_idx = new_begin_idx
        end_idx = new_end_idx

        avg_loss = running_loss / begin_idx if begin_idx > 0 else 0
        avg_perplexity = torch.exp(torch.tensor(avg_loss))
        
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
        
        pbar.set_description(f'LR: {current_lr:.8f} |progres:{progress}| Loss: {avg_loss:.4f} |cur loss:{cur_loss:.4f}| PPL: {avg_perplexity:.2f} |cur PPL: {cur_perplexity:.2f} | Time: {elapsed_str}')            
        csv_logger.log({
            "Iter": total_iters,
            "LR": current_lr,
            "Train_Loss": avg_loss,
            "Train_Perplexity": avg_perplexity,
            "Original_Steps": begin_idx,
            "num_tokens": total_consumed_tokens,
        })
        
        if total_iters % config.val_interval == 0:
            current_model = models[begin_idx] 
            val_loss, val_ppl = evaluate_model(current_model, test_loader, device)
            
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                
            pbar.write(f"\n[Validation at para step {begin_idx}]|{config.modelscope_name}|{config.dataset_name}|{config.optimizer_type}|{config.batch_size}|{config.learning_rate}| Test PPL: {val_ppl:.2f} | Best PPL: {best_ppl:.2f}")

            csv_logger.log({
                "Iter": total_iters,
                "Original_Steps": begin_idx,
                "Val_Loss": val_loss,
                "Val_Perplexity": val_ppl,
                "Best_Perplexity": best_ppl
            })
            csv_logger.save()
        # Cleanup old steps to save memory
        for step_j in range(begin_idx, new_begin_idx):
            models[step_j] = None
            optimizers[step_j] = None
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    pbar.close()

    # --- Terminate Workers and Collect Memory ---
    peak_memories = {}
    
    # Send poison pill
    for _ in processes:
        queues[0].put(None)
    
    # Collect memory from workers
    for _ in processes:
        rank, peak_mem = queues[0].get()
        peak_memories[f"GPU_{rank}_peak_memory_MB"] = peak_mem / (1024 ** 2)

    # Collect master memory
    master_peak = torch.cuda.max_memory_allocated(device)
    peak_memories[f"GPU_0_peak_memory_MB"] = master_peak / (1024 ** 2)
    
    # Find global max
    all_peaks = [v for v in peak_memories.values()]
    max_peak_mb = max(all_peaks) if all_peaks else 0.0

    for p in processes:
        p.join()
        
    final_model = models[T]
    del models, optimizers
    test_loss, test_perplexity = evaluate_model(final_model, test_loader, device)
    m = total_Grad / T
    
    log_data = {
        "final_test_perplexity": test_perplexity,
        "total_consumed_tokens": total_consumed_tokens,
        "time": elapsed,
        "K": total_iters,
        "m": m,
        "peak_memory_max_MB": max_peak_mb
    }
    log_data.update(peak_memories)
    
    csv_logger.log(log_data)
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Total Tokens Consumed: {total_consumed_tokens}")
    print(f"final_test_perplexity: {test_perplexity}")
    print(f"Effective speed-up: {T/total_iters:.2f}x")
    print(f"m: {m}")
    print(f"Maximum GPU Peak Memory: {max_peak_mb:.2f} MB")
    
    return final_model

# =================================================================
# Naive Data Parallel
# =================================================================

def run_naive_dp_worker(rank, world_size, config, queues):
    device = torch.device(f"cuda:{rank}")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    model_wrapper = ModelFactory.create_model(config).to(device)
    model = model_wrapper
    
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_loader, _ = get_dataloaders(config, rank=rank, world_size=world_size)
    data_iter = iter(train_loader)
    
    while True:
        ret = queues[0].get()
        if ret is None:
            # Report peak memory before exiting
            peak_memory = torch.cuda.max_memory_allocated(device)
            queues[2].put((rank, peak_memory))
            return 
            
        params, step = ret
        model.set_params(params, clone=False)
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        batch = to_device(batch, device)
        num_tokens = count_real_tokens(batch, config.pad_token_id)
        
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()
            
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        grads = [p.grad for p in model.get_params()]
        
        metrics = {
            'loss': loss.item(),
            'perplexity': torch.exp(loss.detach()).item(),
            'num_tokens': num_tokens
        }
        
        queues[1].put((rank, grads, metrics))


def train_loop_naive_dp(config, model, train_loader, test_loader, csv_logger):
    world_size = config.device_count
    if world_size <= 1:
        print("Only one GPU found, falling back to serial training.")
        return train_loop_serial(config, model, train_loader, test_loader, csv_logger)

    print(f"Starting Naive Data Parallel with {world_size} devices...")
    
    mp.set_start_method('spawn', force=True)
    # Added 3rd queue
    queues = (mp.Queue(), mp.Queue(), mp.Queue()) 
    processes = []

    for rank in range(1, world_size):
        p = mp.Process(target=run_naive_dp_worker, args=(rank, world_size, config, queues))
        p.start()
        processes.append(p)

    device = torch.device("cuda:0")
    model = model.to(device)
    
    # Reset memory stats (Master)
    torch.cuda.reset_peak_memory_stats(device)
    
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )

    train_loader_master, _ = get_dataloaders(config, rank=0, world_size=world_size)
    data_iter = iter(train_loader_master)
    
    running_loss = 0.0
    total_consumed_tokens = 0
    start_time = time.time()
    best_ppl = float('inf')
    
    pbar = tqdm(total=config.max_steps, desc="Starting Naive DP training...")
    
    for step in range(config.max_steps):
        current_params = [p.data for p in model.get_params()]
        
        for _ in range(1, world_size):
            queues[0].put((current_params, step))
            
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader_master)
            batch = next(data_iter)
            
        batch = to_device(batch, device)
        num_tokens_local = count_real_tokens(batch, config.pad_token_id)
        
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        grad_accumulator = [p.grad if p.grad is not None else torch.zeros_like(p.data) for p in model.get_params()]
        
        step_loss = loss.item()
        step_tokens = num_tokens_local
        
        for _ in range(1, world_size):
            rank_w, grads_w, metrics_w = queues[1].get()
            
            for acc, g_w in zip(grad_accumulator, grads_w):
                if g_w is not None:
                    acc.add_(g_w.to(device))
            
            step_loss += metrics_w['loss']
            step_tokens += metrics_w['num_tokens']
            
        for acc in grad_accumulator:
            acc.div_(world_size)
            
        model.set_grads_from_grads(grad_accumulator)
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        avg_loss_step = step_loss / world_size
        running_loss += avg_loss_step
        total_consumed_tokens += step_tokens
        
        avg_loss_cumulative = running_loss / (step + 1)
        avg_perplexity = np.exp(avg_loss_cumulative)
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
        
        pbar.set_description(f'LR: {current_lr:.8f} | Loss: {avg_loss_cumulative:.4f} | PPL: {avg_perplexity:.2f} | Time: {elapsed_str}')
        
        csv_logger.log({
            "Iter": step + 1,
            "LR": current_lr,
            "Train_Loss": avg_loss_cumulative,
            "Train_Perplexity": avg_perplexity,
            "Original_Steps": step + 1,
            "num_tokens": total_consumed_tokens,
        })
        pbar.update(1)

        if (step + 1) % config.val_interval == 0:
            val_loss, val_ppl = evaluate_model(model, test_loader, device)
            if val_ppl < best_ppl:
                best_ppl = val_ppl
            pbar.write(f"\n[Validation at step {step + 1}]|{config.modelscope_name}|{config.dataset_name}|{config.optimizer_type}|{config.batch_size}|{config.learning_rate}| Test PPL: {val_ppl:.2f} | Best PPL: {best_ppl:.2f}")
            csv_logger.log({
                "Iter": step + 1,
                "Val_Loss": val_loss,
                "Val_Perplexity": val_ppl,
                "Best_Perplexity": best_ppl
            })
            csv_logger.save()

    pbar.close()
    
    # --- Collect Memory ---
    peak_memories = {}
    for _ in processes:
        queues[0].put(None)
        
    for _ in processes:
        rank, peak_mem = queues[2].get()
        peak_memories[f"GPU_{rank}_peak_memory_MB"] = peak_mem / (1024 ** 2)

    master_peak = torch.cuda.max_memory_allocated(device)
    peak_memories[f"GPU_0_peak_memory_MB"] = master_peak / (1024 ** 2)
    
    all_peaks = [v for v in peak_memories.values()]
    max_peak_mb = max(all_peaks) if all_peaks else 0.0

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    test_loss, test_ppl = evaluate_model(model, test_loader, device)
    
    log_data = {
        "final_test_loss": test_loss,
        "final_test_perplexity": test_ppl,
        "total_consumed_tokens": total_consumed_tokens,
        "time": elapsed,
        "peak_memory_max_MB": max_peak_mb
    }
    log_data.update(peak_memories)
    
    csv_logger.log(log_data)
    csv_logger.finish()
    print(f"\nNaive DP Training completed in {str(timedelta(seconds=int(elapsed)))}")
    print(f"\ntotal_consumed_tokens {total_consumed_tokens}")
    print(f"\nfinal_test_perplexity {test_ppl}")
    print(f"Maximum GPU Peak Memory: {max_peak_mb:.2f} MB")
    return model

# =================================================================
# DDP Implementation
# =================================================================

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def run_ddp_worker(rank, world_size, config, csv_logger_filename):
    ddp_setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    csv_logger = CSVLogger(csv_logger_filename) if rank == 0 else None
    
    train_loader, test_loader = get_dataloaders(config, rank=rank, world_size=world_size)
    
    model_wrapper = ModelFactory.create_model(config).to(device)
    ddp_model = DDP(model_wrapper.model, device_ids=[rank]) 
    
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(ddp_model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(ddp_model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.AdamW(ddp_model.parameters(), lr=config.learning_rate)
        
    model = ddp_model 

    scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
    start_time = time.time()
    running_loss = 0.0
    total_consumed_tokens = 0
    best_ppl = float('inf')
    
    if rank == 0:
        pbar = tqdm(total=config.max_steps, desc="Starting DDP training...")
        
    data_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        batch = to_device(batch, device)
        
        real_tokens = count_real_tokens(batch, config.pad_token_id)
        num_tokens_local = torch.tensor(real_tokens, device=device)
        
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_tokens_local, op=dist.ReduceOp.SUM)
        
        avg_loss_step = loss.item() / world_size
        total_tokens_step = num_tokens_local.item()
        
        running_loss += avg_loss_step
        total_consumed_tokens += total_tokens_step
        
        if rank == 0:
            avg_loss_cumulative = running_loss / (step + 1)
            avg_perplexity = np.exp(avg_loss_cumulative)
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            current_lr = scheduler.get_last_lr()[0]
            
            pbar.set_description(f'LR: {current_lr:.8f} | Loss: {avg_loss_cumulative:.4f} | PPL: {avg_perplexity:.2f} | Time: {elapsed_str}')
            
            csv_logger.log({
                "Iter": step + 1,
                "LR": current_lr,
                "Train_Loss": avg_loss_cumulative,
                "Train_Perplexity": avg_perplexity,
                "Original_Steps": step + 1,
                "num_tokens": total_consumed_tokens,
            })
            pbar.update(1)

            if (step + 1) % config.val_interval == 0:
                val_loss, val_ppl = evaluate_model(model_wrapper, test_loader, device)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                pbar.write(f"\n[Validation at step {step + 1}]|{config.modelscope_name}|{config.dataset_name}|{config.optimizer_type}|{config.batch_size}|{config.learning_rate}| Test PPL: {val_ppl:.2f} | Best PPL: {best_ppl:.2f}")
                csv_logger.log({
                    "Iter": step + 1,
                    "Val_Loss": val_loss,
                    "Val_Perplexity": val_ppl,
                    "Best_Perplexity": best_ppl
                })
                csv_logger.save()

    if rank == 0:
        pbar.close()
        
    # --- Capture and Aggregate Peak Memory ---
    peak_memory = torch.cuda.max_memory_allocated(device)
    # Convert to Tensor for all_reduce
    peak_tensor = torch.tensor(peak_memory, device=device)
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    max_peak_memory_mb = peak_tensor.item() / (1024 ** 2)
    
    if rank == 0:
        elapsed = time.time() - start_time
        test_loss, test_ppl = evaluate_model(model_wrapper, test_loader, device)
        
        csv_logger.log({
            "final_test_loss": test_loss,
            "final_test_perplexity": test_ppl,
            "total_consumed_tokens": total_consumed_tokens,
            "time": elapsed,
            "peak_memory_max_MB": max_peak_memory_mb
        })
        csv_logger.finish()
        print(f"\nDDP Training completed in {str(timedelta(seconds=int(elapsed)))}")
        print(f"Total Tokens Consumed: {total_consumed_tokens}")
        print(f"final_test_perplexity: {test_ppl}")
        print(f"Maximum GPU Peak Memory: {max_peak_memory_mb:.2f} MB")
        
    ddp_cleanup()

def train_loop_ddp(config, csv_logger):
    world_size = config.device_count
    if world_size < 2:
        print("DDP requires > 1 GPU. Falling back to Serial.")
        model = ModelFactory.create_model(config)
        train_loader, test_loader = get_dataloaders(config)
        return train_loop_serial(config, model, train_loader, test_loader, csv_logger)
        
    print(f"Spawning {world_size} DDP processes...")
    mp.spawn(run_ddp_worker,
             args=(world_size, config, csv_logger.filename),
             nprocs=world_size,
             join=True)

# =================================================================
# DeepSpeed Implementation
# =================================================================

def train_loop_deepspeed(config, csv_logger):

    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed is not installed.")

    # FIX: FORCE NCCL to avoid MPI errors
    deepspeed.init_distributed(dist_backend="nccl") 
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    
    ds_config = {
        "train_micro_batch_size_per_gpu": config.batch_size // world_size if world_size > 0 else config.batch_size, 
        "gradient_accumulation_steps": 1,
        "steps_per_print": 5,
        "optimizer": {
            "type": "Adam" if config.optimizer_type == 'adam' else ("AdamW" if config.optimizer_type == 'adamw' else "SGD"),
            "params": {
                "lr": config.learning_rate,
                "betas": [0.9, 0.999] if 'adam' in config.optimizer_type else [config.momentum, 0.999], 
            }
        },
        "fp16": { "enabled": False },
        "bf16": { "enabled": True },
        "zero_optimization": {
            "stage": config.stage, 
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True
        }
    }
    
    model_wrapper = ModelFactory.create_model(config)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=model_wrapper.model,
        model_parameters=model_wrapper.parameters(),
        config=ds_config
    )

    # Correctly handle DeepSpeed Optimizer wrapper for the scheduler
    real_optimizer = optimizer
    if hasattr(optimizer, "optimizer"):
        real_optimizer = optimizer.optimizer

    scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=real_optimizer, # Use the unwrapped optimizer here
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
    train_loader, test_loader = get_dataloaders(config, rank=rank, world_size=world_size)
    
    start_time = time.time()
    running_loss = 0.0
    total_consumed_tokens = 0
    best_ppl = float('inf')
    
    if rank == 0:
        pbar = tqdm(total=config.max_steps, desc=f"Starting DeepSpeed (ZeRO-{config.stage})...")
    
    data_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        batch = to_device(batch, model_engine.device)
        
        real_tokens = count_real_tokens(batch, config.pad_token_id)
        num_tokens_local = torch.tensor(real_tokens, device=model_engine.device)
        
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()
            
        loss = model_engine(batch['input_ids'], labels=batch['labels']).loss
        model_engine.backward(loss)
        model_engine.step()
        scheduler.step()
        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_tokens_local, op=dist.ReduceOp.SUM)
        
        avg_loss_step = loss.item() / world_size
        total_tokens_step = num_tokens_local.item()
        
        running_loss += avg_loss_step
        total_consumed_tokens += total_tokens_step
        
        if rank == 0:
            avg_loss_cumulative = running_loss / (step + 1)
            avg_perplexity = np.exp(avg_loss_cumulative)
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            current_lr = scheduler.get_last_lr()[0]
            
            pbar.set_description(f'LR: {current_lr:.8f} | Loss: {avg_loss_cumulative:.4f} | PPL: {avg_perplexity:.2f} | Time: {elapsed_str}')
            
            csv_logger.log({
                "Iter": step + 1,
                "LR": current_lr,
                "Train_Loss": avg_loss_cumulative,
                "Train_Perplexity": avg_perplexity,
                "Original_Steps": step + 1,
                "num_tokens": total_consumed_tokens
            })
            pbar.update(1)

            if rank == 0 and (step + 1) % config.val_interval == 0:
                model_wrapper.model = model_engine.module
                val_loss, val_ppl = evaluate_model(model_wrapper, test_loader, model_engine.device)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                pbar.write(f"\n[Validation at step {step + 1}]|{config.modelscope_name}|{config.dataset_name}|{config.optimizer_type}|{config.batch_size}|{config.learning_rate}| Test PPL: {val_ppl:.2f} | Best PPL: {best_ppl:.2f}")
                csv_logger.log({
                    "Iter": step + 1,
                    "Val_Loss": val_loss,
                    "Val_Perplexity": val_ppl,
                    "Best_Perplexity": best_ppl
                })
                csv_logger.save()

    if rank == 0:
        pbar.close()

    # --- Capture and Aggregate Peak Memory ---
    peak_memory = torch.cuda.max_memory_allocated(model_engine.device)
    peak_tensor = torch.tensor(peak_memory, device=model_engine.device)
    elapsed = time.time() - start_time
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    max_peak_memory_mb = peak_tensor.item() / (1024 ** 2)
    dist.barrier()
    
    
    # print("rank first ", rank)
    model_wrapper.model = model_engine.module
    test_loss, test_ppl = evaluate_model(model_wrapper, test_loader, model_engine.device)
    # print("rank second", rank)
    dist.barrier()
    if rank == 0:
        csv_logger.log({
            "final_test_loss": test_loss,
            "final_test_perplexity": test_ppl,
            "total_consumed_tokens": total_consumed_tokens,
            "time": elapsed,
            "peak_memory_max_MB": max_peak_memory_mb
        })
        print(f"\nDeepSpeed Training completed in {str(timedelta(seconds=int(elapsed)))}")
        print({
            "final_test_loss": test_loss,
            "final_test_perplexity": test_ppl,
            "total_consumed_tokens": total_consumed_tokens,
            "time": elapsed,
            "peak_memory_max_MB": max_peak_memory_mb
        })
    
# ==========================
# Serial Loop 
# ==========================
def train_loop_serial(config, model, train_loader, test_loader, csv_logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else: 
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
        
    start_time = time.time()
    running_loss = 0.0
    running_total = 0     
    total_consumed_tokens = 0 
    best_ppl = float('inf')
    
    pbar = tqdm(total=config.max_steps, desc="Starting serial training...")
    data_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        batch = to_device(batch, device)
        
        num_tokens = count_real_tokens(batch, config.pad_token_id)
        total_consumed_tokens += num_tokens

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        running_total += batch['input_ids'].size(0)
        
        avg_loss = running_loss / (step + 1)
        avg_perplexity = torch.exp(torch.tensor(avg_loss))
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
        current_lr = scheduler.get_last_lr()[0]
        
        pbar.set_description(f'LR: {current_lr:.8f} | Loss: {avg_loss:.4f} | PPL: {avg_perplexity:.2f} | Time: {elapsed_str}')
        
        csv_logger.log({
            "Iter": step + 1,
            "LR": current_lr,
            "Train_Loss": avg_loss,
            "Train_Perplexity": avg_perplexity,
            "Original_Steps": step + 1,
            "num_tokens": total_consumed_tokens 
        })
        pbar.update(1)

        if (step + 1) % config.val_interval == 0:
            val_loss, val_ppl = evaluate_model(model, test_loader, device)
            if val_ppl < best_ppl:
                best_ppl = val_ppl
            pbar.write(f"\n[Validation at step {step + 1}]|{config.modelscope_name}|{config.dataset_name}|{config.optimizer_type}|{config.batch_size}|{config.learning_rate}| Test PPL: {val_ppl:.2f} | Best PPL: {best_ppl:.2f}")
            csv_logger.log({
                "Iter": step + 1,
                "Val_Loss": val_loss,
                "Val_Perplexity": val_ppl,
                "Best_Perplexity": best_ppl
            })
            csv_logger.save()
    
    pbar.close()
    
    # --- Measure Peak Memory ---
    peak_memory_max_mb = 0.0
    if device.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_max_mb = peak_memory_bytes / (1024 ** 2)
        print(f"Single GPU Peak Memory: {peak_memory_max_mb:.2f} MB")
        
    elapsed = time.time() - start_time
    test_loss, test_perplexity = evaluate_model(model, test_loader, device)
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Total Tokens Consumed: {total_consumed_tokens}")
    print(f"final_test_perplexity: {test_perplexity}")
    
    csv_logger.log({
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "total_consumed_tokens": total_consumed_tokens, 
        "time":elapsed,
        "peak_memory_max_MB": peak_memory_max_mb
    })
    
    return model

# ==========================
# Main Execution Logic
# ==========================
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='ParaOpt Training')
    parser.add_argument('--device_count', type=int, help='Number of CUDA devices')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_steps', type=int, help='Total training steps')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--P', type=int, help='Window size for parallel algorithm')
    parser.add_argument('--threshold', type=float, help='Error threshold')
    parser.add_argument('--ema_decay', type=float, help='EMA decay for threshold')
    parser.add_argument('--adaptivity_type', type=str, choices=['mean', 'median'], help='Error computation type')
    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adam', 'adamw'], help='Optimizer type')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial', 'ddp', 'deepspeed', 'naive_dp'], help='Training mode')
    parser.add_argument('--modelscope_name', type=str, help='Model name')
    parser.add_argument('--local_model_dir', type=str, help='Path to local model')
    parser.add_argument('--dataset_name', type=str, choices=['local_wikitext'], help='Dataset')
    parser.add_argument('--local_dataset_dir', type=str, help='Path to local dataset')
    parser.add_argument('--max_length', type=int, help='Maximum sequence length')
    parser.add_argument('--output_csv', type=str, help='Output CSV file path')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by deepspeed/torchrun')
    parser.add_argument('--stage', type=int, default=1, help='deepspeed stage')
    # Added
    parser.add_argument('--val_interval', type=int, help='Validation interval')
    
    # Scheduler
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', 
                        choices=['linear', 'cosine', 'constant'], 
                        help='learning rate scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=100, help='number of warmup steps')

    if DEEPSPEED_AVAILABLE:
        parser = deepspeed.add_config_arguments(parser)
        
    return parser

def train_with_config(config=None, csv_logger=None):
    if config is None:
        config = Config()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if csv_logger is None:
        csv_logger = CSVLogger()
    
    print(f"Starting training in mode: {config.training_mode}")

    if config.training_mode.lower() == "parallel":
        train_loader, test_loader = get_dataloaders(config) 
        model = ModelFactory.create_model(config)
        train_loop_parallel(config, model, train_loader, test_loader, csv_logger)
        csv_logger.finish()
        
    elif config.training_mode.lower() == "ddp":
        train_loop_ddp(config, csv_logger)
        
    elif config.training_mode.lower() == "deepspeed":
        train_loop_deepspeed(config, csv_logger)
        if dist.get_rank() == 0:
             csv_logger.finish()

    elif config.training_mode.lower() == "naive_dp":
        model = ModelFactory.create_model(config)
        _, test_loader = get_dataloaders(config)
        train_loop_naive_dp(config, model, None, test_loader, csv_logger)
              
    else: # Serial
        train_loader, test_loader = get_dataloaders(config)
        model = ModelFactory.create_model(config)
        train_loop_serial(config, model, train_loader, test_loader, csv_logger)
        csv_logger.finish()

def main():
    parser = setup_arg_parser()
    args, unknown = parser.parse_known_args()
    
    config = Config()
    config.update_from_args(args)

    directory_path = os.path.join(
    "./PASO_standard/2026_01_08", config.modelscope_name,
        f"steps_{config.max_steps}_bs_{config.batch_size}_lr_{config.learning_rate}_optim_{config.optimizer_type}_sched_{config.lr_scheduler_type}_warmup_{config.warmup_steps}"
    )
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    if config.training_mode == "deepspeed":
        output_csv = directory_path + f"/{config.training_mode}_stage_{config.stage}.csv"
    else:
        output_csv = directory_path + f"/{config.training_mode}_{config.adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.ema_decay}.csv"
    
    config.checkpoint_dir = directory_path 

    csv_logger = CSVLogger(output_csv) if output_csv else CSVLogger()
    
    train_with_config(config, csv_logger)

if __name__ == "__main__":
    main()