import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, disable_caching
from modelscope import snapshot_download
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
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
from sklearn.metrics import accuracy_score, f1_score
import csv
from datetime import datetime

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
# CSV Logger Class (Unchanged)
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
    P = 8
    threshold = 1e-5
    ema_decay = 0.999
    adaptivity_type = 'median'
    
    # System configuration
    seed = 42
    device_count = torch.cuda.device_count()
    
    # Model and dataset configuration
    modelscope_name = 'gpt2'
    local_model_dir = "./model/gpt2"
    dataset_name = 'local_wikitext'
    local_dataset_dir = "./data/wikitext-2" 
    
    max_length = 512
    optimizer_type = 'adam'
    training_mode = 'parallel' 
    sweep_config_path = 'XXX.json'
    
    # Added to store pad token id globally
    pad_token_id = None 

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

# ==========================
# Utility Functions
# ==========================
def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def calculate_metrics(preds, labels, ignore_index=-100):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]
    if len(labels) == 0:
        return 0.0, 0.0
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return accuracy, f1

# ==========================
# Model Definition (Unchanged)
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
    if config.local_model_dir and os.path.exists(config.local_model_dir):
        model_dir = config.local_model_dir
    else:
        model_dir = snapshot_download(config.modelscope_name, revision='master',cache_dir="./model")
    
    if rank == 0 or rank is None:
        print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ==========================================
    # FIX: Store pad_token_id in config for counting
    # ==========================================
    config.pad_token_id = tokenizer.pad_token_id

    if rank == 0 or rank is None:
        print(f"Loading dataset: {config.dataset_name}...")
    disable_caching()

    if config.dataset_name == 'local_wikitext':
        dataset = load_wikitext_from_local(config.local_dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")

    def tokenize_function(examples, tokenizer, max_length):
        # 使用相同的文本作为输入和标签
        tokenized_inputs = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 添加labels字段
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        # 2. 这里的关键点：利用 attention_mask 将 padding 的部分设置为 -100
        # tokenizer 返回的 attention_mask 中，1 代表有效 token，0 代表 padding
        # 注意：因为是在 dataset.map 中 batched=True，这里的数据是 Tensor 或者是 List of Lists
        # 为了安全起见，我们使用 torch 操作（因为你在 return_tensors="pt"）
        
        # 如果 tokenizer 返回的是 Tensor (因为 return_tensors="pt")
        labels = tokenized_inputs["labels"]
        attention_mask = tokenized_inputs["attention_mask"]
        
        # 将 mask 为 0 的位置 (padding) 的 label 设为 -100
        # 这样 CrossEntropyLoss 就会自动忽略这些位置
        labels[attention_mask == 0] = -100
        
        # 重新赋值回去
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    def preprocess(examples):
        return tokenize_function(examples, tokenizer, config.max_length)
        
    column_names = dataset["train"].column_names
    tokenized_datasets = dataset.map(
        preprocess,
        batched=True,
        remove_columns=column_names,
    )
    tokenized_datasets.set_format("torch")


    train_sampler = None
    test_sampler = None
    if rank is not None and world_size is not None:
        train_sampler = DistributedSampler(tokenized_datasets["train"], num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(tokenized_datasets['validation'], num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = torch.utils.data.DataLoader(
            tokenized_datasets["train"],
            batch_size=config.batch_size//world_size,
            shuffle=(train_sampler is None), 
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
    else: 
        train_loader = torch.utils.data.DataLoader(
            tokenized_datasets["train"],
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
    val_split = 'validation' if 'validation' in tokenized_datasets else 'test'
    test_loader = torch.utils.data.DataLoader(
        tokenized_datasets[val_split],
        batch_size=config.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return train_loader, test_loader

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            accuracy, f1 = calculate_metrics(preds, batch['labels'])
            
            total_loss += loss.item()
            total_perplexity += torch.exp(loss).item()
            total_accuracy += accuracy
            total_f1 += f1
            total_batches += 1
    
    if total_batches == 0: return 0,0,0,0
    
    avg_loss = total_loss / total_batches
    avg_perplexity = total_perplexity / total_batches 
    avg_accuracy = total_accuracy / total_batches
    avg_f1 = total_f1 / total_batches
    model.train()
    return avg_loss, avg_perplexity, avg_accuracy, avg_f1

# =================================================================
# PASO (Parallel) and Worker Logic 
# =================================================================
def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())

# FIX: Added pad_token_id argument
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
    
    # FIX: Use helper to count real tokens
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
        'accuracy': 0.0,
        'f1_score': 0.0,
        'data_iter': data_iter,
        'batch_size': batch['input_ids'].size(0),
        'num_tokens': num_tokens_in_batch # Now real tokens
    }

# FIX: Added pad_token_id argument
def run_worker(model, optimizer, dataloader, queues, device, seed_offset, pad_token_id):
    data_iter = iter(dataloader)
    while True:
        ret = queues[0].get()
        if ret is None:
            return
        params, step = ret
        model.set_params(params, clone=False)
        # Pass pad_token_id to take_step
        res = take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset, pad_token_id)
        data_iter = res['data_iter']
        grads = [p.grad for p in model.get_params()]
        queues[1].put((grads, step, {'loss': res['loss'], 
                                    'perplexity': res['perplexity'],
                                    'accuracy': res['accuracy'],
                                    'f1_score': res['f1_score'],
                                    'batch_size': res['batch_size'],
                                    'num_tokens': res['num_tokens']}))

def train_loop_parallel(config, model, train_loader, test_loader, csv_logger):
    if config.device_count <= 1:
        print("Only one GPU found, falling back to serial training.")
        return train_loop_serial(config, model, train_loader, test_loader, csv_logger)

    mp.set_start_method('spawn', force=True)
    queues = (mp.Queue(), mp.Queue())
    processes = []

    for rank in range(1, config.device_count):
        worker_device = torch.device(f"cuda:{rank}")
        worker_model = ModelFactory.create_model(config).to(worker_device)
        if config.optimizer_type.lower() == 'sgd':
            worker_optimizer = optim.SGD(worker_model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer_type.lower() == 'adam':
            worker_optimizer = optim.Adam(worker_model.parameters(), lr=config.learning_rate)
        else:
            worker_optimizer = optim.AdamW(worker_model.parameters(), lr=config.learning_rate)
        
        # FIX: Pass config.pad_token_id to worker
        p = mp.Process(target=run_worker, args=(worker_model, worker_optimizer, train_loader, queues, worker_device, config.seed, config.pad_token_id))
        p.start()
        processes.append(p)

    device = torch.device("cuda:0")
    model = model.to(device)
    T = config.max_steps
    P = min(config.P, T)
    thresh = config.threshold
    models = [None] * (T + 1)
    optimizers = [None] * (T + 1)
    
    begin_idx, end_idx = 0, P
    total_iters, total_Grad = 0, 0
    running_loss = 0.0
    running_perplexity = 0.0
    total_consumed_tokens = 0

    for step in range(P + 1):
        models[step] = model.clone(device)
        if config.optimizer_type.lower() == 'sgd':
            optimizers[step] = optim.SGD(models[step].parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer_type.lower() == 'adam':
            optimizers[step] = optim.Adam(models[step].parameters(), lr=config.learning_rate)
        else:
            optimizers[step] = optim.AdamW(models[step].parameters(), lr=config.learning_rate)

    pbar = tqdm(total=T, desc="Starting parallel training...")
    total_comm_time = 0.0
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()

    while begin_idx < T:
        start_time_per_iter = time.time()
        parallel_len = end_idx - begin_idx
        pred_f = [None] * parallel_len
        metrics = [None] * parallel_len
        comm_start_time = time.time()
        for i in range(parallel_len):
            step = begin_idx + i
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))

        for _ in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _grads
            metrics[_i] = _metrics
        time_comm_per_iter = time.time() - comm_start_time
        total_comm_time += time_comm_per_iter

        rollout_model = models[begin_idx]
        rollout_optimizer = optimizers[begin_idx]
        ind, errors_all = None, 0

        for i in range(parallel_len):
            step = begin_idx + i
            rollout_model.set_grads_from_grads(pred_f[i])
            rollout_optimizer.step()
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
        time_per_iter = time.time() - start_time_per_iter
        time_comp_per_iter = time_per_iter - time_comm_per_iter

        avg_loss = running_loss / begin_idx if begin_idx > 0 else 0
        avg_perplexity = torch.exp(torch.tensor(avg_loss))
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        pbar.set_description(f'Loss: {avg_loss:.4f} |cur loss:{cur_loss:.4f}| PPL: {avg_perplexity:.2f} |cur PPL: {cur_perplexity:.2f} | Time: {elapsed_str}')          
        csv_logger.log({
            "Iter": total_iters,
            "Train_Loss": avg_loss,
            "Train_Perplexity": avg_perplexity,
            "Original_Steps": begin_idx,
            "num_tokens": total_consumed_tokens,
            "time_per_iter": time_per_iter,
            "time_comm_per_iter": time_comm_per_iter,
            "time_comp_per_iter": time_comp_per_iter
        })

    end_event.record()
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event) / 1000.0
    pbar.close()

    for _ in processes:
        queues[0].put(None)
    for p in processes:
        p.join()
        
    final_model = models[T]
    del models, optimizers
    test_loss, test_perplexity, test_accuracy, test_f1 = evaluate_model(final_model, test_loader, device)
    upper_bound = total_iters * P
    m = total_Grad / T
    
    csv_logger.log({
        # "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        # "final_test_accuracy": test_accuracy,
        # "final_test_f1": test_f1,
        "total_consumed_tokens": total_consumed_tokens,
        "time": elapsed,
        # "time_comm": total_comm_time,
        # "time_step": elapsed - total_comm_time,
        "K": total_iters,
        # "G": total_Grad,
        # "upper_bound": upper_bound,
        "m": m
    })
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Total Tokens Consumed: {total_consumed_tokens}")
    print(f"final_test_perplexity: {test_perplexity}")
    return final_model

# =================================================================
# Naive Data Parallel (New Implementation)
# =================================================================

def run_naive_dp_worker(rank, world_size, config, queues):
    """
    Worker process for Naive Data Parallel.
    Uses DistributedSampler for data partitioning.
    Uses MP Queues for communication (Params Down, Grads Up).
    """
    device = torch.device(f"cuda:{rank}")
    
    # 1. Create local model copy
    model_wrapper = ModelFactory.create_model(config).to(device)
    model = model_wrapper
    
    # 2. Create Optimizer (needed to clear grad, though step is virtually done by master logic)
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 3. Get Data Loader with DistributedSampler
    train_loader, _ = get_dataloaders(config, rank=rank, world_size=world_size)
    data_iter = iter(train_loader)
    
    while True:
        # A. Wait for Parameters from Master
        ret = queues[0].get()
        if ret is None:
            return # Poison pill
            
        params, step = ret
        
        # B. Update local model parameters
        model.set_params(params, clone=False)
        
        # C. Compute Gradients
        # Note: We reuse take_step logic but we don't need the step return value for logic control
        # ensuring seeds are aligned helps reproducibility
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        batch = to_device(batch, device)
        
        # Count tokens
        num_tokens = count_real_tokens(batch, config.pad_token_id)
        
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()
            
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # D. Send Gradients and Metrics back to Master
        grads = [p.grad for p in model.get_params()]
        
        metrics = {
            'loss': loss.item(),
            'perplexity': torch.exp(loss.detach()).item(),
            'num_tokens': num_tokens
        }
        
        queues[1].put((rank, grads, metrics))


def train_loop_naive_dp(config, model, train_loader, test_loader, csv_logger):
    """
    Naive Data Parallel Master Loop.
    Rank 0 acts as Master AND Worker 0.
    Rank 1..N are purely workers.
    Communication is purely via Queues (no NCCL/TorchDist).
    """
    world_size = config.device_count
    if world_size <= 1:
        print("Only one GPU found, falling back to serial training.")
        return train_loop_serial(config, model, train_loader, test_loader, csv_logger)

    print(f"Starting Naive Data Parallel with {world_size} devices...")
    
    # Setup Multiprocessing
    mp.set_start_method('spawn', force=True)
    queues = (mp.Queue(), mp.Queue()) # 0: Params (Master->Workers), 1: Grads (Workers->Master)
    processes = []

    # Start Workers (Rank 1 to N-1)
    for rank in range(1, world_size):
        # Note: run_naive_dp_worker will instantiate its own model/dataset
        p = mp.Process(target=run_naive_dp_worker, args=(rank, world_size, config, queues))
        p.start()
        processes.append(p)

    # Setup Master (Rank 0)
    device = torch.device("cuda:0")
    model = model.to(device)
    
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Master Dataloader (Rank 0)
    # Note: train_loader passed in main is already configured for rank 0 if config isn't parallel
    # But strictly speaking, we should ensure it uses DistributedSampler for Rank 0
    # Re-getting loader to be safe and consistent with workers
    train_loader_master, _ = get_dataloaders(config, rank=0, world_size=world_size)
    data_iter = iter(train_loader_master)
    
    running_loss = 0.0
    total_consumed_tokens = 0
    start_time = time.time()
    
    pbar = tqdm(total=config.max_steps, desc="Starting Naive DP training...")
    
    for step in range(config.max_steps):
        current_params = [p.data for p in model.get_params()]
        
        # 1. Broadcast Parameters to all workers
        for _ in range(1, world_size):
            queues[0].put((current_params, step))
            
        # 2. Compute Gradients Locally (Rank 0)
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
        
        # Local Gradients
        grad_accumulator = [p.grad if p.grad is not None else torch.zeros_like(p.data) for p in model.get_params()]
        
        # Track metrics
        step_loss = loss.item()
        step_tokens = num_tokens_local
        
        # 3. Receive Gradients from Workers
        for _ in range(1, world_size):
            rank_w, grads_w, metrics_w = queues[1].get()
            
            # Aggregate Gradients
            for acc, g_w in zip(grad_accumulator, grads_w):
                if g_w is not None:
                    acc.add_(g_w.to(device))
            
            # Aggregate Metrics
            step_loss += metrics_w['loss']
            step_tokens += metrics_w['num_tokens']
            
        # 4. Average Gradients
        for acc in grad_accumulator:
            acc.div_(world_size)
            
        # 5. Apply Gradients to Master Model
        model.set_grads_from_grads(grad_accumulator)
        optimizer.step()
        
        # 6. Logging
        avg_loss_step = step_loss / world_size
        running_loss += avg_loss_step
        total_consumed_tokens += step_tokens
        
        if (step + 1) % 5 == 0:
            avg_loss_cumulative = running_loss / (step + 1)
            avg_perplexity = np.exp(avg_loss_cumulative)
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            
            pbar.set_description(f'Loss: {avg_loss_cumulative:.4f} | PPL: {avg_perplexity:.2f} | Time: {elapsed_str}')
            
            csv_logger.log({
                "Iter": step + 1,
                "Train_Loss": avg_loss_cumulative,
                "Train_Perplexity": avg_perplexity,
                "Original_Steps": step + 1,
                "num_tokens": total_consumed_tokens,
                "time_per_iter": elapsed / (step + 1)
            })
        pbar.update(1)

    pbar.close()
    
    # Cleanup Workers
    for _ in processes:
        queues[0].put(None)
    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    test_loss, test_ppl, test_acc, test_f1 = evaluate_model(model, test_loader, device)
    
    csv_logger.log({
        "final_test_loss": test_loss,
        "final_test_perplexity": test_ppl,
        "final_test_accuracy": test_acc,
        "final_test_f1": test_f1,
        "total_consumed_tokens": total_consumed_tokens,
        "time": elapsed
    })
    csv_logger.finish()
    print(f"\nNaive DP Training completed in {str(timedelta(seconds=int(elapsed)))}")
    print(f"\ntotal_consumed_tokens {total_consumed_tokens}")
    print(f"\nfinal_test_perplexity {test_ppl}")
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
    
    csv_logger = CSVLogger(csv_logger_filename) if rank == 0 else None
    
    train_loader, test_loader = get_dataloaders(config, rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}")
    model_wrapper = ModelFactory.create_model(config).to(device)
    ddp_model = DDP(model_wrapper.model, device_ids=[rank]) 
    
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(ddp_model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(ddp_model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.AdamW(ddp_model.parameters(), lr=config.learning_rate)
        
    model = ddp_model 
    
    start_time = time.time()
    running_loss = 0.0
    total_consumed_tokens = 0
    
    if rank == 0:
        pbar = tqdm(total=config.max_steps, desc="Starting DDP training...")
        
    data_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            # train_loader.sampler.set_epoch(step) # FIX: Added set_epoch for correctness
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        batch = to_device(batch, device)
        
        # FIX: Count real tokens
        real_tokens = count_real_tokens(batch, config.pad_token_id)
        num_tokens_local = torch.tensor(real_tokens, device=device)
        
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_tokens_local, op=dist.ReduceOp.SUM)
        
        avg_loss_step = loss.item() / world_size
        total_tokens_step = num_tokens_local.item()
        
        running_loss += avg_loss_step
        total_consumed_tokens += total_tokens_step
        
        if rank == 0:
            if (step + 1) % 5 == 0:
                avg_loss_cumulative = running_loss / (step + 1)
                avg_perplexity = np.exp(avg_loss_cumulative)
                elapsed = time.time() - start_time
                elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
                
                pbar.set_description(f'Loss: {avg_loss_cumulative:.4f} | PPL: {avg_perplexity:.2f} | Time: {elapsed_str}')
                
                csv_logger.log({
                    "Iter": step + 1,
                    "Train_Loss": avg_loss_cumulative,
                    "Train_Perplexity": avg_perplexity,
                    "Original_Steps": step + 1,
                    "num_tokens": total_consumed_tokens,
                    "time_per_iter": elapsed / (step + 1) 
                })
            pbar.update(1)
            
    if rank == 0:
        pbar.close()
        elapsed = time.time() - start_time
        test_loss, test_ppl, test_acc, test_f1 = evaluate_model(model_wrapper, test_loader, device)
        
        csv_logger.log({
            "final_test_loss": test_loss,
            "final_test_perplexity": test_ppl,
            "final_test_accuracy": test_acc,
            "final_test_f1": test_f1,
            "total_consumed_tokens": total_consumed_tokens,
            "time": elapsed
        })
        csv_logger.finish()
        print(f"\nDDP Training completed in {str(timedelta(seconds=int(elapsed)))}")
        print(f"Total Tokens Consumed: {total_consumed_tokens}")
        print(f"final_test_perplexity: {test_ppl}")
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
    world_size = dist.get_world_size()
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed is not installed.")
    ds_config = {
        "train_batch_size": config.batch_size * config.device_count, 
        "train_micro_batch_size_per_gpu": config.batch_size // world_size if world_size > 0 else config.batch_size, # Fix for single gpu debug
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
            "stage": 2, 
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
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Note: get_dataloaders populates config.pad_token_id
    train_loader, test_loader = get_dataloaders(config, rank=rank, world_size=world_size)
    
    start_time = time.time()
    running_loss = 0.0
    total_consumed_tokens = 0
    
    if rank == 0:
        pbar = tqdm(total=config.max_steps, desc="Starting DeepSpeed (ZeRO-2)...")
    
    data_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            # train_loader.sampler.set_epoch(step)
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        batch = to_device(batch, model_engine.device)
        
        # FIX: Count real tokens
        real_tokens = count_real_tokens(batch, config.pad_token_id)
        num_tokens_local = torch.tensor(real_tokens, device=model_engine.device)
        
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()
            
        loss = model_engine(batch['input_ids'], labels=batch['labels']).loss
        model_engine.backward(loss)
        model_engine.step()
        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_tokens_local, op=dist.ReduceOp.SUM)
        
        avg_loss_step = loss.item() / world_size
        total_tokens_step = num_tokens_local.item()
        
        running_loss += avg_loss_step
        total_consumed_tokens += total_tokens_step
        
        if rank == 0:
            if (step + 1) % 5 == 0:
                avg_loss_cumulative = running_loss / (step + 1)
                avg_perplexity = np.exp(avg_loss_cumulative)
                elapsed = time.time() - start_time
                elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
                pbar.set_description(f'Loss: {avg_loss_cumulative:.4f} | PPL: {avg_perplexity:.2f} | Time: {elapsed_str}')
                
                csv_logger.log({
                    "Iter": step + 1,
                    "Train_Loss": avg_loss_cumulative,
                    "Train_Perplexity": avg_perplexity,
                    "Original_Steps": step + 1,
                    "num_tokens": total_consumed_tokens,
                    "time_per_iter": elapsed / (step + 1)
                })
            pbar.update(1)
            
    if rank == 0:
        pbar.close()
        elapsed = time.time() - start_time
        model_wrapper.model = model_engine.module
        test_loss, test_ppl, test_acc, test_f1 = evaluate_model(model_wrapper, test_loader, model_engine.device)
        
        csv_logger.log({
            "final_test_loss": test_loss,
            "final_test_perplexity": test_ppl,
            "final_test_accuracy": test_acc,
            "final_test_f1": test_f1,
            "total_consumed_tokens": total_consumed_tokens,
            "time": elapsed
        })
        print(f"\nDeepSpeed Training completed in {str(timedelta(seconds=int(elapsed)))}")
        print({
            # "final_test_loss": test_loss,
            "final_test_perplexity": test_ppl,
            # "final_test_accuracy": test_acc,
            # "final_test_f1": test_f1,
            "total_consumed_tokens": total_consumed_tokens,
            "time": elapsed
        })

# ==========================
# Serial Loop 
# ==========================
def train_loop_serial(config, model, train_loader, test_loader, csv_logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else: 
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
    start_time = time.time()
    running_loss = 0.0
    running_total = 0    
    total_consumed_tokens = 0 

    pbar = tqdm(total=config.max_steps, desc="Starting serial training...")
    data_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        batch = to_device(batch, device)
        
        # FIX: Count real tokens
        num_tokens = count_real_tokens(batch, config.pad_token_id)
        total_consumed_tokens += num_tokens

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_total += batch['input_ids'].size(0)
        
        if (step + 1) % 5 == 0:
            avg_loss = running_loss / (step + 1)
            avg_perplexity = torch.exp(torch.tensor(avg_loss))
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            
            pbar.set_description(f'Loss: {avg_loss:.4f} | PPL: {avg_perplexity:.2f} | Time: {elapsed_str}')
            
            csv_logger.log({
                "Iter": step + 1,
                "Train_Loss": avg_loss,
                "Train_Perplexity": avg_perplexity,
                "Original_Steps": step + 1,
                "num_tokens": total_consumed_tokens 
            })
        pbar.update(1)
    
    pbar.close()
    elapsed = time.time() - start_time
    test_loss, test_perplexity, test_accuracy, test_f1 = evaluate_model(model, test_loader, device)
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Total Tokens Consumed: {total_consumed_tokens}")
    print(f"final_test_perplexity: {test_perplexity}")
    csv_logger.log({
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "final_test_accuracy": test_accuracy,
        "final_test_f1": test_f1,
        "total_consumed_tokens": total_consumed_tokens, 
        "time":elapsed
    })
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]

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
        # For Naive DP, dataloaders are handled inside the loop/workers to ensure correct sampler setup per process
        # We create a dummy model wrapper here for consistency if needed, but train_loop_naive_dp instantiates its own
        model = ModelFactory.create_model(config)
        # Test loader is needed for evaluation at end
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
    "./PASO/new_exp_results", config.modelscope_name,
        f"steps_{config.max_steps}_bs_{config.batch_size}_lr_{config.learning_rate}_optim_{config.optimizer_type}"
    )
    output_csv = directory_path + f"/{config.training_mode}_{config.adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.ema_decay}.csv"
    
    csv_logger = CSVLogger(output_csv) if output_csv else CSVLogger()
    
    train_with_config(config, csv_logger)

if __name__ == "__main__":
    main()