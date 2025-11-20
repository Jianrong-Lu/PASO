#!/usr/bin/env python
# -*- coding: utf-8 -*-



# ==========================
# 1. Imports
# ==========================
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, disable_caching
from modelscope import snapshot_download
from modelscope import AutoTokenizer as ModelScopeAutoTokenizer # Explicit naming to avoid conflicts
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp
from tqdm import tqdm
import time
import argparse
from datetime import timedelta, datetime
import numpy as np
import copy
import os
import json
from sklearn.metrics import accuracy_score, f1_score
import csv

# from torchmetrics.functional import accuracy, f1_score # Keep commented as original code not activated

# ==========================
# 2. CSV Logger Class
# ==========================
class CSVLogger:
    """A simple CSV logger for saving metrics during training."""
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.csv"
        
        self.filename = filename
        self.fieldnames = set()
        self.rows = []
        directory = os.path.dirname(self.filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
    def log(self, data):
        """Log a new row of data (a dictionary)."""
        # Dynamically add new field names
        for key in data.keys():
            if key not in self.fieldnames:
                self.fieldnames.add(key)
        
        self.rows.append(data)
        
    def save(self):
        """Write all buffered data to CSV file."""
        if not self.rows:
            return
            
        with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(list(self.fieldnames)))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
                
    def finish(self):
        """Save and close the log."""
        self.save()

# ==========================
# 3. Configuration Class
# ==========================
class Config:
    """Training configuration parameters."""
    # Training parameters
    batch_size = 8
    num_epochs = 1
    max_steps = 1000
    learning_rate = 6e-5
    momentum = 0.9

    # Acceleration configuration (ParaSolver/PASO)
    P = 7  # Window size
    threshold = 1e-5  # Error threshold
    ema_decay = 0.9  # EMA decay for threshold
    adaptivity_type = 'mean'  # 'mean' or 'median'
    
    # System configuration
    seed = 42
    device_count = torch.cuda.device_count()
    
    # Model and dataset configuration
    # Note: These are best overridden via command line arguments
    modelscope_name = 'openai-community/gpt2'
    local_model_dir = "./model/gpt2"
    dataset_name = 'local_wikitext'
    local_dataset_dir = "./data/wikitext-2" 
    output_dir = "./new_exp_results"
    
    max_length = 512  # Tokenizer maximum sequence length
    optimizer_type = 'adam'
    training_mode = 'serial'
    sweep_config_path = 'XXX.json' # Seems unused

    def update_from_args(self, args):
        """Update configuration from command line argparse namespace."""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

# ==========================
# 4. Utility & Metric Functions
# ==========================
def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def calculate_metrics(preds, labels, ignore_index=-100):

    # Flatten predictions and labels
    preds_flat = preds.cpu().flatten()
    labels_flat = labels.cpu().flatten()
    
    # Create a mask to ignore padding tokens
    mask = (labels_flat != ignore_index)
    
    # Apply mask
    active_preds = preds_flat[mask]
    active_labels = labels_flat[mask]
    
    if len(active_labels) == 0:
        # If no valid labels
        return 0.0, 0.0
        
    # Calculate accuracy
    accuracy = accuracy_score(active_labels, active_preds)
    
    # Calculate F1 score (macro average)
    f1 = f1_score(active_labels, active_preds, average='macro', zero_division=0)
    
    return accuracy, f1

def optimizer_state_clone(optimizer_from, optimizer_to):
    """Copy the state of one optimizer to another."""
    optimizer_to.load_state_dict(optimizer_from.state_dict())

# ==========================
# 5. Data Handling
# ==========================
def load_wikitext_from_local(path):
    """Load wikitext dataset from local directory."""
    if not os.path.exists(path):
        raise ValueError(f"Dataset path {path} does not exist. Please check `local_dataset_dir`.")
    
    data_files = {
        "train": os.path.join(path, "wiki.train.tokens"),
        "validation": os.path.join(path, "wiki.valid.tokens"),
        "test": os.path.join(path, "wiki.test.tokens")
    }
    
    # Filter out non-existent files
    existing_files = {split: file for split, file in data_files.items() if os.path.exists(file)}
    if not existing_files:
        raise FileNotFoundError(f"No valid data files found in {path}. Expected wiki.train.tokens, etc.")
        
    print(f"Loading dataset from files: {existing_files}")
    dataset = load_dataset("text", data_files=existing_files)
    return dataset

def _tokenize_function(examples, tokenizer, max_length):
    """Helper tokenize function for dataset map."""
    # Use the same text as input and labels
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Add labels field (standard practice for Causal LM)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    return tokenized_inputs

def get_dataloaders(config):
    """Load and prepare the specified dataset and return DataLoaders."""
    # Determine model directory for tokenizer
    if config.local_model_dir and os.path.exists(config.local_model_dir):
        model_dir = config.local_model_dir
    else:
        print(f"Downloading tokenizer for '{config.modelscope_name}'...")
        model_dir = snapshot_download(
            config.modelscope_name, 
            revision='master',
            cache_dir=os.path.dirname(config.local_model_dir) # Use parent directory of local_dir in config as cache
        )
        
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token not set. Setting to eos_token.")

    print(f"Loading dataset: {config.dataset_name}...")
    disable_caching() # Recommended for multiprocessing

    if config.dataset_name == 'local_wikitext':
        dataset = load_wikitext_from_local(config.local_dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}. Please use 'local_wikitext'.")

    print("Tokenizing dataset...")
    column_names = dataset["train"].column_names
    
    # Use functools.partial to pass additional parameters
    from functools import partial
    preprocess_fn = partial(_tokenize_function, tokenizer=tokenizer, max_length=config.max_length)
    
    tokenized_datasets = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
        num_proc=16 # Can appropriately increase processes to speed up preprocessing
    )
    tokenized_datasets.set_format("torch")

    train_loader = torch.utils.data.DataLoader(
        tokenized_datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_split = 'validation' if 'validation' in tokenized_datasets else 'test'
    if val_split not in tokenized_datasets:
         raise ValueError("No 'validation' or 'test' split found in the dataset.")
         
    test_loader = torch.utils.data.DataLoader(
        tokenized_datasets[val_split],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader

# ==========================
# 6. Model Definition
# ==========================
class ModelWrapper(nn.Module):
    """Universal wrapper for models to provide consistent interface."""
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        # Disable cache for data parallelism
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def clone(self, device, other=None):
        """Create a deep copy of the model on the specified device."""
        if other is None:
            # Create a new instance
            new_model_instance = copy.deepcopy(self.model)
            other = ModelWrapper(new_model_instance).to(device)
        else:
            # If target model provided, copy parameter state
            with torch.no_grad():
                for param_to, param_from in zip(other.parameters(), self.parameters()):
                    param_to.data = param_from.data.clone()
        return other

    def get_params(self):
        """Return list of model parameters."""
        return list(self.parameters())

    def set_params(self, params, clone=True):
        """Set model parameters from given parameter list."""
        my_params = self.get_params()
        for p, q in zip(my_params, params):
            if clone:
                p.data = q.data.clone().to(p.device)
            else:
                p.data = q.to(p.device)

    def set_grads_from_grads(self, grads):
        """Set gradients for the model."""
        my_params = self.get_params()
        for p, grad in zip(my_params, grads):
            if grad is not None:
                p.grad = grad.to(p.device)

    def compute_error_from_model(self, other):
        """Calculate mean squared error between this model and another model's parameters."""
        my_params = self.get_params()
        other_params = other.get_params()
        with torch.no_grad():
            error, total_num = 0.0, 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p - q).pow(2).item()
                total_num += np.prod(list(q.shape))
            return error / total_num * 1e6

class ModelFactory:
    """Factory for creating specified LLM models using ModelScope."""
    @staticmethod
    def create_model(config, device_map=None):
        """
        Create model, prioritizing local directory.
        device_map: Key parameter. For parallel training, should be None.
        """
        if config.local_model_dir and os.path.exists(config.local_model_dir):
            model_dir = config.local_model_dir
            print(f"Loading model from local path: {model_dir}")
        else:
            print(f"Downloading model '{config.modelscope_name}' from ModelScope...")
            model_dir = snapshot_download(
                config.modelscope_name, 
                revision='master',
                cache_dir=os.path.dirname(config.local_model_dir) # Use parent directory of local_dir in config as cache
            )
            print(f"Model downloaded to: {model_dir}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto", # Use "auto" or torch.bfloat16
            trust_remote_code=True,
            device_map=device_map # Allow caller to control
        )

        return ModelWrapper(model)

# ==========================
# 7. Core Training Logic
# ==========================

def take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset):
    """Execute a single training step (forward and backward pass)."""
    # Set reproducible seed for this specific step
    np.random.seed(step + seed_offset)
    torch.manual_seed(step + seed_offset)
    torch.cuda.manual_seed(step + seed_offset)
    
    try:
        batch = next(data_iter)
    except StopIteration:
        # print(f"Data iterator reset at step {step}") # For debugging
        data_iter = iter(dataloader) 
        batch = next(data_iter)
        
    batch = to_device(batch, device)
    
    # Ensure labels field exists
    if 'labels' not in batch:
        batch['labels'] = batch['input_ids'].clone()   

    optimizer.zero_grad()
    outputs = model(**batch)
    
    if outputs.loss is None:
        raise ValueError("Model did not return loss value. Check input and model configuration.")
    
    loss = outputs.loss
    loss.backward()
    
    # Calculate metrics
    perplexity = torch.exp(loss.detach()) if loss is not None else torch.tensor(float('inf'))
    
    with torch.no_grad():
        preds = torch.argmax(outputs.logits, dim=-1)
        accuracy, f1 = calculate_metrics(preds, batch['labels'])
        
    return {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy,
        'f1_score': f1,
        'data_iter': data_iter,
        'batch_size': batch['input_ids'].size(0)
    }

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test set."""
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_batches = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = to_device(batch, device)
            
            if 'labels' not in batch:
                batch['labels'] = batch['input_ids'].clone()
                
            outputs = model(**batch)
            loss = outputs.loss
            
            if loss is not None:
                perplexity = torch.exp(loss)
                total_loss += loss.item()
                total_perplexity += perplexity.item()
            
            # Calculate accuracy and F1 score
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            accuracy, f1 = calculate_metrics(preds, batch['labels'])
            
            total_accuracy += accuracy
            total_f1 += f1
            total_batches += 1
    
    if total_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / total_batches
    avg_perplexity = total_perplexity / total_batches
    avg_accuracy = total_accuracy / total_batches
    avg_f1 = total_f1 / total_batches
    
    model.train() # Set model back to training mode
    return avg_loss, avg_perplexity, avg_accuracy, avg_f1

# ==========================
# 8. Training Orchestration
# ==========================

def run_worker(model, optimizer, dataloader, queues, device, seed_offset):
    """Function executed by each parallel worker process."""
    data_iter = iter(dataloader)
    while True:
        ret = queues[0].get()
        if ret is None: # Termination signal
            return
            
        params, step = ret
        model.set_params(params, clone=False)
        
        res = take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset)
        data_iter = res['data_iter']
        
        grads = [p.grad.clone() for p in model.get_params()] # Clone gradients for safe sending
        
        queues[1].put((grads, step, {
            'loss': res['loss'], 
            'perplexity': res['perplexity'],
            'accuracy': res['accuracy'],
            'f1_score': res['f1_score'],
            'batch_size': res['batch_size']
        }))

def _create_optimizer(model, config):
    """Create optimizer based on configuration."""
    if config.optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    else: # Default 'adamw'
        return optim.AdamW(model.parameters(), lr=config.learning_rate)

def train_loop_parallel(config, model, train_loader, test_loader, csv_logger):
    """Main parallel training loop using fixed-point iteration algorithm."""
    if config.device_count <= 1:
        print("Only one GPU found, falling back to serial training.")
        return train_loop_serial(config, model, train_loader, test_loader, csv_logger)

    mp.set_start_method('spawn', force=True)
    queues = (mp.Queue(), mp.Queue())
    processes = []

    # Create models and optimizers for worker processes
    for rank in range(1, config.device_count):
        worker_device = torch.device(f"cuda:{rank}")
        # **Important**:
        # Worker processes create their own model instances on their own devices
        # Here device_map=None is correct
        worker_model = ModelFactory.create_model(config, device_map=None).to(worker_device)
        worker_optimizer = _create_optimizer(worker_model, config)
        
        p = mp.Process(target=run_worker, args=(worker_model, worker_optimizer, train_loader, queues, worker_device, config.seed))
        p.start()
        processes.append(p)

    # Main process setup
    device = torch.device("cuda:0")
    model = model.to(device)
    
    T = config.max_steps
    P = min(config.P, T, config.device_count - 1) # Window size cannot exceed worker count
    if P != config.P:
        print(f"Warning: Window size P reduced from {config.P} to {P} to match worker count.")
        
    thresh = config.threshold
    
    models = [None] * (T + 1)
    optimizers = [None] * (T + 1)
    
    begin_idx, end_idx = 0, P
    total_iters, total_Grad = 0, 0

    running_loss, running_perplexity, running_accuracy, running_f1, running_total = 0.0, 0.0, 0.0, 0.0, 0

    # Initialize model and optimizer states
    for step in range(P + 1):
        models[step] = model.clone(device)
        optimizers[step] = _create_optimizer(models[step], config)

    pbar = tqdm(total=T, desc="Starting parallel training...")
    total_comm_time = 0.0
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()

    while begin_idx < T:
        start_time_per_iter = time.time()
        parallel_len = end_idx - begin_idx
        pred_f = [None] * parallel_len
        metrics = [None] * parallel_len

        # 1. Distribute tasks to worker processes
        comm_start_time = time.time()
        for i in range(parallel_len):
            step = begin_idx + i
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))

        # 2. Collect results from worker processes
        for _ in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _grads
            metrics[_i] = _metrics
        time_comm_per_iter = time.time() - comm_start_time
        total_comm_time += time_comm_per_iter

        # 3. Execute fixed-point iteration updates
        rollout_model = models[begin_idx]
        rollout_optimizer = optimizers[begin_idx]
        ind, errors_all = None, 0

        for i in range(parallel_len):
            step = begin_idx + i
            rollout_model.set_grads_from_grads(pred_f[i])
            rollout_optimizer.step()
            rollout_optimizer.zero_grad()
            
            error = rollout_model.compute_error_from_model(models[step + 1])
            
            # Accumulate errors for adaptive threshold
            if config.adaptivity_type == 'median' and i == parallel_len // 2:
                errors_all = error
            elif config.adaptivity_type == 'mean':
                errors_all += error / parallel_len

            # Check convergence
            if ind is None and (error > thresh or i == parallel_len - 1):
                ind = step + 1
                optimizer_state_clone(rollout_optimizer, optimizers[step + 1])

            # Accumulate metrics (only for verified steps)
            if ind is None or step < ind:
                _metrics = metrics[i]
                running_loss += _metrics['loss']
                running_perplexity += _metrics['perplexity']
                running_accuracy += _metrics['accuracy']
                running_f1 += _metrics['f1_score']
                running_total += _metrics['batch_size']
            
            if ind is not None:
                # Once mismatch found, update all subsequent models with 'rollout' true state
                models[step + 1] = rollout_model.clone(device, models[step + 1])
            
            total_Grad += 1

        # 4. Update error threshold (EMA)
        thresh = thresh * config.ema_decay + errors_all * (1 - config.ema_decay)
        
        # 5. Slide window
        progress = ind - begin_idx
        pbar.update(progress)
        total_iters += 1
        
        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)

        # Prepare models for next window
        for step in range(end_idx + 1, new_end_idx + 1):
            prev_model_idx = step - parallel_len - 1
            models[step] = rollout_model.clone(device, models[prev_model_idx] if prev_model_idx >= 0 else None)
            optimizers[step] = optimizers[prev_model_idx] if prev_model_idx >= 0 else None

        begin_idx = new_begin_idx
        end_idx = new_end_idx
        time_per_iter = time.time() - start_time_per_iter
        time_comp_per_iter = time_per_iter - time_comm_per_iter

        # 6. Record progress to CSV and Pbar
        if begin_idx > 0:
            avg_loss = running_loss / begin_idx
            avg_perplexity = running_perplexity / begin_idx
            avg_accuracy = running_accuracy / begin_idx
            avg_f1 = running_f1 / begin_idx
            
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            pbar.set_description(
                f'Loss: {avg_loss:.4f} | PPL: {avg_perplexity:.2f} | Acc: {avg_accuracy:.2%} | F1: {avg_f1:.2%} | Time: {elapsed_str}'
            )           
            csv_logger.log({
                "Iter": total_iters,
                "Train_Loss": avg_loss,
                "Train_Acc": avg_accuracy,
                "Train_Perplexity": avg_perplexity,
                "Train_F1": avg_f1,
                "Original_Steps": begin_idx,
                "time_per_iter": time_per_iter,
                "time_comm_per_iter": time_comm_per_iter,
                "time_comp_per_iter": time_comp_per_iter
            })
            csv_logger.save() # Frequent saves in case of crash

    end_event.record()
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event) / 1000.0
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    pbar.close()

    # Clean up worker processes
    for _ in processes:
        queues[0].put(None)
    for p in processes:
        p.join()
        
    final_model = models[T]
    del models, optimizers
    
    # Final evaluation
    test_loss, test_perplexity, test_accuracy, test_f1 = evaluate_model(final_model, test_loader, device)

    upper_bound = total_iters * P
    m = total_Grad / T
    
    csv_logger.log({
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "final_test_accuracy": test_accuracy,
        "final_test_f1": test_f1,
        "time": elapsed,
        "time_comm": total_comm_time,
        "time_step": elapsed - total_comm_time,
        "K": total_iters,
        "G": total_Grad,
        "upper_bound": upper_bound,
        "m": m
    })
    
    comm_time_str = str(timedelta(seconds=int(total_comm_time)))
    print(f"\n--- Parallel Training Completed in {elapsed_str} ---")
    print(f"Total communication time: {comm_time_str}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test perplexity: {test_perplexity:.2f}")
    print(f"Final test accuracy: {test_accuracy:.2%}")
    print(f"Final test F1 score: {test_f1:.2%}")
    print(f"Total iterations (K): {total_iters} (vs {T} normal iterations)")
    print(f"Total gradients (G): {total_Grad}")
    print(f"Effective speed-up (T/K): {T/total_iters:.2f}x")
    
    return final_model

def train_loop_serial(config, model, train_loader, test_loader, csv_logger):
    """Standard serial training loop."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = _create_optimizer(model, config)
        
    start_time = time.time()
    running_loss = 0.0
    running_perplexity = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0
    running_total = 0    

    pbar = tqdm(total=config.max_steps, desc="Starting serial training...")
    data_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        res = take_step(model, optimizer, train_loader, data_iter, device, step, config.seed)
        data_iter = res['data_iter']
        
        # Gradients already computed in take_step, apply here
        optimizer.step()
        
        # Update statistics
        running_loss += res['loss']
        running_perplexity += res['perplexity']
        running_accuracy += res['accuracy']
        running_f1 += res['f1_score']
        running_total += res['batch_size']
        
        # Update progress bar
        if (step + 1) % 5 == 0 or (step + 1) == config.max_steps:
            avg_loss = running_loss / (step + 1)
            avg_perplexity = running_perplexity / (step + 1)
            avg_accuracy = running_accuracy / (step + 1)
            avg_f1 = running_f1 / (step + 1)
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            pbar.set_description(
                f'Loss: {avg_loss:.4f} | PPL: {avg_perplexity:.2f} | Acc: {avg_accuracy:.2%} | F1: {avg_f1:.2%} | Time: {elapsed_str}'
            )
            
            # Record metrics
            csv_logger.log({
                "Iter": step + 1,
                "Train_Loss": avg_loss,
                "Train_Acc": avg_accuracy,
                "Train_Perplexity": avg_perplexity,
                "Train_F1": avg_f1,
                "Original_Steps": step + 1
            })
            csv_logger.save() # Frequent saves
        
        pbar.update(1)
    
    pbar.close()
    
    elapsed = time.time() - start_time
        
    # Final test
    test_loss, test_perplexity, test_accuracy, test_f1 = evaluate_model(model, test_loader, device)

    csv_logger.log({
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "final_test_accuracy": test_accuracy,
        "final_test_f1": test_f1,
        "time": elapsed
    })
    
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f"\n--- Serial Training Completed in {elapsed_str} ---")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test perplexity: {test_perplexity:.2f}")
    print(f"Final test accuracy: {test_accuracy:.2%}")
    print(f"Final test F1 score: {test_f1:.2%}")      
    return model

# ==========================
# 9. Main Execution Logic
# ==========================

def setup_arg_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description='ParaOpt Training for Language Models (ModelScope Version)')

    # Training hyperparameters
    parser.add_argument('--device_count', type=int, help='Number of CUDA devices')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_steps', type=int, help='Total training steps')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')

    # Parallel hyperparameters
    parser.add_argument('--P', type=int, help='Window size for parallel algorithm')
    parser.add_argument('--threshold', type=float, help='Error threshold')
    parser.add_argument('--ema_decay', type=float, help='EMA decay for threshold')
    parser.add_argument('--adaptivity_type', type=str, choices=['mean', 'median'], help='Error computation type')
    
    # General configuration
    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adam', 'adamw'], help='Optimizer type')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial'], help='Training mode')
    parser.add_argument('--modelscope_name', type=str, help='Model name from ModelScope (e.g., qwen/qwen-1_8b)')
    parser.add_argument('--local_model_dir', type=str, help='Path to local model directory (overrides Config)')
    parser.add_argument('--dataset_name', type=str, choices=['local_wikitext'], help='Dataset to use')
    parser.add_argument('--local_dataset_dir', type=str, help='Path to local dataset directory (overrides Config)')
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for tokenizer')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, help='Directory to save logs (overrides Config)')
    parser.add_argument('--output_csv', type=str, help='Exact output CSV file path (overrides auto-naming)')

    return parser

def train_with_config(config=None, csv_logger=None):
    """Run a complete training process with given configuration."""
    if config is None:
        config = Config()

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize CSV logger
    if csv_logger is None:
        csv_logger = CSVLogger()
    
    print(f"--- Starting Training Run ---")
    print(f"Mode: {config.training_mode} | Model: {config.modelscope_name}")
    print(f"Steps: {config.max_steps} | Batch Size: {config.batch_size} | LR: {config.learning_rate}")
    if config.training_mode == 'parallel':
        print(f"Parallel Config: P={config.P}, Threshold={config.threshold}, Adaptivity={config.adaptivity_type}")
    print(f"Log file: {csv_logger.filename}")
    
    # Load data
    train_loader, test_loader = get_dataloaders(config)
    
    # Create model
    # **Important**:
    # In main process, we don't use device_map="auto".
    # `train_loop_serial` and `train_loop_parallel` will be responsible for moving it to cuda:0.
    model = ModelFactory.create_model(config, device_map=None)
    
    # Start training
    if config.training_mode.lower() == "parallel":
        train_loop_parallel(config, model, train_loader, test_loader, csv_logger)
    else:
        train_loop_serial(config, model, train_loader, test_loader, csv_logger)
    
    csv_logger.finish()
    print(f"--- Training Run Finished ---")
    print(f"Log file saved to: {csv_logger.filename}")

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    config = Config()
    config.update_from_args(args)
    
    # Determine output CSV file path
    output_csv = args.output_csv
    
    if output_csv is None:
        # If exact path not provided, auto-construct path
        directory_path = os.path.join(
            config.output_dir, # Use (possibly arg-overridden) output_dir from config
            config.modelscope_name.split('/')[-1], # Use short name of model
            f"steps_{config.max_steps}_bs_{config.batch_size}_lr_{config.learning_rate}_optim_{config.optimizer_type}"
        )
        
        # File name
        if config.training_mode == 'parallel':
            filename = f"{config.training_mode}_{config.adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.ema_decay}.csv"
        else:
            filename = f"{config.training_mode}.csv"
            
        output_csv = os.path.join(directory_path, filename)
    
    csv_logger = CSVLogger(output_csv)
    
    train_with_config(config, csv_logger)

if __name__ == "__main__":
    main()