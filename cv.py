#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script merges the parallel training logic from cv.py with the 
training features (AMP, LR scheduling, validation) from train_cifar10.py.
It includes adaptation for MaskedAutoencoderViT (MAE) for classification.

UPDATES:
1. CSV Logging now uses 'append' mode to avoid rewriting history and ensure speed.
2. Time accumulation logic ensures correct timing across restarts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
import time
import argparse
from datetime import timedelta
import numpy as np
import copy
import os
import math
import warnings
import torchvision.models as models
from torchvision import datasets
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from datetime import datetime
from transformers import get_scheduler
from functools import partial


from models.vit import ViT as ViTRegular
from models.resnet_gn import ResNet50 
from models.cnn import CNN 

# ========== Config Class ==========
class Config:
    """Training configuration parameters"""
    # Training parameters
    batch_size = 512
    n_epochs = None      # Number of epochs (overrides max_steps if set)
    max_steps = 60000    # Total iteration steps (used if n_epochs is None)
    learning_rate = 0.00001
    momentum = 0.9 
    optimizer_type = 'adam'
    training_mode = 'parallel'

    # Scheduler parameters
    lr_scheduler_type = 'constant' # 'linear', 'cosine', 'constant'
    warmup_steps = 0

    # Acceleration configuration (for parallel mode)
    P = 7   # Window size
    threshold = 0.3 # Error threshold
    ema_decay = 0.9999 # Threshold exponential moving average decay rate
    adaptivity_type = 'median'  # Adaptive strategy: 'mean' or 'median'
    dropout = 0.0
    
    # System configuration
    seed = 42    # Random seed
    device_count = torch.cuda.device_count()    # Number of GPUs

    # Dataset and model configuration
    dataset = 'cifar10' 
    model_name = 'vit_small'    
    num_classes = 10 
    img_size = 32 
    pretrained = False
    use_aug = True 
    
    # ViT/MAE specific config for cifar10
    patch_size = 4
    dimhead = 512

    # Logging and Validation
    use_amp = False 
    val_interval = 100 
    checkpoint_dir = './checkpoint' 
    resume_checkpoint_path = './resume.pth' 

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        
        if args.no_amp:
            self.use_amp = False
        if args.no_aug:
            self.use_aug = False


# ========== Optimized CSV Logger (Append Mode) ==========
class CSVLogger:
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.csv"
        
        self.filename = filename
        self.buffer = [] # Buffer for unsaved rows
        self.fieldnames = set()
        self.header_written = False
        
        directory = os.path.dirname(self.filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Check if file exists and has content to determine if header exists
        if os.path.exists(self.filename) and os.path.getsize(self.filename) > 0:
            self.header_written = True
            # Read existing fieldnames to maintain order
            try:
                with open(self.filename, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        self.fieldnames = set(reader.fieldnames)
            except Exception as e:
                print(f"Warning: Could not read header from existing file: {e}")

    def log(self, data):
        # Update known fields
        self.fieldnames.update(data.keys())
        # Add to buffer (do not write to disk yet)
        self.buffer.append(data)
        
    def save(self):
        """Writes only the buffered (new) rows to the file in append mode."""
        if not self.buffer:
            return

        # Determine mode: 'a' (append) if exists, 'w' (write) if new
        mode = 'a' if os.path.exists(self.filename) else 'w'
        
        # If file doesn't exist or is empty, we need to write header
        need_header = not self.header_written and (not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0)

        with open(self.filename, mode, newline='') as csvfile:
            # Sort fields for consistent column order
            sorted_fieldnames = sorted(list(self.fieldnames))
            writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames)
            
            if need_header:
                writer.writeheader()
                self.header_written = True
            
            for row in self.buffer:
                # Ensure all fields are present (fill missing with empty string)
                full_row = {field: row.get(field, "") for field in sorted_fieldnames}
                writer.writerow(full_row)
        
        # Clear the buffer after successful write
        self.buffer = []

    def finish(self):
        self.save()


# ========== Helper Functions ==========
def to_device(batch, device):
    images, labels = batch
    return images.to(device), labels.to(device)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x): 
        return self.model(x)

    def clone(self, device, other=None):
        if other is None: 
            other = ModelWrapper(copy.deepcopy(self.model)).to(device)
        else:
            with torch.no_grad():
                for param_to, param_from in zip(other.get_params(), self.get_params()):
                    param_to.data = param_from.data.clone()
        return other

    def get_params(self): 
        return list(self.parameters())

    def set_params(self, params, clone=True):
        for p, q in zip(self.get_params(), params):
            if clone: 
                p.data = q.data.clone().to(p.device)
            else: 
                p.data = q.to(p.device)

    def set_grads_from_grads(self, grads):
        for p, grad in zip(self.get_params(), grads):
            if grad is not None: 
                p.grad = grad.to(p.device)

    def compute_error_from_model(self, other):
        my_params, other_params = self.get_params(), other.get_params()
        with torch.no_grad():
            error,total_num = 0.0, 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p - q).pow(2).item()
                total_num += np.prod(list(q.shape))
            return error/total_num *1e6


class ModelFactory:
    @staticmethod
    def create_model(config):
        model_name = config.model_name.lower()
        num_classes = config.num_classes
        img_size = config.img_size
        dropout =  config.dropout
        patch_size = config.patch_size
        dim = config.dimhead
        
        if model_name == 'cnn':
            model = CNN(num_classes=num_classes, img_size=img_size)
        elif model_name == "resnet50":
            model = ResNet50(num_classes=num_classes)
        elif model_name == 'vit':
            model = ViTRegular(
                image_size = img_size,
                patch_size = patch_size,
                num_classes = num_classes,
                dim = int(dim),
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = dropout,
                emb_dropout = dropout
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return ModelWrapper(model)


# ========== Dataset Loading ==========

def get_dataloaders(config):
    if config.dataset == 'cifar10':
        print("Loading CIFAR-10 dataset...")
        config.num_classes = 10
        config.img_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        transform_train_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        
        if config.use_aug:
            print("Using train augmentations: RandomCrop(32, 4) and RandomHorizontalFlip")
            transform_train_list.insert(0, transforms.RandomHorizontalFlip())
            transform_train_list.insert(0, transforms.RandomCrop(32, padding=4))

        transform_train = transforms.Compose(transform_train_list)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    return train_loader, test_loader


# ========== Core Training Logic ==========
def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())


def take_step(model, optimizer, criterion, dataloader, data_iter, device, step, seed_offset, use_amp):
    np.random.seed(step + seed_offset)
    torch.manual_seed(step + seed_offset)
    torch.cuda.manual_seed(step + seed_offset)
    
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)
        
    images, labels = to_device(batch, device)
    
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    loss.backward()
    
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
    
    return {
        'loss': loss.item(), 
        'accuracy': 100 * correct / labels.size(0), 
        'data_iter': data_iter, 
        'batch_size': labels.size(0), 
        'correct': correct
    }


def run_worker(model, optimizer, criterion, dataloader, queues, device, seed_offset, use_amp, rank):
    data_iter = iter(dataloader)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    while True:
        ret = queues[0].get()
        if ret == 'FINISH':
            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated(device)
            else:
                peak_memory = 0 
            
            queues[2].put((rank, peak_memory))
            return
            
        if ret is None:
             return

        params, step = ret
        model.set_params(params, clone=False)
        
        res = take_step(model, optimizer, criterion, dataloader, data_iter, device, step, seed_offset, use_amp)
        data_iter = res['data_iter']
        
        grads = [p.grad for p in model.get_params()]
        
        queues[1].put((grads, step, {
            'loss': res['loss'], 
            'accuracy': res['accuracy'], 
            'batch_size': res['batch_size'], 
            'correct': res['correct']
        }))


def create_optimizer(model, config):
    if config.optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    else: 
        return optim.AdamW(model.parameters(), lr=config.learning_rate)


def train_loop_parallel(config, model, criterion, train_loader, test_loader, csv_logger):
    if config.device_count <= 1:
        print("Error: Only one GPU found")
        return

    mp.set_start_method('spawn', force=True)
    queues = (mp.Queue(), mp.Queue(), mp.Queue())
    processes = []
    

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    

    torch.cuda.reset_peak_memory_stats(0)
    

    for rank in range(1, config.device_count):
        worker_device = torch.device(f"cuda:{rank}")
        model = model.to(worker_device)
        worker_optimizer = create_optimizer(model, config)
            
        p = mp.Process(target=run_worker, args=(
            model, worker_optimizer, criterion, train_loader, 
            queues, worker_device, config.seed, config.use_amp, rank # 传入 rank
        ))
        p.start()
        processes.append(p)

    model = model.to(device)
    del worker_optimizer
    T = config.max_steps
    P = min(config.P, T)
    
    models = [None for _ in range(T + 1)]
    optimizers = [None for _ in range(T + 1)]
    
    begin_idx = 0
    end_idx = P
    total_iters, running_loss, running_correct, running_total, total_Grad = 0, 0.0, 0, 0, 0
    best_acc = 0.0
    last_val_step = -config.val_interval
    thresh = config.threshold
    accumulated_time = 0.0 
    
    # --- RESUME LOGIC ---
    resumed = False
    if os.path.exists(config.resume_checkpoint_path):
        print(f"Resuming from checkpoint: {config.resume_checkpoint_path}")
        checkpoint = torch.load(config.resume_checkpoint_path)
        
        begin_idx = checkpoint['begin_idx']
        end_idx = min(begin_idx + P, T)
        total_iters = checkpoint['total_iters']
        thresh = checkpoint['thresh']
        running_loss = checkpoint['running_loss']
        running_correct = checkpoint['running_correct']
        running_total = checkpoint['running_total']
        best_acc = checkpoint['best_acc']
        last_val_step = checkpoint['last_val_step']
        total_Grad = checkpoint['total_Grad']
        
        accumulated_time = checkpoint.get('accumulated_time', 0.0)
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        models[begin_idx] = model.clone(device)
        optimizers[begin_idx] = create_optimizer(models[begin_idx], config)
        optimizers[begin_idx].load_state_dict(checkpoint['optimizer_state_dict'])
        
        for step in range(begin_idx + 1, end_idx + 1):
            models[step] = models[begin_idx].clone(device)
            optimizers[step] = create_optimizer(models[step], config)
            optimizer_state_clone(optimizers[begin_idx], optimizers[step])
        
        resumed = True
    else:
        for step in range(P + 1):
            models[step] = model.clone(device)
            optimizers[step] = create_optimizer(models[step], config)

    # --- LR Scheduler ---
    main_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizers[begin_idx],
        num_warmup_steps=config.warmup_steps,
        num_training_steps=T
    )

    if resumed:
        main_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    pbar = tqdm(total=T, initial=begin_idx)
    
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()

    while begin_idx < T:
        parallel_len = end_idx - begin_idx
        pred_f = [None] * parallel_len
        metrics = [None] * parallel_len

        # 1. Distribute tasks
        for i in range(parallel_len):
            step = begin_idx + i
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))

        # 2. Collect results
        for _ in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _grads
            metrics[_i] = _metrics

        # 3. Fixed-point iteration
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
            error = rollout_model.compute_error_from_model(models[step + 1])
            rollout_optimizer.zero_grad()
            
            if config.adaptivity_type == 'median' and i == parallel_len // 2:
                errors_all = error
            elif config.adaptivity_type == 'mean':
                errors_all += error / parallel_len
            
            if ind is None and (error > thresh or i == parallel_len - 1):
                ind = step + 1
                optimizer_state_clone(rollout_optimizer, optimizers[step + 1])
           
            if ind is None or step < ind:
                running_loss += metrics[i]['loss']
                running_correct += metrics[i]['correct']
                running_total += metrics[i]['batch_size']
            
            if ind is not None:
                models[step + 1] = rollout_model.clone(device, models[step + 1])
            
            total_Grad += 1

        # 4. Update threshold
        thresh = thresh * config.ema_decay + errors_all * (1 - config.ema_decay)
        
        # 5. Slide window
        progress = ind - begin_idx
        pbar.update(progress)
        total_iters += 1
        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)

        for step_i in range(end_idx + 1, new_end_idx + 1):
            prev_model_idx = step_i - 1 - parallel_len
            models[step_i] = rollout_model.clone(device, models[prev_model_idx] if prev_model_idx >= 0 else None)
            optimizers[step_i] = optimizers[prev_model_idx] if prev_model_idx >= 0 else None

        begin_idx = new_begin_idx
        end_idx = new_end_idx
        
        # 6. Log progress
        accuracy = 100 * running_correct / running_total if running_total > 0 else 0
        avg_loss = running_loss / begin_idx if begin_idx > 0 else 0
        
        end_event.record()
        torch.cuda.synchronize()
        current_session_time = start_event.elapsed_time(end_event) / 1000.0
        total_elapsed_time = accumulated_time + current_session_time
        elapsed_str = str(timedelta(seconds=int(total_elapsed_time)))
        
        pbar.set_description(f'para Lr: {current_lr:.8f} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {elapsed_str}')
        
        csv_logger.log({
            "Iter": total_iters,
            "Original_Steps": begin_idx,
            "LR": current_lr,
            "Train_Loss": avg_loss,
            "Train_Acc": accuracy,
        })
        
        # 7. Validation Logic
        if total_iters % config.val_interval == 0 or begin_idx == config.max_steps:
            last_val_step = begin_idx
            current_model = models[begin_idx]
            val_acc, val_metrics = evaluate_model(current_model, test_loader, device)
            
            pbar.write(f"\n[Validation at para step {begin_idx}]|{config.model_name}|{config.dataset}|{config.optimizer_type}|{config.lr_scheduler_type}|{config.batch_size}|{config.learning_rate}| Test Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%")

            csv_logger.log({
                "Iter": total_iters,
                "Original_Steps": begin_idx,
                "Val_Acc": val_acc,
                "Val_Precision": val_metrics['precision_macro'],
                "Val_Recall": val_metrics['recall_macro'],
                "Val_F1": val_metrics['f1_macro']
            })

            if val_acc > best_acc:
                best_acc = val_acc
                pbar.write(f"Saving new best model to {config.checkpoint_dir}")
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                torch.save(current_model.model.state_dict(), 
                               os.path.join(config.checkpoint_dir, f"{config.model_name}_{config.dataset}_best.pth"))
            
            # Efficient Append Save
            csv_logger.save()

            # --- RESUME CHECKPOINT SAVE ---
            resume_state = {
                'begin_idx': begin_idx,
                'total_iters': total_iters,
                'thresh': thresh,
                'running_loss': running_loss,
                'running_correct': running_correct,
                'running_total': running_total,
                'best_acc': best_acc,
                'last_val_step': last_val_step,
                'total_Grad': total_Grad,
                'model_state_dict': models[begin_idx].model.state_dict(),
                'optimizer_state_dict': optimizers[begin_idx].state_dict(),
                'scheduler_state_dict': main_scheduler.state_dict(),
                'accumulated_time': total_elapsed_time 
            }
            torch.save(resume_state, config.resume_checkpoint_path)

    end_event.record()
    torch.cuda.synchronize()
    current_session_time = start_event.elapsed_time(end_event) / 1000.0
    total_elapsed_time = accumulated_time + current_session_time
    elapsed_str = str(timedelta(seconds=int(total_elapsed_time)))
    pbar.close()
    
    # --- Cleanup & Final Stats ---
    peak_memories = {}
    
    for _ in processes:
        queues[0].put('FINISH')
        
    for _ in processes:
        rank, peak_memory = queues[2].get()
        peak_memories[f"GPU_{rank}_peak_memory_mb"] = peak_memory  / (1024 ** 2)
        
    peak_memories[f"GPU_0_peak_memory_MB"] = torch.cuda.max_memory_allocated(device)  / (1024 ** 2)
    all_peak_values = [v for v in peak_memories.values()]
    max_peak_mb = max(all_peak_values) if all_peak_values else 0

    for p in processes:
        p.join()
        
    final_model = models[T]
    del models, optimizers
    final_accuracy, metrics = evaluate_model(final_model, test_loader, device)
    
    m = total_Grad / T
    
    log_data = {
        "final_test_accuracy": final_accuracy,
        "final_test_precision":metrics["precision_macro"],
        "final_test_recall":metrics["recall_macro"],
        "final_test_f1-score":metrics["f1_macro"],
        "time": total_elapsed_time,
        "K":total_iters,
        "m":m,
        "peak_memory_max_MB": max_peak_mb 
    }
    log_data.update(peak_memories) 
    
    csv_logger.log(log_data)
    csv_logger.finish() # Flush last data
    
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Final test precision: {metrics['precision_macro']:.2f}%")
    print(f"Final test recall: {metrics['recall_macro']:.2f}%")
    print(f"Final test f1-score: {metrics['f1_macro']:.2f}%")        
    print(f"Total iterations: {total_iters} (vs {T} normal iterations)")
    print(f"Effective speed-up: {T/total_iters:.2f}x")
    print(f"Maximum GPU Peak Memory (across all cards): {max_peak_mb:.2f} MB")
    
    if os.path.exists(config.resume_checkpoint_path):
        print(f"Training complete. Removing resume checkpoint: {config.resume_checkpoint_path}")
        os.remove(config.resume_checkpoint_path)

    return final_model


def train_loop_serial(config, model, criterion, train_loader, test_loader, csv_logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = create_optimizer(model, config)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory_max_mb = 0.0
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )

    accumulated_time = 0.0
    running_loss, running_correct, running_total = 0.0, 0, 0
    best_acc = 0.0
    start_step = 0
    
    # --- RESUME LOGIC START ---
    if os.path.exists(config.resume_checkpoint_path):
        print(f"Resuming serial training from: {config.resume_checkpoint_path}")
        checkpoint = torch.load(config.resume_checkpoint_path)
        
        start_step = checkpoint['step']
        running_loss = checkpoint['running_loss']
        running_correct = checkpoint['running_correct']
        running_total = checkpoint['running_total']
        best_acc = checkpoint['best_acc']
        
        accumulated_time = checkpoint.get('accumulated_time', 0.0) 
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    pbar = tqdm(total=config.max_steps, initial=start_step)
    data_iter = iter(train_loader)
    
    session_start_time = time.time()
    
    for step in range(start_step, config.max_steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
            
        images, labels = images.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
        
        running_loss += loss.item()
        running_correct += correct
        running_total += labels.size(0)
        
        avg_loss = running_loss / (step + 1)
        avg_acc = 100 * running_correct / running_total
        
        current_session_elapsed = time.time() - session_start_time
        total_elapsed = accumulated_time + current_session_elapsed
        elapsed_str = str(timedelta(seconds=int(total_elapsed)))
        
        current_lr = scheduler.get_last_lr()[0]
        
        csv_logger.log({
            "Iter": step + 1,
            "Original_Steps": step + 1,
            "LR": current_lr,
            "Train_Loss": avg_loss,
            "Train_Acc": avg_acc,
        })
            
        pbar.set_description(f'serial Lr: {current_lr:.8f} |Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | Time: {elapsed_str}')
            
        pbar.update(1)
        
        # --- Validation Logic ---
        if (step + 1) % config.val_interval == 0 or (step + 1) == config.max_steps:
            val_acc, val_metrics = evaluate_model(model, test_loader, device)
            pbar.write(f"\n[Validation at serial step {step+1}]|{config.model_name}|{config.dataset}|{config.optimizer_type}|{config.lr_scheduler_type}|{config.batch_size}|{config.learning_rate}| Test Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%")
            
            csv_logger.log({
                "Iter": step + 1,
                "Val_Acc": val_acc,
                "Val_Precision": val_metrics['precision_macro'],
                "Val_Recall": val_metrics['recall_macro'],
                "Val_F1": val_metrics['f1_macro']
            })
            
            if val_acc > best_acc:
                best_acc = val_acc
                pbar.write(f"Saving new best model to {config.checkpoint_dir}")
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                torch.save(model.model.state_dict(), 
                               os.path.join(config.checkpoint_dir, f"{config.model_name}_{config.dataset}_best.pth"))
            
            # Efficient Append Save
            csv_logger.save()
            
            # --- RESUME CHECKPOINT SAVE ---
            current_session_elapsed = time.time() - session_start_time
            total_elapsed_for_save = accumulated_time + current_session_elapsed
            
            resume_state = {
                'step': step + 1,
                'running_loss': running_loss,
                'running_correct': running_correct,
                'running_total': running_total,
                'best_acc': best_acc,
                'accumulated_time': total_elapsed_for_save, 
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(resume_state, config.resume_checkpoint_path)
        
    pbar.close()
    
    current_session_elapsed = time.time() - session_start_time
    total_elapsed = accumulated_time + current_session_elapsed
    
    final_accuracy, metrics = evaluate_model(model, test_loader, device)
    
    if device.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_max_mb = peak_memory_bytes / (1024 ** 2)
        print(f"Single GPU Peak Memory: {peak_memory_max_mb:.2f} MB")
    
    log_data = {
        "final_test_accuracy": final_accuracy, 
        "final_test_precision": metrics["precision_macro"], 
        "final_test_recall": metrics["recall_macro"], 
        "final_test_f1-score": metrics["f1_macro"], 
        "time": total_elapsed,
        "peak_memory_max_MB": peak_memory_max_mb if device.type == 'cuda' else 0.0
    }
    csv_logger.log(log_data)
    csv_logger.finish()
    
    print(f"\nTraining completed in {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    
    if os.path.exists(config.resume_checkpoint_path):
        print(f"Training complete. Removing resume checkpoint: {config.resume_checkpoint_path}")
        os.remove(config.resume_checkpoint_path)
    
    return model


# ========== Evaluation Function ==========
def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': 100 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    
    model.train()
    return accuracy, metrics


# ========== Main Execution Logic ==========
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Parallel/Serial Training')

    # Training parameters
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs (overrides max_steps)')
    parser.add_argument('--max_steps', type=int, help='max steps (used if n_epochs is not set)')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adam', 'adamw'], help='optimizer type')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial'], help='training mode')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout for vit')

    # Scheduler parameters
    parser.add_argument('--lr_scheduler_type', type=str, default='constant', 
                        choices=['linear', 'cosine', 'constant'], 
                        help='learning rate scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=0, help='number of warmup steps')

    # Parallel-specific parameters
    parser.add_argument('--P', type=int, help='window size')
    parser.add_argument('--threshold', type=float, help='threshold')
    parser.add_argument('--ema_decay', type=float, help='EMA decay for threshold')
    parser.add_argument('--adaptivity_type', type=str, choices=['mean', 'median'], help='error computation type')
       
    # Model and Dataset
    parser.add_argument('--model_name', type=str, 
                        choices=['cnn', 'vit', 'resnet50'], 
                        help='model to use')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], help='Dataset to use')
    parser.add_argument('--num_classes', type=int, help='number of classes (auto-set by dataset)')
    
    # ViT specific
    parser.add_argument('--patch', default=4, type=int, dest='patch_size', help="patch for ViT")
    parser.add_argument('--dimhead', default=512, type=int, help="dimhead for ViT")

    # System and Logging
    parser.add_argument('--device_count', type=int, help='Number of CUDA devices to use')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs')
    parser.add_argument('--val_interval', type=int, help='Validate every N steps')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory to save best model')
    parser.add_argument('--no_amp', action='store_true', help='Disable Automatic Mixed Precision')
    parser.add_argument('--no_aug', action='store_true', help='Disable training data augmentation')

    return parser


def train_with_config(config=None, csv_logger=None):
    if config == None:
        config = Config()

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # --- Load Data and Set Steps ---
    train_loader, test_loader = get_dataloaders(config)
    
    if config.n_epochs is not None:
        config.max_steps = config.n_epochs * len(train_loader)
        print(f"Running for {config.n_epochs} epochs, calculated as {config.max_steps} total steps.")
    else:
        config.n_epochs = math.ceil(config.max_steps / len(train_loader))
        print(f"Running for {config.max_steps} steps, approximately {config.n_epochs} epochs.")

    if csv_logger is None:
        csv_logger = CSVLogger()
    
    model = ModelFactory.create_model(config)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting {config.training_mode} training...")
    if config.training_mode.lower() == "parallel":
        train_loop_parallel(config, model, criterion, train_loader, test_loader, csv_logger)
    else:
        train_loop_serial(config, model, criterion, train_loader, test_loader, csv_logger)

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    config = Config()
    config.update_from_args(args)
    
    log_dir_base = args.log_dir if args.log_dir else "./PASO_standard/2025_12_12_1_04"
    
    if config.model_name in ['vit','vit_small']:
        directory_path = os.path.join(
            log_dir_base, config.dataset, config.model_name,
            f"steps_{config.max_steps}_bs_{config.batch_size}_lr_{config.learning_rate}_optim_{config.optimizer_type}_sched_{config.lr_scheduler_type}_warmup_{config.warmup_steps}_dropout_{config.dropout}"
        )
    else:
        directory_path = os.path.join(
            log_dir_base, config.dataset, config.model_name,
            f"steps_{config.max_steps}_bs_{config.batch_size}_lr_{config.learning_rate}_optim_{config.optimizer_type}_sched_{config.lr_scheduler_type}_warmup_{config.warmup_steps}"
        )
    
    if config.training_mode.lower() == 'parallel':
        output_csv = os.path.join(directory_path, 
            f"{config.training_mode}_{config.adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.ema_decay}.csv")
    else:
        output_csv = os.path.join(directory_path, f"{config.training_mode}.csv")
        
    print(f"Saving logs to: {output_csv}")

    resume_checkpoint_filename = os.path.join(directory_path, f"{config.training_mode}_{config.adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.ema_decay}"+"_resume_checkpoint.pth")
    config.resume_checkpoint_path = resume_checkpoint_filename

    csv_logger = CSVLogger(output_csv)
    
    train_with_config(config, csv_logger)


if __name__ == "__main__":
    main()