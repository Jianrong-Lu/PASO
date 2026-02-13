# -*- coding: utf-8 -*-

"""
Optimized PASO (Parallel Asynchronous Stochastic Optimization)
- Fixed memory fragmentation via pre-allocated buffers.
- Improved communication speed using flattened tensor views and non-blocking P2P.
- Reduced VRAM overhead by avoiding redundant clones.
- STRICT PRE-ALLOCATION applied to all communication, metric tensors, AND optimizer states.
"""

import os
import random
import time
import copy
from typing import Any, Tuple
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def log_memory(tag=""):
    """
    Print current GPU memory usage.
    :param tag: Marker for the current code position (string)
    """
    if not torch.cuda.is_available():
        return

    # Get current device
    device = torch.cuda.current_device()
    
    # Force synchronization to get accurate numbers (slightly impacts performance, debug only)
    torch.cuda.synchronize()
    
    # Get memory data (Byte -> MB)
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    
    # Get current process Rank
    try:
        rank = dist.get_rank()
    except:
        rank = 0

    print(f"[Rank {rank}] {tag} | Alloc: {allocated:.2f}MB | Rsrv: {reserved:.2f}MB | Peak: {max_allocated:.2f}MB")


# ==============================================================================
# 1. PASO DATA LOADER
# ==============================================================================

class PASODataLoader:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.seed_offset = 42

    def __len__(self) -> int:
        return len(self.dataloader)

    def __getitem__(self, step: int) -> Any:
        np.random.seed(step + self.seed_offset)
        torch.manual_seed(step + self.seed_offset)
        torch.cuda.manual_seed(step + self.seed_offset)
        
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch


# ==============================================================================
# 2. PASO OPTIMIZER (OPTIMIZED WITH FULL PRE-ALLOCATION)
# ==============================================================================

class PASOOptimizer:
    def __init__(self, model: torch.nn.Module, optimizer: optim.Optimizer, config: Any):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = next(model.parameters()).device
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.begin_idx = 0
        self.end_idx = min(self.begin_idx + self.world_size, self.config.max_steps)
        
        self.start_rank = 0
        self.thresh = config.threshold
        self._update_rank_maps()

        # --- DeepSpeed-style Bucketing ---

        self.param_list = [p for p in model.parameters()]
        self.total_params = sum(p.numel() for p in self.param_list)
        
        # Reduce memory footprint for single communication (50M elements approx 200MB FP32)
        limit_size = 50 * 1024 * 1024
        self.bucket_size = min(self.total_params, limit_size)

        # Pre-define bucket index mapping
        self.buckets = []
        current_bucket = []
        current_size = 0
        for p in self.param_list:
            current_bucket.append(p)
            current_size += p.numel()
            if current_size >= self.bucket_size:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
        if current_bucket:
            self.buckets.append(current_bucket)

        # Allocate buffers
        actual_max_bucket = max(sum(p.numel() for p in b) for b in self.buckets)
        self.flat_buffer = torch.zeros(actual_max_bucket, device=self.device)
        self.grad_buffer = torch.zeros(actual_max_bucket, device=self.device)
        self.error_tensor = torch.zeros(1, device=self.device)
        
        # --- Metrics Pre-allocation ---
        self.local_metric_buffer = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.gather_buffer_list = [
            torch.zeros(3, device=self.device, dtype=torch.float32) 
            for _ in range(self.world_size)
        ]
        
        # Sync Data: [new_start_rank, sync_point, window_loss, window_correct, window_total]
        self.sync_data = torch.zeros(5, device=self.device, dtype=torch.float32)

        # --- Optimizer State Pre-allocation (FIXED HERE) ---
        self.supported_keys = ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', 'momentum_buffer']
        
        # Force initialize all optimizer states for all parameters now
        # so we don't need torch.zeros_like later.
        for p in self.param_list:
            if p not in self.optimizer.state:
                self.optimizer.state[p] = {}
            
            # Pre-allocate step
            if 'step' not in self.optimizer.state[p]:
                self.optimizer.state[p]['step'] = torch.tensor(0.0, device=self.device)
            
            # Pre-allocate potential state tensors (SGD/Adam buffers)
            # This consumes VRAM upfront but guarantees no allocation during training
            for key in self.supported_keys:
                if key not in self.optimizer.state[p]:
                    self.optimizer.state[p][key] = torch.zeros_like(p, device=self.device)

        # Protocol buffers
        self.protocol_buffer = torch.zeros(1 + len(self.supported_keys), device=self.device, dtype=torch.float32)
        self.step_buffer = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.found_keys_mask = [0.0] * len(self.supported_keys)

        if self.rank == 0:
            log_memory("PASOOptimizer Init Completed (Buffers & States Pre-Allocated)")

    def _process_buckets(self, mode='send_params', target_rank=None):
        """Process parameter or gradient sending/receiving bucket by bucket."""
        for bucket in self.buckets:
            
            b_size = sum(p.numel() for p in bucket)
            active_buffer = self.flat_buffer[:b_size]
            active_grad_buffer = self.grad_buffer[:b_size]
            
            if mode == 'send_params':
                offset = 0
                for p in bucket:
                    active_buffer[offset:offset+p.numel()].copy_(p.data.view(-1))
                    offset += p.numel()
                dist.send(tensor=active_buffer, dst=target_rank)

            elif mode == 'recv_params':
                dist.recv(tensor=active_buffer, src=target_rank)
                offset = 0
                for p in bucket:
                    p.data.copy_(active_buffer[offset:offset+p.numel()].view(p.shape))
                    offset += p.numel()

            elif mode == 'send_grads':
                offset = 0
                for p in bucket:
                    if p.grad is not None:
                        active_grad_buffer[offset:offset+p.numel()].copy_(p.grad.data.view(-1))
                    else:
                        active_grad_buffer[offset:offset+p.numel()].zero_()
                    offset += p.numel()
                dist.send(tensor=active_grad_buffer, dst=target_rank)

            elif mode == 'recv_grads':
                dist.recv(tensor=active_grad_buffer, src=target_rank)
                offset = 0
                for p in bucket:
                    if p.grad is not None:
                        p.grad.data.copy_(active_grad_buffer[offset:offset+p.numel()].view(p.shape))
                    offset += p.numel()

    def get_step_indices(self) -> Tuple[int, int]:
        return self.begin_idx, self.rank_to_order_map[self.rank]

    def _update_rank_maps(self):
        self.rank_to_order_map = [(r - self.start_rank) % self.world_size for r in range(self.world_size)]
        self.order_to_rank_map = [(self.start_rank + o) % self.world_size for o in range(self.world_size)]

    def _sync_optimizer_state(self, src_rank: int, dest_rank: int):
        """
        [P2P Optimized] Point-to-point synchronization.
        Uses completely pre-allocated state tensors.
        """
        if self.rank != src_rank and self.rank != dest_rank:
            return

        SUPPORTED_KEYS = self.supported_keys
        protocol_tensor = self.protocol_buffer
        protocol_tensor.zero_()

        # ======================================================================
        # Phase 1: Handshake
        # ======================================================================
        if self.rank == src_rank:
            # --- Probe Logic (Sender) ---
            global_step = 0.0
            for i in range(len(self.found_keys_mask)):
                self.found_keys_mask[i] = 0.0
            
            # Check logic: Check existing values. 
            # Note: We pre-allocated everything as zeros. 
            # Ideally, we should check if the optimizer has actually stepped or updated them.
            # However, for synchronization, it is safe to sync whatever is there.
            # To be precise, we check the first param's step value.
            first_p = self.param_list[0]
            if 'step' in self.optimizer.state[first_p]:
                v = self.optimizer.state[first_p]['step']
                global_step = v.item() if isinstance(v, torch.Tensor) else float(v)

            # We assume all keys are "active" since we pre-allocated them.
            # Or we can check if they are non-zero if we want to save bandwidth (optional optimization).
            # For strict correctness and simplicity here: we sync all supported keys.
            for idx in range(len(SUPPORTED_KEYS)):
                self.found_keys_mask[idx] = 1.0 
            
            protocol_tensor[0] = global_step
            for i, val in enumerate(self.found_keys_mask):
                protocol_tensor[1 + i] = val
                
            dist.send(protocol_tensor, dst=dest_rank)
            
        elif self.rank == dest_rank:
            dist.recv(protocol_tensor, src=src_rank)

        # ======================================================================
        # Phase 2: Parse Handshake & Update Step
        # ======================================================================
        current_step = protocol_tensor[0].item()
        
        active_key_indices = []
        for i in range(len(SUPPORTED_KEYS)):
            if protocol_tensor[1 + i] > 0.5:
                active_key_indices.append(i)

        if self.rank == dest_rank:
            self.step_buffer.fill_(current_step)
            for p in self.param_list:
                # Direct copy to pre-allocated tensor
                self.optimizer.state[p]['step'].copy_(self.step_buffer)

        # ======================================================================
        # Phase 3: P2P Tensor State Transfer
        # ======================================================================
        for key_idx in active_key_indices:
            key_name = SUPPORTED_KEYS[key_idx]
            
            for bucket in self.buckets:
                b_size = sum(p.numel() for p in bucket)
                active_buffer = self.flat_buffer[:b_size]

                if self.rank == src_rank:
                    # --- Pack ---
                    offset = 0
                    for p in bucket:
                        # Directly access pre-allocated state
                        active_buffer[offset:offset+p.numel()].copy_(
                            self.optimizer.state[p][key_name].view(-1)
                        )
                        offset += p.numel()
                    
                    dist.send(active_buffer, dst=dest_rank)

                elif self.rank == dest_rank:
                    # --- Receive ---
                    dist.recv(active_buffer, src=src_rank)
                    
                    # --- Unpack ---
                    offset = 0
                    for p in bucket:
                        # Direct copy to pre-allocated state - NO ALLOCATION HERE
                        self.optimizer.state[p][key_name].view(-1).copy_(
                            active_buffer[offset:offset+p.numel()]
                        )
                        offset += p.numel()
    
    def multi_step(self, local_metrics: Tuple[float, int, int]) -> Tuple[int, Tuple[float, int, int]]:
        rank_order = self.rank_to_order_map[self.rank]
        parallel_len = self.end_idx - self.begin_idx

        self.local_metric_buffer[0] = local_metrics[0]
        self.local_metric_buffer[1] = float(local_metrics[1])
        self.local_metric_buffer[2] = float(local_metrics[2])
        
        root_rank = self.order_to_rank_map[0]
        gather_target = self.gather_buffer_list if self.rank == root_rank else None
        
        dist.gather(self.local_metric_buffer, gather_target, dst=root_rank)

        if rank_order == 0:
            sync_point = None
            cumulative_error, window_loss, window_correct, window_total = 0.0, 0.0, 0, 0
            
            for i in range(parallel_len):
                step = self.begin_idx + i
                self.optimizer.step()
                
                if i + 1 < parallel_len:
                    target_rank = self.order_to_rank_map[i + 1]
                    self._process_buckets(mode='send_params', target_rank=target_rank)
                    dist.recv(tensor=self.error_tensor, src=target_rank)
                    error = self.error_tensor.item()
                    self._process_buckets(mode='recv_grads', target_rank=target_rank)
                else:
                    error = 2 * self.thresh
                
                if self.config.adaptivity_type == 'mean':
                    cumulative_error += error / parallel_len
                elif self.config.adaptivity_type == 'median' and i == parallel_len // 2:
                    cumulative_error = error
           
                if sync_point is None or step < sync_point:
                    metrics = self.gather_buffer_list[i]
                    window_loss += metrics[0].item()
                    window_correct += metrics[1].item()
                    window_total += metrics[2].item()
                
                if sync_point is None and (error > self.thresh or i == parallel_len - 1):
                    sync_point = step + 1
                    new_start_rank = self.order_to_rank_map[i + 1] if i + 1 < parallel_len else self.order_to_rank_map[0]

            self.thresh = self.thresh * self.config.ema_decay + cumulative_error * (1 - self.config.ema_decay)
            
            self.sync_data[0] = float(new_start_rank)
            self.sync_data[1] = float(sync_point)
            self.sync_data[2] = window_loss
            self.sync_data[3] = float(window_correct)
            self.sync_data[4] = float(window_total)
        
        elif rank_order < parallel_len:
            with torch.no_grad():
                total_diff_sq = 0.0
                for bucket in self.buckets:
                    b_size = sum(p.numel() for p in bucket)
                    active_buffer = self.flat_buffer[:b_size]
                    dist.recv(tensor=active_buffer, src=self.order_to_rank_map[0])
                    
                    offset = 0
                    for p in bucket:
                        p_flat = p.data.view(-1)
                        target_part = active_buffer[offset:offset+p.numel()]
                        total_diff_sq += torch.sum((p_flat - target_part)**2).item()
                        p_flat.copy_(target_part) 
                        offset += p.numel()
                
                error = (total_diff_sq / self.total_params) * 1e6

            self.error_tensor.fill_(error)
            dist.send(tensor=self.error_tensor, dst=self.order_to_rank_map[0])
            self._process_buckets(mode='send_grads', target_rank=self.order_to_rank_map[0])

        dist.broadcast(self.sync_data, src=self.order_to_rank_map[0])
        
        new_start_rank_val = int(self.sync_data[0].item())
        sync_point_val = int(self.sync_data[1].item())
        wl = self.sync_data[2].item()
        wc = int(self.sync_data[3].item())
        wt = int(self.sync_data[4].item())
        
        if self.start_rank != new_start_rank_val:
            self._sync_optimizer_state(src_rank=self.start_rank, dest_rank=new_start_rank_val)
        
        torch.cuda.synchronize()
        
        new_begin_idx = sync_point_val
        progress = new_begin_idx - self.begin_idx
        self.begin_idx = new_begin_idx
        self.end_idx = min(new_begin_idx + self.world_size, self.config.max_steps)
        self.start_rank = new_start_rank_val
        self._update_rank_maps()
        
        return progress, (wl, wc, wt)

# ==============================================================================
# 3. TRAINING INFRASTRUCTURE
# ==============================================================================

class Config:
    max_steps: int = 60000
    learning_rate: float = 0.00085
    batch_size: int = 4096
    threshold: float = 0.001
    ema_decay: float = 0.999
    adaptivity_type: str = 'median'
    seed: int = 42

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            _, pred = torch.max(out.data, 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()
    return 100 * correct / total

# ==============================================================================
# 4. TRAINING METHODS
# ==============================================================================

def train_sequential(config, model, train_loader, test_loader, device, rank):
    if rank != 0: return None, None
    
    print("\n--- Starting Sequential Training ---")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    paso_loader = PASODataLoader(train_loader)
    
    start_time = time.time()
    pbar = tqdm(total=config.max_steps, desc="Sequential")
    
    for step in range(config.max_steps):
        inputs, labels = paso_loader[step]
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        pbar.update(1)
        
    pbar.close()
    elapsed = time.time() - start_time
    acc = evaluate(model, test_loader, device)
    print(f"Sequential Results -> Acc: {acc:.2f}%, Time: {elapsed:.2f}s")
    return acc, elapsed

def main():
    DATA_ROOT = './data'
    config = Config()
    set_seed(config.seed)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_train_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    print("Using train augmentations: RandomCrop(32, 4) and RandomHorizontalFlip")
    transform_train_list.insert(0, transforms.RandomHorizontalFlip())
    transform_train_list.insert(0, transforms.RandomCrop(32, padding=4))
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    raw_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    model_paso = CNN().to(device)
    model_seq = CNN().to(device)

    seq_acc, seq_time = train_sequential(config, model_seq, raw_loader, test_loader, device, rank)
    dist.barrier()

    if rank == 0: print("\n--- Starting PASO Training ---")
    paso_train_loader = PASODataLoader(raw_loader)
    optimizer = optim.AdamW(model_paso.parameters(), lr=config.learning_rate)
    paso_opt = PASOOptimizer(model_paso, optimizer, config)
    criterion = nn.CrossEntropyLoss()

    completed_steps, total_iters = 0, 0
    total_loss, total_correct, total_samples = 0.0, 0, 0
    start_time = time.time()
    log_memory("Model Loaded to GPU")
    if rank == 0: pbar = tqdm(total=config.max_steps, desc="PASO")

    while completed_steps < config.max_steps:
        begin_idx, rank_order = paso_opt.get_step_indices()
        local_metrics = (0.0, 0, 0)

        if begin_idx + rank_order < config.max_steps:
            inputs, labels = paso_train_loader[begin_idx + rank_order]
            inputs, labels = inputs.to(device), labels.to(device)

            paso_opt.optimizer.zero_grad()
            outputs = model_paso(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            _, pred = torch.max(outputs.data, 1)
            local_metrics = (loss.item(), (pred == labels).sum().item(), labels.size(0))

        progress, (w_loss, w_corr, w_total) = paso_opt.multi_step(local_metrics)
        
        completed_steps += progress
        total_loss += w_loss
        total_correct += w_corr
        total_samples += w_total
        total_iters += 1

        if rank == 0:
            pbar.update(progress)
            pbar.set_postfix({"Loss": f"{total_loss/max(1, completed_steps):.4f}"})

    dist.barrier()
    paso_time = time.time() - start_time
    
    if rank == 0:
        pbar.close()
        paso_acc = evaluate(model_paso, test_loader, device)
        
        print("\n" + "="*50)
        print(f"{'Metric':<20} | {'Sequential':<12} | {'PASO':<12}")
        print("-" * 50)
        print(f"{'Accuracy':<20} | {seq_acc:>11.2f}% | {paso_acc:>11.2f}%")
        print(f"{'Time (s)':<20} | {seq_time:>12.2f} | {paso_time:>12.2f}")
        print(f"{'Total Iterations':<20} | {config.max_steps:>12} | {total_iters:>12}")
        print("-" * 50)
        if paso_time > 0:
            print(f"Speed-up: {seq_time/paso_time:.2f}x")
        print("="*50)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()