# paso_dataloader.py
# -*- coding: utf-8 -*-

"""
PASODataLoader module, providing efficient data loading for the PASO strategy.
"""

from typing import Any
from torch.utils.data import DataLoader
import torch

class PASODataLoader:
    """
    A DataLoader wrapper that can efficiently fetch specific batches based on 
    iteration steps without occupying extra GPU memory.

    Attributes:
        dataloader (DataLoader): The original PyTorch DataLoader instance.
    """
    def __init__(self, dataloader: DataLoader):
        """
        Initialize PASODataLoader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader instance.
        
        Raises:
            TypeError: If input is not a DataLoader instance.
            ValueError: If DataLoader is empty.
        """
        if not isinstance(dataloader, DataLoader):
            raise TypeError("Input must be a DataLoader instance.")
        if len(dataloader) == 0:
            raise ValueError("DataLoader cannot be empty.")

        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.collate_fn = dataloader.collate_fn
        self.num_batches = len(dataloader)
        
        # Cache the list of batch indices to avoid recalculation.
        # This list only contains integer indices, so memory usage is negligible.
        self.batch_sampler_iter = iter(self.dataloader.batch_sampler)
        self._batch_indices_list = list(self.dataloader.batch_sampler)

    def __len__(self) -> int:
        """Returns the total number of batches."""
        return self.num_batches

    def __getitem__(self, step: int) -> Any:
        """
        Fetch the corresponding batch data based on the training step.

        Args:
            step (int): Current training iteration step.

        Returns:
            Any: A batch of data.
        """
        if not isinstance(step, int):
            raise TypeError(f"Step must be an integer, got {type(step).__name__}")
            
        # 1. Calculate the target batch ID for the current epoch logic
        target_batch_id = step % self.num_batches

        # 2. Get batch indices from cached list
        batch_indices = self._batch_indices_list[target_batch_id]

        # 3. Retrieve samples from the dataset
        # Note: For very large datasets on disk, this might need optimization,
        # but for CIFAR10 in memory, this is fast.
        samples = [self.dataset[i] for i in batch_indices]

        # 4. Collate samples into a batch
        batch = self.collate_fn(samples)
        return batch


# paso_optimizer.py
# -*- coding: utf-8 -*-

"""
PASOOptimizer module implementing the PASO algorithm.
Optimized for high-efficiency communication using flattened tensors.
"""

from typing import Tuple, Any, List
import time
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.nn.utils import vector_to_parameters

class PASOOptimizer:
    """
    An optimizer wrapper implementing the PASO algorithm.
    It coordinates communication, synchronization, and model updates across ranks.
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: Optimizer, 
                 config: Any):
        """
        Initialize PASOOptimizer.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (Optimizer): The standard PyTorch optimizer.
            config (object): Configuration object containing PASO parameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        # Ensure model is on the correct device
        self.device = next(model.parameters()).device
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.begin_idx = 0
        self.end_idx = min(self.begin_idx + self.world_size, self.config.max_steps)
        
        self.start_rank = 0
        self.thresh = config.threshold

        self._update_rank_maps()

        # --- Memory & Communication Optimization ---
        # Pre-allocate buffers for flattening parameters and gradients.
        self.total_param_num = sum(p.numel() for p in self.model.parameters())
        self.param_buffer = torch.empty(self.total_param_num, device=self.device)
        self.grad_buffer = torch.empty(self.total_param_num, device=self.device)
        
        # Buffer for error reception
        self.error_buffer = torch.zeros(1, device=self.device)

    def get_step_indices(self) -> Tuple[int, int]:
        return self.begin_idx, self.rank_to_order_map[self.rank]

    def _update_rank_maps(self):
        self.rank_to_order_map = [(r - self.start_rank) % self.world_size for r in range(self.world_size)]
        self.order_to_rank_map = [(self.start_rank + o) % self.world_size for o in range(self.world_size)]

    def _clone_optimizer_state(self, src_rank: int, dest_rank: int):
        group = dist.new_group(ranks=sorted([src_rank, dest_rank]))
        try:
            if self.rank not in [src_rank, dest_rank]:
                return

            objects_to_broadcast = [None]
            if self.rank == src_rank:
                objects_to_broadcast = [self.optimizer.state_dict()]
            
            dist.broadcast_object_list(objects_to_broadcast, src=src_rank, group=group)
            
            if self.rank == dest_rank:
                self.optimizer.load_state_dict(objects_to_broadcast[0])
        finally:
            dist.destroy_process_group(group)

    def _flatten_params(self, target_buffer: torch.Tensor):
        """Helper to flatten parameters into a target buffer (Zero allocation)."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            target_buffer[offset:offset+numel].copy_(p.data.view(-1))
            offset += numel

    def _flatten_grads(self, target_buffer: torch.Tensor):
        """Helper to flatten gradients into a single buffer."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            if p.grad is not None:
                target_buffer[offset:offset+numel].copy_(p.grad.data.view(-1))
            else:
                target_buffer[offset:offset+numel].zero_()
            offset += numel

    def _unflatten_grads(self, source_buffer: torch.Tensor):
        """Helper to restore gradients from a single buffer."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            p.grad.data.copy_(source_buffer[offset:offset+numel].view_as(p.data))
            offset += numel

    def multi_step(self, local_metrics: Tuple[float, int, int]) -> Tuple[int, Tuple[float, int, int]]:
        rank_order = self.rank_to_order_map[self.rank]
        parallel_len = self.end_idx - self.begin_idx

        # 1. Gather metrics
        metric_tensors = torch.tensor(local_metrics, dtype=torch.float32, device=self.device)
        gather_list = [torch.zeros_like(metric_tensors) for _ in range(self.world_size)] if rank_order == 0 else None
        dist.gather(metric_tensors, gather_list, dst=self.order_to_rank_map[0])

        sync_data = torch.empty(5, device=self.device, dtype=torch.float32)

        # ======================= Master Rank (Order 0) Logic =======================
        if rank_order == 0:
            sync_point = None
            cumulative_error, window_loss, window_correct, window_total = 0.0, 0.0, 0, 0
            
            st = time.time()
            
            for i in range(parallel_len):
                step = self.begin_idx + i
                self.optimizer.step()

                if i + 1 < parallel_len:
                    target_rank = self.order_to_rank_map[i + 1]

                    # --- OPTIMIZATION: Send Flattened Parameters ---
                    # 1. Flatten to buffer using custom helper
                    self._flatten_params(self.param_buffer)
                    
                    # 2. Send buffer
                    dist.send(tensor=self.param_buffer, dst=target_rank)
                    
                    # 3. Receive Error
                    dist.recv(tensor=self.error_buffer, src=target_rank)
                    error = self.error_buffer.item()
                    
                    # 4. Receive Flattened Gradients
                    dist.recv(tensor=self.grad_buffer, src=target_rank)
                    self._unflatten_grads(self.grad_buffer)

                else:
                    error = 2 * self.thresh
                
                # Adaptivity Logic
                if self.config.adaptivity_type == 'mean':
                    cumulative_error += error / parallel_len
                if self.config.adaptivity_type == 'median':
                    if i == parallel_len // 2:
                        cumulative_error = error
          
                # Metrics
                if sync_point is None or step < sync_point:
                    metrics = gather_list[self.order_to_rank_map[i]]
                    window_loss += metrics[0].item()
                    window_correct += metrics[1].item()
                    window_total += metrics[2].item()
                
                if sync_point is None and (error > self.thresh or i == parallel_len - 1):
                    sync_point = step + 1
                    new_start_rank = self.order_to_rank_map[i + 1] if i + 1 < parallel_len else self.order_to_rank_map[0]
            
            print(f"rank {self.rank} loop time: {time.time() - st} s")
            self.thresh = self.thresh * self.config.ema_decay + cumulative_error * (1 - self.config.ema_decay)
            sync_data.copy_(torch.tensor([new_start_rank, sync_point, window_loss, window_correct, window_total], device=self.device))
        
        # ======================= Worker Ranks Logic =======================
        elif rank_order < parallel_len:
            with torch.no_grad():
                # 1. Receive Flattened Parameters
                dist.recv(tensor=self.param_buffer, src=self.order_to_rank_map[0])
                
                # 2. Calculate Error 
                # Optimization: Reuse grad_buffer to store current local params temporarily 
                # (since grad_buffer is not needed until step 5).
                self._flatten_params(self.grad_buffer) # Flatten CURRENT params into grad_buffer
                
                # Calculate Norm: ||Current - Received||^2
                error = torch.linalg.norm(self.grad_buffer - self.param_buffer).pow(2).item()
                
                # 3. Update local model with received parameters
                vector_to_parameters(self.param_buffer, self.model.parameters())
                
                # Normalize Error
                error = (error / self.total_param_num) * 1e6
            
            # 4. Send Error back
            dist.send(tensor=torch.tensor([error], device=self.device), dst=self.order_to_rank_map[0])
            
            # 5. Send Flattened Gradients back
            # Now we actually fill grad_buffer with gradients
            self._flatten_grads(self.grad_buffer)
            dist.send(tensor=self.grad_buffer, dst=self.order_to_rank_map[0])
        
        # ======================= Synchronization =======================
        dist.broadcast(sync_data, src=self.order_to_rank_map[0])
        new_start_rank, sync_point, wl, wc, wt = sync_data.tolist()
        
        if self.start_rank != int(new_start_rank):
            self._clone_optimizer_state(src_rank=self.start_rank, dest_rank=int(new_start_rank))

        new_begin_idx = int(sync_point)
        new_end_idx = min(new_begin_idx + parallel_len, self.config.max_steps)
        progress = new_begin_idx - self.begin_idx
        self.begin_idx = new_begin_idx
        self.end_idx = new_end_idx
        self.start_rank = int(new_start_rank)
        self._update_rank_maps()
        
        return progress, (wl, int(wc), int(wt))

"""
Main script for PASO parallel training.
"""
import os
import random
import time
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

# from paso_dataloader import PASODataLoader
# from paso_optimizer import PASOOptimizer

# --- Global Config & Definitions ---
DATA_ROOT = './PASO/data'

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

class Config:
    max_steps: int = 1000
    learning_rate: float = 0.00084439
    momentum: float = 0.9
    batch_size: int = 4096
    threshold: float = 1e-5
    ema_decay: float = 0.95
    adaptivity_type: str = 'median'
    seed: int = 42
    eval_interval: int = 1

def main():
    config = Config()
    set_seed(config.seed)

    # Initialize Distributed Process Group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    # Transforms & Datasets
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Ensure download only happens on one rank to avoid race conditions (optional safety)
    if rank == 0:
        torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    dist.barrier()

    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform)
    # Important: num_workers > 0 for efficiency
    paso_train_loader = PASODataLoader(DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True))
    
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    base_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Init PASO Optimizer
    paso_optimizer = PASOOptimizer(model=model, optimizer=base_optimizer, config=config)

    completed_steps = 0
    total_iters  = 0
    total_loss, total_correct, total_samples = 0.0, 0, 0
    start_time = time.time()
    
    if rank == 0:
        pbar = tqdm(total=config.max_steps, unit="step")

    # Main Training Loop
    while completed_steps < config.max_steps:
        # Get step index for current rank
        begin_idx, rank_order = paso_optimizer.get_step_indices()
        infor_gatter_rank = paso_optimizer.order_to_rank_map[0]
        local_metrics = (0.0, 0, 0)

        # Only execute if the rank has a task in the current window
        if begin_idx + rank_order < config.max_steps:
            # 1. Forward/Backward
            inputs, labels = paso_train_loader[begin_idx + rank_order]
            inputs, labels = inputs.to(device), labels.to(device)
            
            paso_optimizer.optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            _, predicted = torch.max(outputs.data, 1)
            local_metrics = (loss.item(), (predicted == labels).sum().item(), labels.size(0))

        # 2. Multi-step communication & Sync
        progress, (win_loss, win_correct, win_total) = paso_optimizer.multi_step(local_metrics)
        
        # Update stats
        completed_steps += progress
        total_loss += win_loss
        total_correct += win_correct
        total_samples += win_total
        total_iters += 1

        # Global stats broadcasting for logging
        global_matric_tensor = torch.zeros(1, dtype=torch.float64, device=device)
        # Note: Actual evaluation logic can be uncommented here if needed
        
        dist.broadcast(global_matric_tensor, src=infor_gatter_rank)                
        
        if rank == 0 and completed_steps % (config.eval_interval) == 0:
            train_avg_loss = total_loss / completed_steps if completed_steps > 0 else 0
            train_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
            
            pbar.update(progress)
            postfix_stats = {
                "Train_Loss": f"{train_avg_loss:.4f}",
                "Train_Acc": f"{train_accuracy:.4f}"
            }
            pbar.set_description(f'Iters: {total_iters}')
            pbar.set_postfix(postfix_stats)

    dist.barrier()
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    
    if rank == 0:
        pbar.close()
        
    if paso_optimizer.rank_to_order_map[rank] == 0: 
        print("\nFinal Evaluation:")
        final_accuracy = evaluate(model, test_loader, device)
        print(f"\nTraining completed in {elapsed_str}")
        print(f"Final test accuracy: {final_accuracy:.2f}%")  
        print(f"Total iterations: {total_iters} (vs {config.max_steps} sequential iterations)")
        print(f"Effective speed-up: {config.max_steps/max(1, total_iters):.2f}x")
    
    # Cleanup Process Group
    dist.destroy_process_group()

if __name__ == '__main__':
    main()