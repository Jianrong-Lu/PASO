# paso_dataloader.py

"""
PASODataLoader module, providing efficient data loading for PASO parallel strategy.
"""

from typing import Any
from torch.utils.data import DataLoader
import time

class PASODataLoader:
    """
    A DataLoader wrapper that can efficiently retrieve specific batches of data based on iteration steps,
    without occupying additional GPU memory.

    This class pre-caches batch indices during initialization, enabling fast direct access to any batch.

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
            raise TypeError("Input should be a DataLoader instance.")
        if len(dataloader) == 0:
            raise ValueError("DataLoader cannot be empty.")

        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.collate_fn = dataloader.collate_fn
        self.num_batches = len(dataloader)
        
        # Cache the index list for each batch to avoid repeated computation.
        # This list only contains integer indices, with very small memory footprint.
        self._batch_indices_list = list(self.dataloader.batch_sampler)

    def __len__(self) -> int:
        """Return the total number of batches in the DataLoader."""
        return self.num_batches

    def __getitem__(self, step: int) -> Any:
        """
        Retrieve the corresponding batch data based on the given training iteration step.

        Args:
            step (int): Current training iteration step.

        Returns:
            Any: A batch of data.
            
        Raises:
            TypeError: If step is not an integer.
            IndexError: If the calculated batch ID is out of range.
        """
        if not isinstance(step, int):
            raise TypeError(f"Step must be an integer, but got {type(step).__name__}")
            
        # 1. Calculate the target batch ID corresponding to the current epoch
        target_batch_id = step % self.num_batches

        # 2. Get the data indices for this batch from the cached list
        batch_indices = self._batch_indices_list[target_batch_id]

        # 3. Retrieve corresponding samples from the dataset
        samples = [self.dataset[i] for i in batch_indices]

        # 4. Use the dataloader's collate_fn to pack samples into a batch
        batch = self.collate_fn(samples)
        return batch

# paso_optimizer.py

"""
PASOOptimizer module, implementing the PASO algorithm.
This version keeps forward/backward propagation in the main loop, with multi_step responsible for all subsequent
communication, synchronization, and model updates.
"""

from typing import Tuple, Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer

class PASOOptimizer:
    """
    An optimizer wrapper that implements the PASO algorithm.

    It assumes gradients have already been computed in the main loop via `loss.backward()`.
    The `multi_step` method is responsible for coordinating all ranks to complete all
    communication, model updates, and state synchronization within a PASO window.
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: Optimizer, 
                 config: Any):
        """
        Initialize PASOOptimizer.

        Args:
            model (torch.nn.Module): Model to be trained.
            optimizer (Optimizer): Wrapped standard PyTorch optimizer (e.g., SGD, Adam).
            config (object): Configuration object containing PASO parameters.
        """
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

    def get_step_indices(self) -> Tuple[int, int]:
        """
        Get information needed for the current rank to process the required data batch in this window.

        Returns:
            Tuple[int, int]: Returns a tuple (begin_idx, rank_order),
                             which the main loop can use to calculate the index for `train_loader`.
        """
        return self.begin_idx, self.rank_to_order_map[self.rank]

    def _update_rank_maps(self):
        self.rank_to_order_map = [(r - self.start_rank) % self.world_size for r in range(self.world_size)]
        self.order_to_rank_map = [(self.start_rank + o) % self.world_size for o in range(self.world_size)]

    def _clone_optimizer_state(self, src_rank: int, dest_rank: int):
        group = dist.new_group(ranks=sorted([src_rank, dest_rank]))
        if self.rank not in [src_rank, dest_rank]:
            return
        objects_to_broadcast = [None]
        if self.rank == src_rank:
            objects_to_broadcast = [self.optimizer.state_dict()]
        dist.broadcast_object_list(objects_to_broadcast, src=src_rank, group=group)
        if self.rank == dest_rank:
            self.optimizer.load_state_dict(objects_to_broadcast[0])
        # dist.destroy_process_group(group)
        
    def multi_step(self, local_metrics: Tuple[float, int, int]) -> Tuple[int, Tuple[float, int, int]]:
        """
        Execute one PASO window training flow under the condition that gradients have been computed.

        Args:
            local_metrics (Tuple[float, int, int]): (loss, correct, total) computed by the current rank
                                                    after forward/backward propagation.

        Returns:
            Tuple[int, Tuple[float, int, int]]: Returns (progress, (window_loss, window_correct, window_total)).
        """
        
        rank_order = self.rank_to_order_map[self.rank]
        parallel_len = self.end_idx - self.begin_idx

        
        # 1. Collect statistical information from all ranks
        metric_tensors = torch.tensor(local_metrics, dtype=torch.float32, device=self.device)
        gather_list = [torch.zeros_like(metric_tensors) for _ in range(self.world_size)] if rank_order == 0 else None
        dist.gather(metric_tensors, gather_list, dst=self.order_to_rank_map[0])

        sync_data = torch.empty(5, device=self.device, dtype=torch.float32)

        # ======================= Main rank (order 0) logic =======================
        if rank_order == 0:
            sync_point = None
            cumulative_error, window_loss, window_correct, window_total = 0.0, 0.0, 0, 0
            st = time.time()
            for i in range(parallel_len):
                step = self.begin_idx + i
                self.optimizer.step()
                if i + 1 < parallel_len:
                    for param in self.model.parameters():
                        
                        dist.send(tensor=param.data, dst=self.order_to_rank_map[i + 1])
                        
                    error_tensor = torch.zeros(1, device=self.device)
                    dist.recv(tensor=error_tensor, src=self.order_to_rank_map[i + 1])
                    error = error_tensor.item()
                    
                    for param in self.model.parameters():
                        if param.grad is not None:
                            dist.recv(tensor=param.grad.data, src=self.order_to_rank_map[i + 1])
                else:
                    error = 2 * self.thresh
                
                if self.config.adaptivity_type == 'mean':
                    cumulative_error += error / parallel_len
                if self.config.adaptivity_type == 'median':
                    if i == parallel_len // 2:
                        cumulative_error = error
         
                if sync_point is None or step < sync_point:
                    metrics = gather_list[self.order_to_rank_map[i]]
                    window_loss += metrics[0].item()
                    window_correct += metrics[1].item()
                    window_total += metrics[2].item()
                
                if sync_point is None and (error > self.thresh or i == parallel_len - 1):
                    sync_point = step + 1
                    new_start_rank = self.order_to_rank_map[i + 1] if i + 1 < parallel_len else self.order_to_rank_map[0]
            print(f"rank {self.rank} sent param.data  to all rank: {time.time() - st} s!!!")
            self.thresh = self.thresh * self.config.ema_decay + cumulative_error * (1 - self.config.ema_decay)
            sync_data.copy_(torch.tensor([new_start_rank, sync_point, window_loss, window_correct, window_total], device=self.device))
        
        # ======================= Other ranks logic =======================
        elif rank_order < parallel_len:
            with torch.no_grad():
                error, total_numel = 0.0, 0
                for param in self.model.parameters():
                    received_param = torch.empty_like(param.data)
                    dist.recv(tensor=received_param, src=self.order_to_rank_map[0])
                    error += torch.linalg.norm(param.data - received_param).pow(2).item()
                    param.data = received_param.clone()
                    del received_param
                    total_numel += param.numel()
                error = (error / total_numel) * 1e6
            
            dist.send(tensor=torch.tensor([error], device=self.device), dst=self.order_to_rank_map[0])
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.send(tensor=param.grad.data, dst=self.order_to_rank_map[0])
        
        
        # 3. Synchronization
        dist.broadcast(sync_data, src=self.order_to_rank_map[0])
        new_start_rank, sync_point, wl, wc, wt = sync_data.tolist()
        
        if self.start_rank != int(new_start_rank):
            st = time.time()
            self._clone_optimizer_state(src_rank=self.start_rank, dest_rank=int(new_start_rank))
            print(f"rank {self.rank} _clone_optimizer_state  to new start rank {int(new_start_rank)}: {time.time() - st} s")

        new_begin_idx = int(sync_point)
        new_end_idx = min(new_begin_idx + parallel_len, self.config.max_steps)
        progress = new_begin_idx - self.begin_idx
        self.begin_idx = new_begin_idx
        self.end_idx = new_end_idx
        self.start_rank = int(new_start_rank)
        self._update_rank_maps()
        
        return progress, (wl, int(wc), int(wt))

# main.py

"""
Main script for data parallel training using PASO strategy.
This version keeps forward/backward propagation in the main loop for a more traditional code structure.
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

# --- Global configuration & Model definition & Helper functions (same as before) ---
DATA_ROOT = './PASO/data'

class CNN(nn.Module):
    # ... (unchanged code) ...
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
    # ... (unchanged code) ...
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device):
    # ... (unchanged code) ...
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
    # if dist.get_rank() == 0:
    # print(f'\nTest set accuracy (Test Accuracy): {accuracy:.2f}%')
    return accuracy

class Config:
    max_steps: int = 1000
    learning_rate: float = 0.001
    momentum: float = 0.9
    batch_size: int = 512
    threshold: float = 1e-5
    ema_decay: float = 0.95
    adaptivity_type: str = 'mean'
    seed: int = 42
    eval_interval: int = 1

def main():
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
    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    paso_train_loader = PASODataLoader(DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=16,pin_memory=True, persistent_workers=True))
    
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=4,pin_memory=True, persistent_workers=True)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    base_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Initialize PASO optimizer, now more concise
    paso_optimizer = PASOOptimizer(model=model, optimizer=base_optimizer, config=config)

    completed_steps = 0
    total_iters  = 0
    total_loss, total_correct, total_samples = 0.0, 0, 0
    start_time = time.time()
    
    if rank == 0:
        pbar = tqdm(total=config.max_steps, unit="step")
    while completed_steps < config.max_steps:
        # Get the data index that the current rank needs to process from the optimizer
        begin_idx, rank_order = paso_optimizer.get_step_indices()
        infor_gatter_rank = paso_optimizer.order_to_rank_map[0]
        local_metrics = (0.0, 0, 0)
        # Only execute computation if the rank has tasks in the current window
        if begin_idx + rank_order < config.max_steps:
            # ===================================================================
            # Forward and backward propagation logic remains in the main loop
            # ===================================================================
            inputs, labels = paso_train_loader[begin_idx + rank_order]
            inputs, labels = inputs.to(device), labels.to(device)
            
            paso_optimizer.optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            _, predicted = torch.max(outputs.data, 1)
            local_metrics = (loss.item(), (predicted == labels).sum().item(), labels.size(0))

        # ===================================================================
        # Call multi_step, which handles all complex parallel communication and synchronization
        # ===================================================================
        progress, (win_loss, win_correct, win_total) = paso_optimizer.multi_step(local_metrics)
        
        # Update global statistics and completed steps
        completed_steps += progress
        total_loss += win_loss
        total_correct += win_correct
        total_samples += win_total
        total_iters += 1

        global_matric_tensor = torch.zeros(1, dtype=torch.float64, device=device)
        if rank_order == 0 and completed_steps % (config.eval_interval) == 0:  # Evaluate every eval_interval
            pass
            # test_accur = evaluate(model, test_loader, device)
            # model.train()  # Switch back to training mode
            # global_matric_tensor.copy_(torch.tensor([
            #         test_accur
            #         # total_loss,
            #         # total_correct,
            #         # total_samples,
            #         # total_iters,
            #         # progress # Pass progress as well, for pbar.update()
            #     ], dtype=torch.float64, device=device))
        
        dist.broadcast(global_matric_tensor, src=infor_gatter_rank)                
        if rank == 0 and completed_steps % (config.eval_interval) == 0:  # Display every eval_interval
            train_avg_loss = total_loss / completed_steps
            train_accuracy = 100 * total_correct / total_samples
            test_accuracy = global_matric_tensor[0].item()
            pbar.update(progress)
            postfix_stats = {}
            postfix_stats["Train_Loss"] =  f"{train_avg_loss:.4f}"
            postfix_stats["Train_Acc"] = f"{train_accuracy:.4f}"
            # postfix_stats["Test_Acc"] =  f"{test_accuracy:.4f}"
            # postfix_stats["Test_Loss"] =  f"{test_accuracy:.4f}"
            pbar.set_description(
                    f'Iters: {total_iters}'
                )
            pbar.set_postfix(postfix_stats)

    dist.barrier()
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    if rank == 0:
        pbar.close()
    if rank_order == 0: 
        print("\nFinal Evaluation:")
        final_accuracy = evaluate(model, test_loader, device)
        print(f"\nTraining completed in {elapsed_str}")
        print(f"Final test accuracy: {final_accuracy:.2f}%")
        # print(f"Final test precision: {metrics['precision_macro']:.2f}%")
        # print(f"Final test recall: {metrics['recall_macro']:.2f}%")
        # print(f"Final test f1-score: {metrics['f1_macro']:.2f}%")    
        print(f"Total iterations: {total_iters} (vs {config.max_steps} sequential iterations)")
        print(f"Effective speed-up: {config.max_steps/total_iters:.2f}x")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()