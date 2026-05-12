# -*- coding: utf-8 -*-
"""
Memory-profiling fork of cv_nccl.py.

Purpose: pinpoint where the GPU memory goes when running PASO with NCCL.
Specifically, distinguish:
  (a) model weights
  (b) flat_buffer + grad_buffer   (app-layer NCCL staging buffers, fixed ~400MB)
  (c) small metric/sync buffers
  (d) supported_keys optimizer-state pre-allocation  (suspect: 4x model size)
  (e) NCCL library-internal buffers (only allocated after first collective)

What changed vs cv_nccl.py:
  - Skips train_sequential to save time.
  - PASO loop exits after PROFILE_NUM_STEPS iterations.
  - PASOOptimizer.__init__ takes an optional tracker and checkpoints memory
    between each allocation phase. The actual allocations are unchanged.
  - MemTracker prints a final stage-by-stage table on rank 0.

Run:
  torchrun --nproc_per_node=<N> --master_port=12345 cv_nccl_profile.py
"""

import os
import random
import time
import copy
from typing import Any, Tuple, Optional, List
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


PROFILE_NUM_STEPS = int(os.environ.get("PROFILE_NUM_STEPS", 100))
PROFILE_DUMP_SNAPSHOT = True
SNAPSHOT_PATH = "./mem_snapshot_rank{rank}.pickle"


# ==============================================================================
# Memory tracker
# ==============================================================================

class MemTracker:
    def __init__(self, rank: int, world_size: int, verbose: bool = True):
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.history: List[dict] = []
        self._last_allocated = 0
        self._baseline = 0

    def set_baseline(self):
        torch.cuda.synchronize()
        self._baseline = torch.cuda.memory_allocated()
        self._last_allocated = self._baseline

    def checkpoint(self, tag: str):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        delta = allocated - self._last_allocated
        cumulative = allocated - self._baseline
        row = {
            "tag": tag,
            "allocated_MB": allocated / 1024**2,
            "delta_MB": delta / 1024**2,
            "cumulative_MB": cumulative / 1024**2,
            "reserved_MB": reserved / 1024**2,
            "peak_MB": peak / 1024**2,
        }
        self.history.append(row)
        self._last_allocated = allocated
        if self.verbose and self.rank == 0:
            print(
                f"[MEM rank={self.rank}] {tag:<55} "
                f"alloc={row['allocated_MB']:>9.2f}MB  "
                f"Δ={row['delta_MB']:>+9.2f}MB  "
                f"cum={row['cumulative_MB']:>+9.2f}MB  "
                f"peak={row['peak_MB']:>9.2f}MB"
            )

    def report(self):
        if self.rank != 0:
            return
        print("\n" + "=" * 110)
        print(f"MEMORY REPORT (rank=0, world_size={self.world_size})")
        print("=" * 110)
        print(
            f"{'Stage':<55} | {'Alloc(MB)':>10} | {'Delta(MB)':>10} | {'Cum(MB)':>10} | {'Peak(MB)':>10}"
        )
        print("-" * 110)
        for h in self.history:
            print(
                f"{h['tag']:<55} | {h['allocated_MB']:>10.2f} | {h['delta_MB']:>+10.2f} | "
                f"{h['cumulative_MB']:>+10.2f} | {h['peak_MB']:>10.2f}"
            )
        print("=" * 110)


# ==============================================================================
# 1. PASO DATA LOADER  (verbatim from cv_nccl.py)
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
# 2. PASO OPTIMIZER (instrumented; allocations unchanged)
# ==============================================================================

class PASOOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        config: Any,
        tracker: Optional[MemTracker] = None,
    ):
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

        if tracker:
            tracker.checkpoint("PASOOpt: entry (bookkeeping only)")

        # ---- Bucketing setup ----
        self.param_list = [p for p in model.parameters()]
        self.total_params = sum(p.numel() for p in self.param_list)
        limit_size = 50 * 1024 * 1024
        self.bucket_size = min(self.total_params, limit_size)

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

        actual_max_bucket = max(sum(p.numel() for p in b) for b in self.buckets)

        if tracker and self.rank == 0:
            print(
                f"[INFO rank=0] total_params={self.total_params:,}  "
                f"buckets={len(self.buckets)}  max_bucket_elements={actual_max_bucket:,}  "
                f"per-buffer fp32={actual_max_bucket * 4 / 1024**2:.2f}MB"
            )

        # ---- (B) flat_buffer + grad_buffer ----
        self.flat_buffer = torch.zeros(actual_max_bucket, device=self.device)
        self.grad_buffer = torch.zeros(actual_max_bucket, device=self.device)
        if tracker:
            tracker.checkpoint("(B) flat_buffer + grad_buffer")

        # ---- (C) small metric/sync buffers ----
        self.error_tensor = torch.zeros(1, device=self.device)
        self.local_metric_buffer = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.gather_buffer_list = [
            torch.zeros(3, device=self.device, dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        self.sync_data = torch.zeros(5, device=self.device, dtype=torch.float32)
        if tracker:
            tracker.checkpoint("(C) small metric/sync buffers")

        # ---- (D) Optimizer state pre-allocation (suspected hot spot) ----
        self.supported_keys = ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', 'momentum_buffer']

        for p in self.param_list:
            if p not in self.optimizer.state:
                self.optimizer.state[p] = {}
            if 'step' not in self.optimizer.state[p]:
                self.optimizer.state[p]['step'] = torch.tensor(0.0, device=self.device)
            for key in self.supported_keys:
                if key not in self.optimizer.state[p]:
                    self.optimizer.state[p][key] = torch.zeros_like(p, device=self.device)
        if tracker:
            tracker.checkpoint(f"(D) optimizer state x{len(self.supported_keys)} per-param")

        # ---- (E) Protocol buffers ----
        self.protocol_buffer = torch.zeros(
            1 + len(self.supported_keys), device=self.device, dtype=torch.float32
        )
        self.step_buffer = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.found_keys_mask = [0.0] * len(self.supported_keys)
        if tracker:
            tracker.checkpoint("(E) protocol/step buffers")

    def _process_buckets(self, mode='send_params', target_rank=None):
        for bucket in self.buckets:
            b_size = sum(p.numel() for p in bucket)
            active_buffer = self.flat_buffer[:b_size]
            active_grad_buffer = self.grad_buffer[:b_size]

            if mode == 'send_params':
                offset = 0
                for p in bucket:
                    active_buffer[offset:offset + p.numel()].copy_(p.data.view(-1))
                    offset += p.numel()
                dist.send(tensor=active_buffer, dst=target_rank)
            elif mode == 'recv_params':
                dist.recv(tensor=active_buffer, src=target_rank)
                offset = 0
                for p in bucket:
                    p.data.copy_(active_buffer[offset:offset + p.numel()].view(p.shape))
                    offset += p.numel()
            elif mode == 'send_grads':
                offset = 0
                for p in bucket:
                    if p.grad is not None:
                        active_grad_buffer[offset:offset + p.numel()].copy_(p.grad.data.view(-1))
                    else:
                        active_grad_buffer[offset:offset + p.numel()].zero_()
                    offset += p.numel()
                dist.send(tensor=active_grad_buffer, dst=target_rank)
            elif mode == 'recv_grads':
                dist.recv(tensor=active_grad_buffer, src=target_rank)
                offset = 0
                for p in bucket:
                    if p.grad is not None:
                        p.grad.data.copy_(active_grad_buffer[offset:offset + p.numel()].view(p.shape))
                    offset += p.numel()

    def get_step_indices(self) -> Tuple[int, int]:
        return self.begin_idx, self.rank_to_order_map[self.rank]

    def _update_rank_maps(self):
        self.rank_to_order_map = [(r - self.start_rank) % self.world_size for r in range(self.world_size)]
        self.order_to_rank_map = [(self.start_rank + o) % self.world_size for o in range(self.world_size)]

    def _sync_optimizer_state(self, src_rank: int, dest_rank: int):
        if self.rank != src_rank and self.rank != dest_rank:
            return

        SUPPORTED_KEYS = self.supported_keys
        protocol_tensor = self.protocol_buffer
        protocol_tensor.zero_()

        if self.rank == src_rank:
            global_step = 0.0
            for i in range(len(self.found_keys_mask)):
                self.found_keys_mask[i] = 0.0
            first_p = self.param_list[0]
            if 'step' in self.optimizer.state[first_p]:
                v = self.optimizer.state[first_p]['step']
                global_step = v.item() if isinstance(v, torch.Tensor) else float(v)
            for idx in range(len(SUPPORTED_KEYS)):
                self.found_keys_mask[idx] = 1.0
            protocol_tensor[0] = global_step
            for i, val in enumerate(self.found_keys_mask):
                protocol_tensor[1 + i] = val
            dist.send(protocol_tensor, dst=dest_rank)
        elif self.rank == dest_rank:
            dist.recv(protocol_tensor, src=src_rank)

        current_step = protocol_tensor[0].item()
        active_key_indices = []
        for i in range(len(SUPPORTED_KEYS)):
            if protocol_tensor[1 + i] > 0.5:
                active_key_indices.append(i)

        if self.rank == dest_rank:
            self.step_buffer.fill_(current_step)
            for p in self.param_list:
                self.optimizer.state[p]['step'].copy_(self.step_buffer)

        for key_idx in active_key_indices:
            key_name = SUPPORTED_KEYS[key_idx]
            for bucket in self.buckets:
                b_size = sum(p.numel() for p in bucket)
                active_buffer = self.flat_buffer[:b_size]
                if self.rank == src_rank:
                    offset = 0
                    for p in bucket:
                        active_buffer[offset:offset + p.numel()].copy_(
                            self.optimizer.state[p][key_name].view(-1)
                        )
                        offset += p.numel()
                    dist.send(active_buffer, dst=dest_rank)
                elif self.rank == dest_rank:
                    dist.recv(active_buffer, src=src_rank)
                    offset = 0
                    for p in bucket:
                        self.optimizer.state[p][key_name].view(-1).copy_(
                            active_buffer[offset:offset + p.numel()]
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
                        target_part = active_buffer[offset:offset + p.numel()]
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
# 3. TRAINING INFRASTRUCTURE  (verbatim from cv_nccl.py)
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
    """Scaled CNN. fc_width / fc_layers come from env so both profiles match.

    Layout: conv → conv → flatten → Linear(4096, W) → fc_layers × Linear(W, W) → Linear(W, 10)
    fc_width=128, fc_layers=0 reproduces the original ~545K-param model.
    """

    def __init__(self, fc_width: int = 128, fc_layers: int = 0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        in_dim = 64 * 8 * 8
        fc_modules = [nn.Linear(in_dim, fc_width), nn.ReLU()]
        for _ in range(fc_layers):
            fc_modules += [nn.Linear(fc_width, fc_width), nn.ReLU()]
        fc_modules.append(nn.Linear(fc_width, 10))
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 4. MAIN  (profile-only: skip sequential, short PASO loop)
# ==============================================================================

def main():
    DATA_ROOT = './data'
    config = Config()
    set_seed(config.seed)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    tracker = MemTracker(rank=rank, world_size=world_size)

    if PROFILE_DUMP_SNAPSHOT:
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    if rank == 0:
        print(f"\n{'=' * 110}\nMEMORY PROFILE RUN  world_size={world_size}  device={device}\n{'=' * 110}\n")

    tracker.set_baseline()
    tracker.checkpoint("baseline (post init_process_group)")

    # --- data ---
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    raw_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    tracker.checkpoint("data loaded (CPU; should be ~0)")

    # --- (A) model to GPU ---
    fc_width = int(os.environ.get("CNN_FC_WIDTH", 128))
    fc_layers = int(os.environ.get("CNN_FC_LAYERS", 0))
    model_paso = CNN(fc_width=fc_width, fc_layers=fc_layers).to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in model_paso.parameters())
        print(
            f"[CNN cfg] fc_width={fc_width}  fc_layers={fc_layers}  "
            f"total_params={n_params:,}  fp32_size={n_params * 4 / 1024**2:.2f}MB"
        )
    tracker.checkpoint("(A) model on GPU")

    # --- optimizer (lazy state init by default) ---
    optimizer = optim.AdamW(model_paso.parameters(), lr=config.learning_rate)
    tracker.checkpoint("AdamW constructed (states lazy, expect ~0)")

    # --- PASOOptimizer: phases B/C/D/E checkpointed inside __init__ ---
    paso_opt = PASOOptimizer(model_paso, optimizer, config, tracker=tracker)
    criterion = nn.CrossEntropyLoss()

    paso_train_loader = PASODataLoader(raw_loader)

    tracker.checkpoint("post PASOOptimizer init (steady before any NCCL collective)")

    # --- run a few iterations to see first-collective and steady-state ---
    completed_steps, total_iters = 0, 0
    torch.cuda.synchronize()
    loop_start_time = time.time()
    for it in range(PROFILE_NUM_STEPS):
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

        if it == 0:
            tracker.checkpoint("after first forward+backward (pre-NCCL)")

        progress, _ = paso_opt.multi_step(local_metrics)
        completed_steps += progress
        total_iters += 1

        if it == 0:
            tracker.checkpoint("after FIRST multi_step (NCCL collectives active)")
        elif it + 1 in {10, 50, 100, 500, 1000, PROFILE_NUM_STEPS}:
            tracker.checkpoint(f"after multi_step #{it + 1}")

    dist.barrier()
    torch.cuda.synchronize()
    loop_elapsed = time.time() - loop_start_time
    tracker.checkpoint("end (post barrier)")
    tracker.report()

    if rank == 0:
        per_iter_ms = loop_elapsed / PROFILE_NUM_STEPS * 1000
        per_step_ms = (loop_elapsed / completed_steps * 1000) if completed_steps else float("nan")
        print(
            f"\n[TIME rank=0] world_size={world_size}  "
            f"PROFILE_NUM_STEPS={PROFILE_NUM_STEPS}  "
            f"total_loop={loop_elapsed:.2f}s  "
            f"per_iter={per_iter_ms:.2f}ms  "
            f"completed_steps={completed_steps}  "
            f"per_completed_step={per_step_ms:.2f}ms"
        )

    if PROFILE_DUMP_SNAPSHOT:
        path = SNAPSHOT_PATH.format(rank=rank)
        torch.cuda.memory._dump_snapshot(path)
        if rank == 0:
            print(f"\n[snapshot] dumped per-rank memory history to {SNAPSHOT_PATH.format(rank='*')}")
            print("[snapshot] open at https://pytorch.org/memory_viz for callstack-level breakdown\n")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
