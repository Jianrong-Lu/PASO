# -*- coding: utf-8 -*-
"""
Serial baseline memory profile.

Purpose: measure GPU memory of plain single-GPU AdamW training on the same
model/batch/dtype as cv_nccl_profile.py. Diffing the two reports tells you
exactly what PASO+NCCL adds on top of "just train it normally."

Run:
  CUDA_VISIBLE_DEVICES=0 python serial_profile.py 2>&1 | tee mem_serial.log

Then diff against cv_nccl_profile.py's MEMORY REPORT.

Stages are intentionally named to mirror the NCCL profile where possible:
  baseline                              -> baseline (post init_process_group)
  (A) model on GPU                      -> (A) model on GPU
  AdamW constructed (states lazy ~0)    -> AdamW constructed (states lazy ~0)
  after first forward+backward          -> after first forward+backward (pre-NCCL)
  after first optimizer.step()          -> [NCCL has no analog here; this is where
                                           AdamW's REAL state allocates lazily]
  steady-state after N steps            -> after multi_step #N (steady)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


PROFILE_NUM_STEPS = 5
PROFILE_DUMP_SNAPSHOT = True
SNAPSHOT_PATH = "./mem_snapshot_serial.pickle"


class MemTracker:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = []
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
        if self.verbose:
            print(
                f"[MEM serial] {tag:<55} "
                f"alloc={row['allocated_MB']:>9.2f}MB  "
                f"Δ={row['delta_MB']:>+9.2f}MB  "
                f"cum={row['cumulative_MB']:>+9.2f}MB  "
                f"peak={row['peak_MB']:>9.2f}MB"
            )

    def report(self):
        print("\n" + "=" * 110)
        print("MEMORY REPORT (single-GPU serial baseline)")
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


# ---- CNN, Config, set_seed: identical to cv_nccl_profile.py / cv_nccl.py ----
# Same architecture so direct subtraction makes sense.

class Config:
    max_steps: int = 60000
    learning_rate: float = 0.00085
    batch_size: int = 4096
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

    def forward(self, x):
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


def main():
    DATA_ROOT = './data'
    config = Config()
    set_seed(config.seed)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    tracker = MemTracker()

    if PROFILE_DUMP_SNAPSHOT:
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    print(f"\n{'=' * 110}\nSERIAL BASELINE PROFILE  device={device}\n{'=' * 110}\n")
    tracker.set_baseline()
    tracker.checkpoint("baseline (post init_process_group)")

    # --- data (CPU; no GPU footprint) ---
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    raw_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    tracker.checkpoint("data loaded (CPU; should be ~0)")

    # --- (A) model on GPU --- matches NCCL stage (A)
    model = CNN().to(device)
    tracker.checkpoint("(A) model on GPU")

    # --- AdamW constructed --- matches NCCL "AdamW constructed"
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    tracker.checkpoint("AdamW constructed (states lazy, expect ~0)")

    # --- training loop ---
    criterion = nn.CrossEntropyLoss()
    data_iter = iter(raw_loader)

    for step in range(PROFILE_NUM_STEPS):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(raw_loader)
            inputs, labels = next(data_iter)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if step == 0:
            tracker.checkpoint("after first forward+backward (pre-step)")

        optimizer.step()

        if step == 0:
            # AdamW lazily allocates exp_avg + exp_avg_sq on first step().
            # This is the REAL optimizer-state footprint to compare against
            # NCCL's (D) 4x pre-allocated states.
            tracker.checkpoint("after first optimizer.step() (Adam state alloc'd)")
        elif step == PROFILE_NUM_STEPS - 1:
            tracker.checkpoint(f"after step #{step + 1} (steady)")

    tracker.checkpoint("end")
    tracker.report()

    if PROFILE_DUMP_SNAPSHOT:
        torch.cuda.memory._dump_snapshot(SNAPSHOT_PATH)
        print(f"\n[snapshot] dumped to {SNAPSHOT_PATH}")
        print("[snapshot] open at https://pytorch.org/memory_viz\n")


if __name__ == "__main__":
    main()
