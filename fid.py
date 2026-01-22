import torch
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        
        # --- Optimization: use os.listdir instead of recursive glob ---
        self.paths = []
        valid_exts = {f'.{ext.lower()}' for ext in exts}
        
        # If the images are in the top-level directory, this step will be completed instantly
        print("Scanning directory...")
        for file in os.listdir(folder):
            if os.path.splitext(file)[1].lower() in valid_exts:
                self.paths.append(os.path.join(folder, file))
        
        print(f"✅ Successfully discovered {len(self.paths)} images")
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# ==========================================
# 1. Helper Functions
# ==========================================
def cycle(dl):
    """Infinite iterator wrapper for distributed DataLoader"""
    while True:
        for data in dl:
            yield data

def load_ema_weights(diffusion_model, weights_path, device):
    """
    Specifically loads the .pt file saved by the save_ema_model function in the training code.
    This file contains only the state_dict of the Unet.
    """
    if not os.path.exists(weights_path):
        print(f"❌ Error: weight file {weights_path} not found")
        return False
    
    print(f"🔍 Loading EMA weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    
    # In your training code, save_ema_model saves ema.ema_model.model.state_dict()
    # This corresponds to self.model in GaussianDiffusion (i.e., the Unet)
    try:
        diffusion_model.load_state_dict(state_dict)
        print("✅ EMA weights successfully loaded into the Unet within Diffusion")
        return True
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        return False

# ==========================================
# 2. Main Program
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Distributed FID Evaluation")
    
    parser.add_argument("--results_folder", type=str, required=True, help="Folder path where the checkpoints are located")
    parser.add_argument("--data_folder", type=str, required=True, help="Ground truth dataset path")
    parser.add_argument("--milestone", type=int, required=True, help="Milestone number")
    parser.add_argument("--num_samples", type=int, default=50000, help="Total number of FID samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per card")
    parser.add_argument("--image_size", type=int, default=128, help="Image size")
    parser.add_argument("--sampling_timesteps", type=int, default=100, help="Sampling steps")
    parser.add_argument("--method", type=str, default='parallel', help="Evaluation method")
    
    args = parser.parse_args()

    # --- A. Initialize Accelerator ---
    # split_batches=True means total effective batch_size is handled by Accelerator
    accelerator = Accelerator(split_batches = True)
    device = accelerator.device

    # --- B. Instantiate Model Architecture ---
    # Keep strictly consistent with training configuration
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 4, 8, 8),
        flash_attn = False,
        full_attn = (False, False, False, False, True, True)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.image_size,
        timesteps = 1000,
        sampling_timesteps = args.sampling_timesteps,
        objective = 'pred_noise',
        beta_schedule = 'linear',
        auto_normalize = True
    ).to(device)

    # --- C. Prepare Components ---
    
    # Load weights: can be loaded using unwrap_model before or after prepare
    ckpt_name = f'model-{args.milestone}.pt'
    ckpt_path = Path(args.results_folder) / ckpt_name
    
    success = load_ema_weights(diffusion, str(ckpt_path), accelerator.device)
    if not success: return

    # Dataset
    ds = Dataset(args.data_folder, image_size = args.image_size, augment_horizontal_flip = True)
    print("Dataset loaded")
    dl = DataLoader(ds, batch_size = args.batch_size, shuffle = True, num_workers = 32, pin_memory = True)

    # Wrap with Accelerator. It will automatically handle data distribution (DistributedSampler)
    # Note: Here we only prepare the dataloader because sampling logic is manually controlled inside diffusion
    diffusion, dl = accelerator.prepare(diffusion, dl)
    fid_dl = cycle(dl)

    # --- D. Execute FID Calculation ---
    if accelerator.is_main_process:
        print(f"\n🚀 Starting multi-card parallel FID evaluation")
        print(f"Target total samples: {args.num_samples}")
        print(f"Batch Size per card: {args.batch_size}")
        print(f"Sampling steps: {args.sampling_timesteps}")

    # FIDEvaluation automatically recognizes multi-card environments and aggregates Inception Features
    fid_scorer = FIDEvaluation(
        batch_size = args.batch_size,
        dl = fid_dl,
        sampler = accelerator.unwrap_model(diffusion), 
        channels = 3,
        accelerator = accelerator,
        stats_dir = args.results_folder,
        device = device,
        num_fid_samples = args.num_samples,
        inception_block_idx = 2048
    )

    # Begin calculation
    fid_score = fid_scorer.fid_score()
    
    # Output results via main process
    if accelerator.is_main_process:
        print(f"\n" + "="*40)
        print(f"Evaluation complete!")
        print(f"Milestone: {args.milestone}")
        print(f"FID Score: {fid_score:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()