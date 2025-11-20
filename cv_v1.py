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

# Note: The original code references 'MsDataset', 'DownloadMode', and 'Image' 
# without importing them. Per the request to only remove unused code and 
# not add/fix logic, these unresolved references remain.

# ========== Config Class ==========
class Config:
    """Training configuration parameters"""
    # Training parameters
    batch_size = 512 #
    max_steps = 1000   # Total iteration steps
    learning_rate = 0.001
    momentum = 0.9
    optimizer_type = 'adam'
    training_mode = 'parallel'

    # Acceleration configuration
    P = 7   # Window size
    threshold = 0.00001 # Error threshold
    ema_decay = 0.999 # Threshold exponential moving average decay rate
    adaptivity_type = 'median'  # Adaptive strategy: 'mean' or 'median'

    
    # System configuration
    seed = 42   # Random seed
    device_count = torch.cuda.device_count()   # Number of GPUs

    # Dataset and model configuration
    dataset = 'cifar10' # Default dataset: 'cifar10' or 'tiny_imagenet'
    model_name = 'cnn'   # Default using cnn
    num_classes = 10 # Default number of classes
    img_size = 32 # Default image size
    pretrained = False

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


# ========== CSV Logger ==========
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
        # Add new fields if they don't exist
        for key in data.keys():
            if key not in self.fieldnames:
                self.fieldnames.add(key)
        
        # Store the data
        self.rows.append(data)
        
    def save(self):
        # Write all data to CSV file
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(self.fieldnames))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
                
    def finish(self):
        self.save()


# ========== Helper Functions ==========
def to_device(batch, device):
    images, labels = batch
    return images.to(device), labels.to(device)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# ========== Model Definitions ==========
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-head Self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP module"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer Model"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

class CNN(nn.Module):
    """A simple Convolutional Neural Network"""
    def __init__(self, num_classes=10, img_size=32):
        super(CNN, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        feature_map_size = img_size // 4
        flattened_size = 64 * feature_map_size * feature_map_size
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x      


class ModelWrapper(nn.Module):
    """Model wrapper class providing a unified interface"""
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
            error, total_num = 0.0, 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p - q).pow(2).item()
                total_num += np.prod(list(q.shape))
            return error / total_num * 1e6




class ModelFactory:
    """Model factory for creating specified models"""
    @staticmethod
    def create_model(config):
        model_name = config.model_name.lower()
        num_classes = config.num_classes
        img_size = config.img_size # This will be 64
        pretrained = config.pretrained

        if model_name == 'cnn':
            model = CNN(num_classes=num_classes, img_size=img_size)
        elif model_name == 'vit':
            patch_size = 4 if img_size == 32 else 8
            model = ViT(num_classes=num_classes, img_size=img_size, patch_size=patch_size)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return ModelWrapper(model)

# ========== ModelScope PyTorch Dataset Wrapper ==========
class ModelScopeDataset(Dataset):
    def __init__(self, ms_dataset, transform=None):
        """
        Args:
            ms_dataset (MsDataset): Dataset instance loaded from ModelScope.
            transform (callable, optional): Transform to apply to samples.
        """
        # Convert iterator to list for indexed access
        self.data = list(ms_dataset)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get image file path using 'image:FILE' key
        image_path = item['image:FILE']
        
        # Get label using 'category' key
        label = item['category']

        # Load image from file path
        # 'Image' is not defined/imported in the original script
        image = Image.open(image_path) 

        # If image is grayscale, convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(config):
    """Load and return corresponding data loaders based on config"""
    if config.dataset == 'cifar10':
        print("Loading CIFAR-10 dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif config.dataset == 'tiny_imagenet':
        print("Loading Tiny ImageNet dataset...")
        data_dir = '../.cache/huggingface/datasets/tiny-imagenet-200'
        val_dir = '../.cache/huggingface/datasets/tiny-imagenet-200/val_structured'
        
        if not os.path.exists(val_dir):
            raise FileNotFoundError("Tiny ImageNet validation set not prepared. Run `prepare_val_folder.py` first.")
            
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        test_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
    


    # ##################################################################
    # ### Integration end ###
    # ##################################################################
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=32, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    
    return train_loader, test_loader


# ========== Core Training Logic ==========
def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())


def take_step(model, optimizer, criterion, dataloader, data_iter, device, step, seed_offset):
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


def run_worker(model, optimizer, criterion, dataloader, queues, device, seed_offset):
    data_iter = iter(dataloader)
    while True:
        ret = queues[0].get()
        if ret is None:
            return
            
        params, step = ret
        model.set_params(params, clone=False)
        
        res = take_step(model, optimizer, criterion, dataloader, data_iter, device, step, seed_offset)
        data_iter = res['data_iter']
        
        grads = [p.grad for p in model.get_params()]
        
        queues[1].put((grads, step, {
            'loss': res['loss'], 
            'accuracy': res['accuracy'], 
            'batch_size': res['batch_size'], 
            'correct': res['correct']
        }))


def train_loop_parallel(config, model, criterion, train_loader, test_loader, csv_logger):
    if config.device_count <= 1:
        print("Only one GPU found, falling back to serial training.")
        return train_loop_serial(config, model, criterion, train_loader, test_loader, csv_logger)

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
            
        p = mp.Process(target=run_worker, args=(worker_model, worker_optimizer, criterion, train_loader, queues, worker_device, config.seed))
        p.start()
        processes.append(p)

    device = torch.device("cuda:0")
    model = model.to(device)
    
    T = config.max_steps
    P = min(config.P, T)
    thresh = config.threshold
    
    models = [None for _ in range(T + 1)]
    optimizers = [None for _ in range(T + 1)]
    
    begin_idx, end_idx = 0, P
    total_iters, running_loss, running_correct, running_total, total_Grad = 0, 0.0, 0, 0, 0

    for step in range(P + 1):
        models[step] = model.clone(device)
        if config.optimizer_type.lower() == 'sgd':
            optimizers[step] = optim.SGD(models[step].parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer_type.lower() == 'adam':
            optimizers[step] = optim.Adam(models[step].parameters(), lr=config.learning_rate)
        else:
            optimizers[step] = optim.AdamW(models[step].parameters(), lr=config.learning_rate)
            
    pbar = tqdm(total=T)
    total_comm_time = 0.0
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()

    while begin_idx < T:
        # --- Time start ---
        start_time_per_iter = time.time()
        parallel_len = end_idx - begin_idx
        pred_f = [None] * parallel_len
        metrics = [None] * parallel_len

        # --- 1. Distribute tasks to worker processes ---
        comm_start_time = time.time()
        for i in range(parallel_len):
            step = begin_idx + i
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))

        # --- 2. Collect results from worker processes ---
        for _ in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _grads
            metrics[_i] = _metrics
        # --- Record communication time end ---
        time_comm_per_iter = time.time() - comm_start_time
        total_comm_time += time_comm_per_iter


        # --- 3. Perform fixed-point iteration update ---
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
                running_loss += metrics[i]['loss']
                running_correct += metrics[i]['correct']
                running_total += metrics[i]['batch_size']
            
            if ind is not None:
                models[step + 1] = rollout_model.clone(device, models[step + 1])
            
            total_Grad += 1

        # --- 4. Update error threshold ---
        thresh = thresh * config.ema_decay + errors_all * (1 - config.ema_decay)
        
        # --- 5. Slide window ---
        progress = ind - begin_idx
        pbar.update(progress)
        # print("total_iters",total_iters,"progress",progress)
        total_iters += 1
        
        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)

        for step in range(end_idx + 1, new_end_idx + 1):
            prev_model_idx = step - 1 - parallel_len
            models[step] = rollout_model.clone(device, models[prev_model_idx] if prev_model_idx >= 0 else None)
            optimizers[step] = optimizers[prev_model_idx] if prev_model_idx >= 0 else None

        begin_idx = new_begin_idx
        end_idx = new_end_idx
        time_per_iter = time.time() - start_time_per_iter
        time_comp_per_iter = time_per_iter - time_comm_per_iter
        # --- 6. Log progress ---
        # if total_iters % 5 == 0 and running_total > 0:
        accuracy = 100 * running_correct / running_total if running_total > 0 else 0
        avg_loss = running_loss / begin_idx if begin_idx > 0 else 0
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        pbar.set_description(f'Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {elapsed_str}')
        
        # Log to CSV
        csv_logger.log({
            "Iter": total_iters,
            "Train_Loss": avg_loss,
            "Train_Acc": accuracy,
            "Original_Steps": begin_idx,
            "time_per_iter": time_per_iter,
            "time_comm_per_iter": time_comm_per_iter,
            "time_comp_per_iter": time_comp_per_iter
        })


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
    final_accuracy, metrics = evaluate_model(final_model, test_loader, device)
    
    
    
    upper_bound = total_iters * P
    m = total_Grad / T
    
    # Log final metrics
    csv_logger.log({"final_test_accuracy": final_accuracy,
                    "final_test_precision":metrics["precision_macro"],
                    "final_test_recall":metrics["recall_macro"],
                    "final_test_f1-score":metrics["f1_macro"],
                    "time":elapsed,
                    "time_comm":total_comm_time,
                    "time_step":elapsed - total_comm_time,
                    "K":total_iters,
                    "G":total_Grad,
                    "upper_bound":upper_bound,
                    "m":m
                    })
    
    comm_time_str = str(timedelta(seconds=int(total_comm_time)))
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Total communication time: {comm_time_str}") # Print communication time
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Final test precision: {metrics['precision_macro']:.2f}%")
    print(f"Final test recall: {metrics['recall_macro']:.2f}%")
    print(f"Final test f1-score: {metrics['f1_macro']:.2f}%")      
    print(f"Total iterations: {total_iters} (vs {T} normal iterations)")
    print(f"Effective speed-up: {T/total_iters:.2f}x")
    
    return final_model


def train_loop_serial(config, model, criterion, train_loader, test_loader, csv_logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("config.optimizer_type",config.optimizer_type,config.learning_rate,device)
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
    start_time = time.time()
    running_loss, running_correct, running_total = 0.0, 0, 0
    pbar = tqdm(total=config.max_steps)
    data_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        
        running_loss += loss.item()
        running_correct += correct
        running_total += labels.size(0)
        
        # if (step + 1) % 5 == 0:
        avg_loss = running_loss / (step + 1)
        avg_acc = 100 * running_correct / running_total
        elapsed_str = str(timedelta(seconds=int(time.time() - start_time)))
        
        # Log to CSV
        csv_logger.log({
            "Iter": step + 1,
            "Train_Loss": avg_loss,
            "Train_Acc": avg_acc,
            "Original_Steps": step + 1
        })
        
        pbar.set_description(f'Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | Time: {elapsed_str}')
            
        pbar.update(1)
        
    pbar.close()
    
    elapsed = time.time() - start_time
    final_accuracy, metrics = evaluate_model(model, test_loader, device)
    
    # Log final metrics
    csv_logger.log({
        "final_test_accuracy": final_accuracy, 
        "final_test_precision": metrics["precision_macro"], 
        "final_test_recall": metrics["recall_macro"], 
        "final_test_f1-score": metrics["f1_macro"], 
        "time": elapsed
    })
    
    print(f"\nTraining completed in {str(timedelta(seconds=int(elapsed)))}")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    
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
    parser = argparse.ArgumentParser(description='ParaOpt Training with Multiple Datasets')

    parser.add_argument('--device_count', type=int, help='Number of CUDA devices to use')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--max_steps', type=int, help='max steps')
    parser.add_argument('--learning_rate', type=float, help='learning rate')

    parser.add_argument('--P', type=int, help='window size')
    parser.add_argument('--threshold', type=float, help='threshold')
    parser.add_argument('--ema_decay', type=float, help='EMA decay for threshold')
    parser.add_argument('--adaptivity_type', type=str, choices=['mean', 'median'], help='error computation type')
    
    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adam', 'adamw'], help='optimizer type')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial'], help='training mode')
    
    parser.add_argument('--model_name', type=str, 
                        choices=['cnn', 'vit'], 
                        help='model to use')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')

    parser.add_argument('--dataset', type=str, choices=['cifar10', 'tiny_imagenet'], help='Dataset to use')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs')

    return parser


def train_with_config(config=None, csv_logger=None):
    if config is None:
        config = Config()

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Initialize CSV logger if not provided
    if csv_logger is None:
        csv_logger = CSVLogger()
    
    if config.dataset == 'tiny_imagenet':
        config.num_classes = 200
        config.img_size = 64
    elif config.dataset == 'cifar10':
        config.num_classes = 10
        config.img_size = 32


    # Load data
    train_loader, test_loader = get_dataloaders(config)
    
    # Create model and loss function
    model = ModelFactory.create_model(config)
    criterion = nn.CrossEntropyLoss()
    
    # Start training
    if config.training_mode.lower() == "parallel":
        train_loop_parallel(config, model, criterion, train_loader, test_loader, csv_logger)
    else:
        train_loop_serial(config, model, criterion, train_loader, test_loader, csv_logger)
    
    csv_logger.finish()

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Regular single run, using command-line arguments
    config = Config()
    config.update_from_args(args)
    
    # Build directory path (excluding filename part)
    directory_path = os.path.join(
    "./new_exp_results", config.dataset, config.model_name,
        f"steps_{config.max_steps}_bs_{config.batch_size}_lr_{config.learning_rate}_optim_{config.optimizer_type}"
    )
    
    output_csv = directory_path + f"/{config.training_mode}_{config.adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.ema_decay}.csv"
    print(output_csv)


    csv_logger = CSVLogger(output_csv) if output_csv else CSVLogger()
    
    train_with_config(config, csv_logger)


if __name__ == "__main__":
    main()