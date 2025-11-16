#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于大型语言模型的高级并行训练脚本。

实现了基于固定点迭代的自定义并行算法 (ParaSolver/PASO)，
并支持使用 ModelScope 和 Hugging Face Transformers 进行模型加载和训练。
"""

# ==========================
# 1. Imports
# ==========================
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, disable_caching
from modelscope import snapshot_download
from modelscope import AutoTokenizer as ModelScopeAutoTokenizer # 显式命名以避免冲突
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

# from torchmetrics.functional import accuracy, f1_score # 保持注释，因为原始代码未激活

# ==========================
# 2. CSV Logger Class
# ==========================
class CSVLogger:
    """一个简单的CSV日志记录器，用于在训练期间保存指标。"""
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
        """记录一行新数据（一个字典）。"""
        # 动态添加新的字段名
        for key in data.keys():
            if key not in self.fieldnames:
                self.fieldnames.add(key)
        
        self.rows.append(data)
        
    def save(self):
        """将所有缓冲的数据写入CSV文件。"""
        if not self.rows:
            return
            
        with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(list(self.fieldnames)))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
                
    def finish(self):
        """保存并关闭日志。"""
        self.save()

# ==========================
# 3. Configuration Class
# ==========================
class Config:
    """训练配置参数。"""
    # 训练参数
    batch_size = 8
    num_epochs = 1
    max_steps = 10
    learning_rate = 6e-5
    momentum = 0.9

    # 加速配置 (ParaSolver/PASO)
    P = 7  # 窗口大小
    threshold = 1e-5  # 误差阈值
    ema_decay = 0.9  # 阈值的EMA衰减
    adaptivity_type = 'mean'  # 'mean' 或 'median'
    
    # 系统配置
    seed = 42
    device_count = torch.cuda.device_count()
    
    # 模型和数据集配置
    # 注意：这些最好通过命令行参数覆盖
    modelscope_name = 'LLM-Research/Llama-3.2-1B'
    local_model_dir = "/gruntdata/heyuan67/lu_25/model/Llama-3.2-1B"
    dataset_name = 'local_wikitext'
    local_dataset_dir = "/gruntdata/heyuan67/lu_25/dataset/wikitext/wikitext-2" 
    output_dir = "/gruntdata/heyuan67/lu_25/PASO/new_exp_results"
    
    max_length = 512  # Tokenizer 最大序列长度
    optimizer_type = 'adamw'
    training_mode = 'parallel'
    sweep_config_path = 'XXX.json' # 似乎未使用

    def update_from_args(self, args):
        """从命令行的 argparse 命名空间更新配置。"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

# ==========================
# 4. Utility & Metric Functions
# ==========================
def to_device(batch, device):
    """将一个数据批次（字典）移动到指定设备。"""
    return {k: v.to(device) for k, v in batch.items()}

def calculate_metrics(preds, labels, ignore_index=-100):
    """
    计算准确率和F1分数 (基于Sklearn的CPU版本)。
    注意：这是您脚本中三个定义中的最后一个，也是唯一一个被激活的。
    """
    # 展平预测和标签
    preds_flat = preds.cpu().flatten()
    labels_flat = labels.cpu().flatten()
    
    # 创建一个mask来忽略填充标记
    mask = (labels_flat != ignore_index)
    
    # 应用mask
    active_preds = preds_flat[mask]
    active_labels = labels_flat[mask]
    
    if len(active_labels) == 0:
        # 如果没有有效的标签
        return 0.0, 0.0
        
    # 计算准确率
    accuracy = accuracy_score(active_labels, active_preds)
    
    # 计算F1分数(宏平均)
    f1 = f1_score(active_labels, active_preds, average='macro', zero_division=0)
    
    return accuracy, f1

def optimizer_state_clone(optimizer_from, optimizer_to):
    """将一个优化器的状态复制到另一个。"""
    optimizer_to.load_state_dict(optimizer_from.state_dict())

# ==========================
# 5. Data Handling
# ==========================
def load_wikitext_from_local(path):
    """从本地目录加载 wikitext 数据集。"""
    if not os.path.exists(path):
        raise ValueError(f"Dataset path {path} does not exist. Please check `local_dataset_dir`.")
    
    data_files = {
        "train": os.path.join(path, "wiki.train.tokens"),
        "validation": os.path.join(path, "wiki.valid.tokens"),
        "test": os.path.join(path, "wiki.test.tokens")
    }
    
    # 过滤掉不存在的文件
    existing_files = {split: file for split, file in data_files.items() if os.path.exists(file)}
    if not existing_files:
        raise FileNotFoundError(f"No valid data files found in {path}. Expected wiki.train.tokens, etc.")
        
    print(f"Loading dataset from files: {existing_files}")
    dataset = load_dataset("text", data_files=existing_files)
    return dataset

def _tokenize_function(examples, tokenizer, max_length):
    """用于数据集 map 的辅助 Tokenize 函数。"""
    # 使用相同的文本作为输入和标签
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 添加labels字段 (Causal LM 的标准做法)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    return tokenized_inputs

def get_dataloaders(config):
    """加载和准备指定的数据集并返回 DataLoaders。"""
    # 确定用于 tokenizer 的模型目录
    if config.local_model_dir and os.path.exists(config.local_model_dir):
        model_dir = config.local_model_dir
    else:
        print(f"Downloading tokenizer for '{config.modelscope_name}'...")
        model_dir = snapshot_download(
            config.modelscope_name, 
            revision='master',
            cache_dir=os.path.dirname(config.local_model_dir) # 使用 config 中 local_dir 的父目录作为缓存
        )
        
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token not set. Setting to eos_token.")

    print(f"Loading dataset: {config.dataset_name}...")
    disable_caching() # 推荐用于 multiprocessing

    if config.dataset_name == 'local_wikitext':
        dataset = load_wikitext_from_local(config.local_dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}. Please use 'local_wikitext'.")

    print("Tokenizing dataset...")
    column_names = dataset["train"].column_names
    
    # 使用 functools.partial 来传递额外参数
    from functools import partial
    preprocess_fn = partial(_tokenize_function, tokenizer=tokenizer, max_length=config.max_length)
    
    tokenized_datasets = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
        num_proc=16 # 可以适当增加进程数以加快预处理
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
    """模型的通用包装器，以提供一致的接口。"""
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        # 为数据并行禁用缓存
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def clone(self, device, other=None):
        """在指定设备上创建模型的深层副本。"""
        if other is None:
            # 创建一个新实例
            new_model_instance = copy.deepcopy(self.model)
            other = ModelWrapper(new_model_instance).to(device)
        else:
            # 如果提供了目标模型，则复制参数状态
            with torch.no_grad():
                for param_to, param_from in zip(other.parameters(), self.parameters()):
                    param_to.data = param_from.data.clone()
        return other

    def get_params(self):
        """返回模型参数列表。"""
        return list(self.parameters())

    def set_params(self, params, clone=True):
        """从给定的参数列表设置模型参数。"""
        my_params = self.get_params()
        for p, q in zip(my_params, params):
            if clone:
                p.data = q.data.clone().to(p.device)
            else:
                p.data = q.to(p.device)

    def set_grads_from_grads(self, grads):
        """设置模型的梯度。"""
        my_params = self.get_params()
        for p, grad in zip(my_params, grads):
            if grad is not None:
                p.grad = grad.to(p.device)

    def compute_error_from_model(self, other):
        """计算此模型与另一个模型参数之间的均方误差。"""
        my_params = self.get_params()
        other_params = other.get_params()
        with torch.no_grad():
            error, total_num = 0.0, 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p - q).pow(2).item()
                total_num += np.prod(list(q.shape))
            return error / total_num * 1e6

class ModelFactory:
    """使用 ModelScope 创建指定LLM模型的工厂。"""
    @staticmethod
    def create_model(config, device_map=None):
        """
        创建模型，优先使用本地目录。
        device_map: 关键参数。对于并行训练，应为 None。
        """
        if config.local_model_dir and os.path.exists(config.local_model_dir):
            model_dir = config.local_model_dir
            print(f"Loading model from local path: {model_dir}")
        else:
            print(f"Downloading model '{config.modelscope_name}' from ModelScope...")
            model_dir = snapshot_download(
                config.modelscope_name, 
                revision='master',
                cache_dir=os.path.dirname(config.local_model_dir) # 使用 config 中 local_dir 的父目录作为缓存
            )
            print(f"Model downloaded to: {model_dir}")
        
        # 加载模型
        # **重要修复**：
        # 将 device_map="auto" 更改为 device_map=None（或由调用者指定）。
        # `device_map="auto"` 会与 `train_loop_parallel` 中的手动设备放置冲突。
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto", # 使用 "auto" 或 torch.bfloat16
            trust_remote_code=True,
            device_map=device_map # 允许调用者控制
        )

        return ModelWrapper(model)

# ==========================
# 7. Core Training Logic
# ==========================

def take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset):
    """执行单个训练步骤（前向和后向传播）。"""
    # 为此特定步骤设置可复现的种子
    np.random.seed(step + seed_offset)
    torch.manual_seed(step + seed_offset)
    torch.cuda.manual_seed(step + seed_offset)
    
    try:
        batch = next(data_iter)
    except StopIteration:
        # print(f"Data iterator reset at step {step}") # 用于调试
        data_iter = iter(dataloader) 
        batch = next(data_iter)
        
    batch = to_device(batch, device)
    
    # 确保有labels字段
    if 'labels' not in batch:
        batch['labels'] = batch['input_ids'].clone()   

    optimizer.zero_grad()
    outputs = model(**batch)
    
    if outputs.loss is None:
        raise ValueError("Model did not return loss value. Check input and model configuration.")
    
    loss = outputs.loss
    loss.backward()
    
    # 计算指标
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
    """在测试集上评估模型性能。"""
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
            
            # 计算准确率和F1分数
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
    
    model.train() # 将模型设置回训练模式
    return avg_loss, avg_perplexity, avg_accuracy, avg_f1

# ==========================
# 8. Training Orchestration
# ==========================

def run_worker(model, optimizer, dataloader, queues, device, seed_offset):
    """每个并行工作进程执行的函数。"""
    data_iter = iter(dataloader)
    while True:
        ret = queues[0].get()
        if ret is None: # 终止信号
            return
            
        params, step = ret
        model.set_params(params, clone=False)
        
        res = take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset)
        data_iter = res['data_iter']
        
        grads = [p.grad.clone() for p in model.get_params()] # 克隆梯度以安全发送
        
        queues[1].put((grads, step, {
            'loss': res['loss'], 
            'perplexity': res['perplexity'],
            'accuracy': res['accuracy'],
            'f1_score': res['f1_score'],
            'batch_size': res['batch_size']
        }))

def _create_optimizer(model, config):
    """根据配置创建优化器。"""
    if config.optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    else: # 默认 'adamw'
        return optim.AdamW(model.parameters(), lr=config.learning_rate)

def train_loop_parallel(config, model, train_loader, test_loader, csv_logger):
    """使用固定点迭代算法的主并行训练循环。"""
    if config.device_count <= 1:
        print("Only one GPU found, falling back to serial training.")
        return train_loop_serial(config, model, train_loader, test_loader, csv_logger)

    mp.set_start_method('spawn', force=True)
    queues = (mp.Queue(), mp.Queue())
    processes = []

    # 为工作进程创建模型和优化器
    for rank in range(1, config.device_count):
        worker_device = torch.device(f"cuda:{rank}")
        # **重要**：
        # 工作进程在自己的设备上创建自己的模型实例
        # 这里 device_map=None 是正确的
        worker_model = ModelFactory.create_model(config, device_map=None).to(worker_device)
        worker_optimizer = _create_optimizer(worker_model, config)
        
        p = mp.Process(target=run_worker, args=(worker_model, worker_optimizer, train_loader, queues, worker_device, config.seed))
        p.start()
        processes.append(p)

    # 主进程设置
    device = torch.device("cuda:0")
    model = model.to(device)
    
    T = config.max_steps
    P = min(config.P, T, config.device_count - 1) # 窗口大小不能超过工作进程数
    if P != config.P:
        print(f"Warning: Window size P reduced from {config.P} to {P} to match worker count.")
        
    thresh = config.threshold
    
    models = [None] * (T + 1)
    optimizers = [None] * (T + 1)
    
    begin_idx, end_idx = 0, P
    total_iters, total_Grad = 0, 0

    running_loss, running_perplexity, running_accuracy, running_f1, running_total = 0.0, 0.0, 0.0, 0.0, 0

    # 初始化模型和优化器状态
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

        # 1. 分发任务给工作进程
        comm_start_time = time.time()
        for i in range(parallel_len):
            step = begin_idx + i
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))

        # 2. 从工作进程收集结果
        for _ in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _grads
            metrics[_i] = _metrics
        time_comm_per_iter = time.time() - comm_start_time
        total_comm_time += time_comm_per_iter

        # 3. 执行固定点迭代更新
        rollout_model = models[begin_idx]
        rollout_optimizer = optimizers[begin_idx]
        ind, errors_all = None, 0

        for i in range(parallel_len):
            step = begin_idx + i
            rollout_model.set_grads_from_grads(pred_f[i])
            rollout_optimizer.step()
            rollout_optimizer.zero_grad()
            
            error = rollout_model.compute_error_from_model(models[step + 1])
            
            # 累积误差用于自适应阈值
            if config.adaptivity_type == 'median' and i == parallel_len // 2:
                errors_all = error
            elif config.adaptivity_type == 'mean':
                errors_all += error / parallel_len

            # 检查收敛
            if ind is None and (error > thresh or i == parallel_len - 1):
                ind = step + 1
                optimizer_state_clone(rollout_optimizer, optimizers[step + 1])

            # 累积指标 (仅限已验证的步骤)
            if ind is None or step < ind:
                _metrics = metrics[i]
                running_loss += _metrics['loss']
                running_perplexity += _metrics['perplexity']
                running_accuracy += _metrics['accuracy']
                running_f1 += _metrics['f1_score']
                running_total += _metrics['batch_size']
            
            if ind is not None:
                # 一旦找到不匹配点，用 'rollout' 的真实状态更新所有后续模型
                models[step + 1] = rollout_model.clone(device, models[step + 1])
            
            total_Grad += 1

        # 4. 更新误差阈值 (EMA)
        thresh = thresh * config.ema_decay + errors_all * (1 - config.ema_decay)
        
        # 5. 滑动窗口
        progress = ind - begin_idx
        pbar.update(progress)
        total_iters += 1
        
        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)

        # 准备下一个窗口的模型
        for step in range(end_idx + 1, new_end_idx + 1):
            prev_model_idx = step - parallel_len - 1
            models[step] = rollout_model.clone(device, models[prev_model_idx] if prev_model_idx >= 0 else None)
            optimizers[step] = optimizers[prev_model_idx] if prev_model_idx >= 0 else None

        begin_idx = new_begin_idx
        end_idx = new_end_idx
        time_per_iter = time.time() - start_time_per_iter
        time_comp_per_iter = time_per_iter - time_comm_per_iter

        # 6. 记录进度到 CSV 和 Pbar
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
            csv_logger.save() # 频繁保存以防崩溃

    end_event.record()
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event) / 1000.0
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    pbar.close()

    # 清理工作进程
    for _ in processes:
        queues[0].put(None)
    for p in processes:
        p.join()
        
    final_model = models[T]
    del models, optimizers
    
    # 最终评估
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
    """标准的串行训练循环。"""
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
        
        # 梯度已在 take_step 中计算，这里应用
        optimizer.step()
        
        # 更新统计数据
        running_loss += res['loss']
        running_perplexity += res['perplexity']
        running_accuracy += res['accuracy']
        running_f1 += res['f1_score']
        running_total += res['batch_size']
        
        # 更新进度条
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
            
            # 记录指标
            csv_logger.log({
                "Iter": step + 1,
                "Train_Loss": avg_loss,
                "Train_Acc": avg_accuracy,
                "Train_Perplexity": avg_perplexity,
                "Train_F1": avg_f1,
                "Original_Steps": step + 1
            })
            csv_logger.save() # 频繁保存
        
        pbar.update(1)
    
    pbar.close()
    
    elapsed = time.time() - start_time
        
    # 最终测试
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
    """设置命令行参数解析器。"""
    parser = argparse.ArgumentParser(description='ParaOpt Training for Language Models (ModelScope Version)')

    # 训练超参数
    parser.add_argument('--device_count', type=int, help='Number of CUDA devices')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_steps', type=int, help='Total training steps')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')

    # 并行超参数
    parser.add_argument('--P', type=int, help='Window size for parallel algorithm')
    parser.add_argument('--threshold', type=float, help='Error threshold')
    parser.add_argument('--ema_decay', type=float, help='EMA decay for threshold')
    parser.add_argument('--adaptivity_type', type=str, choices=['mean', 'median'], help='Error computation type')
    
    # 通用配置
    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adam', 'adamw'], help='Optimizer type')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial'], help='Training mode')
    parser.add_argument('--modelscope_name', type=str, help='Model name from ModelScope (e.g., qwen/qwen-1_8b)')
    parser.add_argument('--local_model_dir', type=str, help='Path to local model directory (overrides Config)')
    parser.add_argument('--dataset_name', type=str, choices=['local_wikitext'], help='Dataset to use')
    parser.add_argument('--local_dataset_dir', type=str, help='Path to local dataset directory (overrides Config)')
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for tokenizer')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, help='Directory to save logs (overrides Config)')
    parser.add_argument('--output_csv', type=str, help='Exact output CSV file path (overrides auto-naming)')

    return parser

def train_with_config(config=None, csv_logger=None):
    """使用给定的配置运行一个完整的训练过程。"""
    if config is None:
        config = Config()

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 初始化 CSV logger
    if csv_logger is None:
        csv_logger = CSVLogger()
    
    print(f"--- Starting Training Run ---")
    print(f"Mode: {config.training_mode} | Model: {config.modelscope_name}")
    print(f"Steps: {config.max_steps} | Batch Size: {config.batch_size} | LR: {config.learning_rate}")
    if config.training_mode == 'parallel':
        print(f"Parallel Config: P={config.P}, Threshold={config.threshold}, Adaptivity={config.adaptivity_type}")
    print(f"Log file: {csv_logger.filename}")
    
    # 加载数据
    train_loader, test_loader = get_dataloaders(config)
    
    # 创建模型
    # **重要**：
    # 在主进程中，我们不使用 device_map="auto"。
    # `train_loop_serial` 和 `train_loop_parallel` 将负责将其移动到 cuda:0。
    model = ModelFactory.create_model(config, device_map=None)
    
    # 开始训练
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
    
    # 确定输出CSV文件的路径
    output_csv = args.output_csv
    
    if output_csv is None:
        # 如果未提供确切路径，则自动构建路径
        directory_path = os.path.join(
            config.output_dir, # 使用 config 中的 (可能被 arg 覆盖的) output_dir
            config.modelscope_name.split('/')[-1], # 使用模型的简称
            f"steps_{config.max_steps}_bs_{config.batch_size}_lr_{config.learning_rate}_optim_{config.optimizer_type}"
        )
        
        # 文件名
        if config.training_mode == 'parallel':
            filename = f"{config.training_mode}_{config.adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.ema_decay}.csv"
        else:
            filename = f"{config.training_mode}.csv"
            
        output_csv = os.path.join(directory_path, filename)
    
    csv_logger = CSVLogger(output_csv)
    
    train_with_config(config, csv_logger)

if __name__ == "__main__":
    main()