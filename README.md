# PASO: Step-Parallel Training

**PASO** is the first step-parallel training method based on equation system solving. By reformulating the optimization process as a fixed-point problem, PASO breaks the sequential dependency of standard optimizers, allowing for **significant acceleration in training** without sacrificing model accuracy.

-----

##  🖥️ Our Experiments Environments
All experiments were conducted on a single high-performance computing node equipped with 8 NVIDIA A100 GPUs (80GB). The GPUs are interconnected via high-bandwidth NVLink and NVSwitch (fully connected topology). The software environment is standardized on Python 3.11, PyTorch 2.5.1, CUDA 12.1, and NCCL 2.21.5 running on Ubuntu 22.04.

## ⚡ Quick Start

### 1\. Installation

```bash
pip install -r rq.txt
```

### 2\. Basic Verification (OptEx vs PASO)
This script makes a comparison between OptEx and our PASO. For OptEx, we directly use their source code.

```bash
python  optex_cmp.py
```

-----

## 🚀 Experiments

### 🖼️ Computer Vision (CIFAR-10)

#### A. High-Performance Version (`cv_v2.py`)

> **Note:** This version achieves high accuracy, comparable to community results from the reviewr provided link [vision-transformers-cifar10](https://www.google.com/search?q=//github.com/kentaroy47/vision-transformers-cifar10).

**Sequential Training (Baseline):**

```bash
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --training_mode serial --optimizer_type adam
```

**PASO Training:**

```bash
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.999 --training_mode parallel --P 7 --optimizer_type adam
```

#### B. Memory-Efficient Distributed Version

> **Note:** In this version, each GPU only needs to store one model and optimizer state.

```bash
torchrun --nproc_per_node=8 --master_port=12345 cv_nccl.py
```

#### C. Legacy Version (`cv_v1.py`)

> **Note:** Older version with lower accuracy.

**Sequential Training:**

```bash
python cv_v1.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --training_mode serial --optimizer_type adam
```

**PASO Training:**

```bash
python cv_v1.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.999 --training_mode parallel --P 7 --optimizer_type adam
```

-----

### 📝 LLM Tasks (WikiText-2)

#### A. High-Performance Version (`llm_v2.py`)

> **Note:** This version achieves perplexity comparable to community results from the reivewer provided link [gpt2-finetuned-wikitext103](https://huggingface.co/neulab/gpt2-finetuned-wikitext103).

**Sequential Training:**

```bash
python llm_v2.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 30 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.999 --training_mode parallel --P 7
```

**PASO Training:**

```bash
python llm_v2.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 112 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.9 --training_mode serial --P 7
```
**Naive Data-Parallel Training:**

```bash
python llm_v2.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 112 --learning_rate 6e-5  --training_mode navie_dp 
```

**Pytorch's Data-Parallel Training:**

```bash
python llm_v2.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 112 --learning_rate 6e-5  --training_mode ddp 
```

#### B. Legacy Version (`llm_v1.py`)

> **Note:** Older version with lower perplexity.

**Sequential Training:**

```bash
python llm_v1.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 30 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.999 --training_mode serial --P 7
```

**PASO Training:**

```bash
python llm_v1.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 30 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.999 --training_mode parallel --P 7
```

-----

## 📊 Visualization

Use the plotting script to visualize and compare training metrics (e.g., Loss, Accuracy).

```bash
python plot.py --csv_path1 "Path to the first CSV file (will be labeled 'AdamW')." --csv_path2 "Path to the second CSV file (will be labeled 'ParaAdamW')." --column_name "Name of the column to plot (e.g., Train_Loss, Train_Acc, final_test_accuracy, etc.)."
```