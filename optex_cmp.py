#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for comparing optimization algorithms on synthetic functions.

This script compares three optimization methods:
1.  **Vanilla**: Standard optimizers (Adam).
2.  **OptEx**: A parallel optimization method (requires 'optex' module).
3.  **PASO**: A step-parallel optimization method (implemented in this file).

It is designed to test these methods on high-dimensional synthetic objective
functions like Sphere, Ackley, and Rosenbrock.

Usage:
    python optex_cmp.py 

Requirements:
    - torch
    - numpy
    - matplotlib
    - tqdm
    - optex (local module for OpTex)
"""

# ========================================================
# 1. Standard Library Imports
# ========================================================
import logging
import sys
import os
import time
import copy
import torch.multiprocessing as mp

# ========================================================
# 2. Third-Party Imports
# ========================================================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm



# ========================================================
# 3. Local Module Imports (PASO / OptEx)
# ========================================================

# Assume optex module is in the current path
try:
    from optex import OptEx
except ImportError:
    print("Error: Failed to import the optex module. Please ensure optex.py is in your Python path.")
    sys.exit(1)


# ========================================================
# 4. Logging Configuration
# ========================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ######################################################################
# 0. PASO Core Classes (from User's Code)
# ######################################################################

class Config:
    """
    Training configuration parameters - simplified for PASO synthetic benchmark.
    
    This class holds all hyperparameters for training and PASO acceleration.
    """
    # Training parameters
    batch_size: int = 512       # Not used in this script (placeholder)
    num_epochs: int = 1         # Not used in this script (placeholder)
    max_steps: int = 100        # Will be overwritten by T
    learning_rate: float = 0.1  # Will be overwritten by lr
    momentum: float = 0.9
    
    # Acceleration config (PASO-specific)
    P: int = 8                    # Window size
    threshold: float = 0.00001    # Error threshold
    ema_decay: float = 0.999      # EMA decay for threshold
    adaptivity_type: str = 'median' # Adaptivity strategy: 'mean' or 'median'
    
    # System config
    seed: int = 42                # Random seed 
    device_count: int = torch.cuda.device_count() # Number of GPUs
    
    # Model and optimizer
    optimizer_type: str = 'adam'  # Will be overwritten by optimizer_type
    
    def update(self, T: int, lr: float, opt_type: str, devices: list):
        """Update config with benchmark parameters."""
        self.max_steps = T
        self.learning_rate = lr
        self.optimizer_type = opt_type.lower()
        self.device_count = len(devices)

class ModelWrapper(nn.Module):
    """
    Model wrapper class to provide a unified interface (from User's Code).
    
    This wrapper handles differences between models that require input (like a
    standard NN) and models that are just parameter containers (like ThetaModel).
    """
    def __init__(self, model: nn.Module):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        - For ThetaModel, returns the 'theta' parameter directly.
        - For standard models, passes args and kwargs to the underlying model.
        """
        if isinstance(self.model, ThetaModel):
            # For ThetaModel, we want it to return theta
            return self.model()
        else:
            # For standard models
            return self.model(*args, **kwargs)

    def clone(self, device: str, other: 'ModelWrapper' = None) -> 'ModelWrapper':
        """
        Efficiently clones the model and its state.
        
        If 'other' is provided, it copies the state into 'other'.
        Otherwise, it creates a new ModelWrapper instance.
        """
        if other is None:  
            # Create a new instance for ThetaModel
            if isinstance(self.model, ThetaModel):
                new_model_instance = ThetaModel(self.model.theta.shape[0], device)
                other = ModelWrapper(new_model_instance).to(device)
                # Copy state
                other.model.theta.data = self.model.theta.data.clone().to(device)
            else:
                # Assume other model types support deepcopy
                other = ModelWrapper(copy.deepcopy(self.model)).to(device)
        else:
            # Copy parameters into the existing 'other' model
            with torch.no_grad():
                for param_to, param_from in zip(other.get_params(), self.get_params()):
                    param_to.data = param_from.data.clone()
        return other

    def get_params(self) -> list[nn.Parameter]:
        """Returns a list of the model's parameters."""
        return list(self.parameters())

    def set_params(self, params: list[nn.Parameter], clone: bool = True):
        """Sets the model's parameters from a list."""
        for p, q in zip(self.get_params(), params):
            if clone:  
                p.data = q.data.clone().to(p.device)
            else:  
                p.data = q.to(p.device)

    def set_grads_from_grads(self, grads: list[torch.Tensor]):
        """Sets the gradients of the model's parameters from a list of tensors."""
        for p, grad in zip(self.get_params(), grads):
            if grad is not None:  
                p.grad = grad.to(p.device)

    def compute_error_from_model(self, other: 'ModelWrapper') -> float:
        """
        Computes the mean squared error (MSE) between this model's parameters
        and another model's parameters.
        """
        my_params, other_params = self.get_params(), other.get_params()
        with torch.no_grad():
            error, total_num = 0.0, 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p - q).pow(2).item()
                total_num += np.prod(list(q.shape))
            return (error / total_num * 1e6) if total_num > 0 else 0.0

def optimizer_state_clone(optimizer_from: optim.Optimizer, optimizer_to: optim.Optimizer):
    """
    Clones the state from one optimizer to another. (from User's Code)
    """
    optimizer_to.load_state_dict(optimizer_from.state_dict())




# ######################################################################
# 2. Synthetic Function Code
# ######################################################################

def sphere_function(theta: torch.Tensor) -> torch.Tensor:
    """Computes the Sphere function: f(x) = sum(x_i^2)."""
    return torch.sum(theta.pow(2))

def ackley_function(theta: torch.Tensor) -> torch.Tensor:
    """Computes the Ackley function."""
    d = theta.shape[0]
    sum1 = torch.sum(theta ** 2)
    sum2 = torch.sum(torch.cos(2 * torch.pi * theta))
    term1 = -20 * torch.exp(-0.2 * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)
    return term1 + term2 + 20 + torch.e

def rosenbrock_function(theta: torch.Tensor) -> torch.Tensor:
    """Computes the Rosenbrock function."""
    return torch.sum(100 * (theta[1:] - theta[:-1] ** 2) ** 2 + (1 - theta[:-1]) ** 2)

class ThetaModel(nn.Module):
    """
    A simple model that wraps a single parameter vector 'theta'.
    This is used for optimizing synthetic functions.
    """
    def __init__(self, d: int, device: str, init_range: tuple[float, float] = (-5.0, 5.0)):
        super(ThetaModel, self).__init__()
        self.theta = nn.Parameter(torch.empty(d, device=device))
        nn.init.uniform_(self.theta, *init_range)

    def forward(self) -> torch.Tensor:
        """Returns the parameter vector 'theta'."""
        return self.theta

def optimize_vanilla(
    function,
    d: int,
    T: int,
    lr: float,
    num_runs: int,
    optimizer_type: str = 'Adam',
    init_range: tuple[float, float] = (-5.0, 5.0),
    max_grad_norm: float = None
) -> list[list[float]]:
    """
    Runs a standard (vanilla) optimization loop.
    
    Args:
        function: The objective function to minimize.
        d: Dimensionality of the problem.
        T: Number of optimization steps.
        lr: Learning rate.
        num_runs: Number of independent runs.
        optimizer_type: 'Adam'.
        init_range: Range for parameter initialization.
        max_grad_norm: Value for gradient clipping (if any).

    Returns:
        A list of loss histories, where each history is a list of losses.
    """
    loss_histories = []
    devices = OptEx.detect_devices()
    device = torch.device(devices[0])
    
    for run in range(num_runs):
        theta = nn.Parameter(torch.empty(d, device=device))
        nn.init.uniform_(theta, *init_range)
        
        if optimizer_type == 'SGD':
            optimizer = optim.SGD([theta], lr=lr)
        elif optimizer_type == 'Adam':
            optimizer = optim.Adam([theta], lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
        loss_history = []
        
        for t in range(T):
            optimizer.zero_grad()
            loss = function(theta)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_([theta], max_norm=max_grad_norm)

            if not torch.isfinite(theta.grad).all():
                logger.warning(f"Vanilla Run {run+1}, Iter {t+1}: Encountered non-finite gradient. Stopping run.")
                break 

            optimizer.step()
            loss_history.append(loss.item())
            
            if (t + 1) % 10 == 0:
                logger.info(f"Vanilla Run {run+1}/{num_runs}, Iteration {t+1}/{T}, Loss: {loss.item()}")
                
        loss_histories.append(loss_history)
        
    return loss_histories

def optimize_optex(
    function,
    d: int,
    T: int,
    lr: float,
    devices: list[str],
    num_runs: int,
    optimizer_type: str = 'Adam',
    init_range: tuple[float, float] = (-5.0, 5.0),
    max_grad_norm: float = None,
    use_dim_wise_kernel: bool = True
) -> list[list[float]]:
    """
    Runs the optimization loop using the OptEx parallel optimizer.
    
    Args:
        (See optimize_vanilla for common arguments)
        devices: List of device strings (e.g., ['cuda:0', 'cuda:1']).
        use_dim_wise_kernel: Flag for OptEx's kernel.

    Returns:
        A list of loss histories.
    """
    num_parall = 8
    loss_histories = []
    
    for run in range(num_runs):
        nets = [ThetaModel(d, devices[i % len(devices)], init_range=init_range) for i in range(num_parall)]
        
        if optimizer_type == 'SGD':
            opts = [optim.SGD(net.parameters(), lr=lr) for net in nets]
        elif optimizer_type == 'Adam':
            opts = [optim.Adam(net.parameters(), lr=lr) for net in nets]
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        opt_ex = OptEx(
            num_parall=num_parall,
            max_history=20,
            kernel=None,
            lengthscale=1.0,
            std=0.001,
            ard_num_dims=d if use_dim_wise_kernel else None,
            use_dim_wise_kernel=use_dim_wise_kernel,
            devices=devices
        )

        if not opt_ex.use_dim_wise_kernel:
            opt_ex.reset_activedims(d)

        loss_history = []

        for t in range(T):
            # Note: using run_parallel_iteration as it suits the f(theta) form
            opt_ex.run_parallel_iteration(
                nets, opts, function, max_grad_norm=max_grad_norm
            )

            if (t + 1) % 10 == 0:
                opt_ex.tune_lengthscale()
                if not opt_ex.use_dim_wise_kernel:
                    opt_ex.tune_activedims(d)

            # Get loss from the primary model
            theta = nets[0].theta
            loss = function(theta).item()
            loss_history.append(loss)

            if (t + 1) % 10 == 0:
                logger.info(f"OptEx Run {run+1}/{num_runs}, Iteration {t+1}/{T}, Loss: {loss}")
                
        loss_histories.append(loss_history)
        
    return loss_histories


# ######################################################################
# 3. PASO Synthetic Function Optimization (New)
# ######################################################################

def run_worker_synthetic(
    model: ModelWrapper, 
    optimizer: optim.Optimizer, 
    objective_function, 
    queues: tuple[mp.Queue, mp.Queue], 
    device: str, 
    seed_offset: int
):
    """
    PASO worker process for f(theta) synthetic functions.
    Adapted from the user's run_worker.
    
    This function runs in a separate process, waits for tasks (parameters)
    from the main process, computes gradients, and sends them back.
    """
    while True:
        ret = queues[0].get()
        if ret is None:
            return  # Exit signal
            
        params, step = ret
        model.set_params(params, clone=False)
        
        # Simplified "take_step" for synthetic functions
        np.random.seed(step + seed_offset)
        torch.manual_seed(step + seed_offset)
        torch.cuda.manual_seed(step + seed_offset)
        
        optimizer.zero_grad()
        
        # ThetaModel.forward() returns theta
        theta = model.forward() 
        loss = objective_function(theta)
        
        loss.backward()
        
        grads = [p.grad for p in model.get_params()]
        
        queues[1].put((grads, step, {
            'loss': loss.item(),
            # Accuracy/batch size not needed here
        }))

def run_paso_synthetic_core(
    config: Config, 
    model: ModelWrapper, 
    objective_function, 
    devices: list[str]
) -> list[float]:
    """
    PASO main loop for f(theta) synthetic functions.
    Adapted from the user's train_loop_parallel.
    
    This function manages the main (master) model and coordinates the worker
    processes for parallel gradient computation and fixed-point iteration.
    """
    if config.device_count <= 1:
        logger.warning("PASO requires at least 2 devices (1 master, 1+ workers). Falling back.")
        # Note: This is just a quick fallback.
        # A full vanilla implementation is not provided here.
        return []

    mp.set_start_method('spawn', force=True)
    queues = (mp.Queue(), mp.Queue())
    processes = []
    
    # Create worker processes on devices[1:]
    for rank in range(1, config.device_count):
        worker_device = torch.device(devices[rank])
        worker_model = model.clone(worker_device)
        
        if config.optimizer_type.lower() == 'sgd':
            worker_optimizer = optim.SGD(worker_model.parameters(), lr=config.learning_rate)
        elif config.optimizer_type.lower() == 'adam':
            worker_optimizer = optim.Adam(worker_model.parameters(), lr=config.learning_rate)
        else:
            # Default to AdamW (as in user's code)
            worker_optimizer = optim.AdamW(worker_model.parameters(), lr=config.learning_rate)
            
        p = mp.Process(target=run_worker_synthetic, args=(
            worker_model, worker_optimizer, objective_function, 
            queues, worker_device, config.seed
        ))
        p.start()
        processes.append(p)

    # Main (master) device
    device = torch.device(devices[0])
    model = model.to(device)
    
    T = config.max_steps
    P = min(config.P, T)
    thresh = config.threshold
    
    models = [None] * (T + 1)
    optimizers = [None] * (T + 1)
    
    begin_idx, end_idx = 0, P
    total_iters, total_Grad = 0, 0
    
    # Final loss history
    loss_history = [] 

    for step in range(P + 1):
        models[step] = model.clone(device)
        if config.optimizer_type.lower() == 'sgd':
            optimizers[step] = optim.SGD(models[step].parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer_type.lower() == 'adam':
            optimizers[step] = optim.Adam(models[step].parameters(), lr=config.learning_rate)
        else:
            optimizers[step] = optim.AdamW(models[step].parameters(), lr=config.learning_rate)
            
    # pbar = tqdm(total=T, desc=f"PASO ({config.optimizer_type})")
    start_time = time.time()

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

        # 3. Perform fixed-point iteration
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
                # Collect loss from the last converged model in the window
                current_step_loss = metrics[i]['loss']
            
            if ind is not None:
                models[step + 1] = rollout_model.clone(device, models[step + 1])
            
            total_Grad += 1

        # 4. Update threshold
        thresh = thresh * config.ema_decay + errors_all * (1 - config.ema_decay)
        
        # 5. Slide window
        # progress = ind - begin_idx
        # pbar.update(progress)
        total_iters += 1
        
        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)

        # Recycle models and optimizers for the new window extension
        for step in range(end_idx + 1, new_end_idx + 1):
            prev_model_idx = step - 1 - parallel_len
            models[step] = rollout_model.clone(device, models[prev_model_idx] if prev_model_idx >= 0 else None)
            optimizers[step] = optimizers[prev_model_idx] if prev_model_idx >= 0 else None

        begin_idx = new_begin_idx
        end_idx = new_end_idx

        # 6. Log progress
        # Store the loss for this step to return
        loss_history.append(current_step_loss)
        logger.info(f"PASO Run Iteration {total_iters} (Step {begin_idx}), Loss: {current_step_loss}")

    
    # Cleanup
    for _ in processes:
        queues[0].put(None)  # Send exit signal
    for p in processes:
        p.join()
        
    logger.info(f"PASO Complete. Total iterations {total_iters} (vs {T} steps). Effective Speedup: {T/total_iters:.2f}x")

    return loss_history


def optimize_paso(
    function,
    d: int,
    T: int,
    lr: float,
    devices: list[str],
    num_runs: int,
    optimizer_type: str = 'SGD',
    init_range: tuple[float, float] = (-5.0, 5.0),
    max_grad_norm: float = None
) -> list[list[float]]:
    """
    PASO wrapper for synthetic benchmarks.
    
    This function sets up the Config and ModelWrapper required by the
    PASO core logic and runs it for `num_runs`.
    """
    if max_grad_norm is not None:
        logger.warning("PASO (as implemented here) does not currently support max_grad_norm. Ignoring.")

    loss_histories = []
    
    for run in range(num_runs):
        logger.info(f"PASO Run {run+1}/{num_runs} starting...")
        
        # 1. Create config
        config = Config() # from User's Code
        config.update(T, lr, optimizer_type, devices)
        config.seed = 42 + run # Change seed for each run
        
        # 2. Create model
        # PASO code assumes model is initially on devices[0]
        base_device = torch.device(devices[0])
        # We need the ModelWrapper from PASO
        base_model = ModelWrapper(ThetaModel(d, base_device, init_range)).to(base_device)

        # 3. Run PASO core logic
        loss_history = run_paso_synthetic_core(config, base_model, function, devices)
        loss_histories.append(loss_history)
        
    return loss_histories


# ######################################################################
# 4. Main Function (Integration & Plotting) - Updated
# ######################################################################

def plot_results(
    vanilla_loss_histories: list[list[float]],
    optex_loss_histories: list[list[float]],
    paso_loss_histories: list[list[float]], # <-- Added
    f_star: float,
    func_name: str,
    output_path: str
):
    """Plot and save comparison results."""

    vanilla_mean_loss = torch.tensor(vanilla_loss_histories).mean(dim=0).cpu().numpy()
    vanilla_opt_gap = vanilla_mean_loss - f_star

    optex_mean_loss = torch.tensor(optex_loss_histories).mean(dim=0).cpu().numpy()
    optex_opt_gap = optex_mean_loss - f_star

    # --- Added ---
    paso_mean_loss = torch.tensor(paso_loss_histories).mean(dim=0).cpu().numpy()
    paso_opt_gap = paso_mean_loss - f_star
    # --- End ---

    epsilon = 1e-20

    vanilla_opt_gap_log = torch.log10(torch.clamp(torch.tensor(vanilla_opt_gap), min=epsilon)).numpy()
    optex_opt_gap_log = torch.log10(torch.clamp(torch.tensor(optex_opt_gap), min=epsilon)).numpy()
    paso_opt_gap_log = torch.log10(torch.clamp(torch.tensor(paso_opt_gap), min=epsilon)).numpy() # Added
    y_label = 'Optimality Gap (log scale)'


    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(vanilla_opt_gap_log) + 1),
        vanilla_opt_gap_log,
        linestyle='--',
        marker='x',
        markersize=5,
        color='blue',
        label='Vanilla'
    )
    plt.plot(
        range(1, len(optex_opt_gap_log) + 1),
        optex_opt_gap_log,
        linestyle='-',
        marker='o',
        markersize=5,
        color='orange',
        markerfacecolor='none',
        label='OptEx'
    )
    # --- Added ---
    plt.plot(
        range(1, len(paso_opt_gap_log) + 1),
        paso_opt_gap_log,
        linestyle=':',
        marker='s',
        markersize=5,
        color='green',
        markerfacecolor='none',
        label='PASO'
    )
    # --- End ---

    # --- Font size settings ---
    AXIS_LABEL_SIZE = 26
    TICK_LABEL_SIZE = 26
    TITLE_SIZE = 30
    LEGEND_SIZE = 26
    # --------------------------

    plt.xlabel('Iterations', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    plt.title(f'{func_name} Function Optimization (8 GPUs)', fontsize=TITLE_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)

    # --- Set tick label sizes ---
    plt.xticks(fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    # ------------------------

    all_log_gaps = torch.cat([
        torch.tensor(vanilla_opt_gap_log), 
        torch.tensor(optex_opt_gap_log),
        torch.tensor(paso_opt_gap_log) # Added
    ])
    all_log_gaps = all_log_gaps[torch.isfinite(all_log_gaps)]
    
    if len(all_log_gaps) > 0:
        y_min = int(torch.floor(torch.min(all_log_gaps)).item())
        y_max = int(torch.ceil(torch.max(all_log_gaps)).item())
        if y_max - y_min > 0:
            y_ticks_major_interval = max(1, (y_max - y_min) // 5) # Aim for ~5-10 major ticks
            y_ticks = range(y_min, y_max + 1, y_ticks_major_interval)
            
            # Ensure custom Y-axis ticks also use the large font size
            plt.yticks(y_ticks, [str(tick) for tick in y_ticks], fontsize=TICK_LABEL_SIZE)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved {func_name} function optimization plot to '{output_path}'.")



def main():
    """
    Main execution function.
    
    Sets up the environment, runs the benchmarks for each synthetic function,
    and calls the plotting function to save the results.
    """
    # Ensure __file__ is available, fallback to '.' for REPL/notebook
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.path.abspath('.')
        
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    devices = OptEx.detect_devices()
    if not devices:
        logger.error("No available devices (CPU/GPU) detected. Exiting.")
        return
    elif len(devices) < 2 and 'cuda' in devices[0]:
         logger.warning(
             "Detected fewer than 2 GPUs. PASO and OptEx may not function "
             "correctly or provide speedup. PASO will fallback."
         )
         # Still continue, but PASO will fallback or fail
    
    logger.info(f"Detected devices: {devices}")
        
    num_runs = 1 # Number of independent runs (e.g., set to 3 for robustness, 1 for speed)

    # --- 1. Run Synthetic Functions ---
    functions_synthetic = {
        'Sphere': (sphere_function, 0.0),
        'Ackley': (ackley_function, 0.0),
        'Rosenbrock': (rosenbrock_function, 0.0)
    }

    for func_name, (function, f_star) in functions_synthetic.items():
        logger.info(f"\nRunning optimization for {func_name} function")
        optimizer_type = 'Adam'
        T = 1000
        
        if func_name == 'Rosenbrock':
            d = int(1e4)
            lr = 0.0001
            init_range = (-1.0, 1.0)
            max_grad_norm = 100.0
            use_dim_wise_kernel = True
        else: # Sphere & Ackley
            d = int(1e4)
            lr = 0.001
            init_range = (-1.0, 1.0)
            max_grad_norm = None
            use_dim_wise_kernel = True

        logger.info("Running Vanilla optimization (synthetic)")
        vanilla_loss_histories = optimize_vanilla(
            function=function, d=d, T=T, lr=lr, num_runs=num_runs,
            optimizer_type=optimizer_type, init_range=init_range,
            max_grad_norm=max_grad_norm
        )

        logger.info("Running OptEx optimization (synthetic)")
        optex_loss_histories = optimize_optex(
            function=function, d=d, T=T, lr=lr, devices=devices,
            num_runs=num_runs, optimizer_type=optimizer_type,
            init_range=init_range, max_grad_norm=max_grad_norm,
            use_dim_wise_kernel=use_dim_wise_kernel
        )

        # --- Added: Run PASO ---
        logger.info("Running PASO optimization (synthetic)")
        paso_loss_histories = optimize_paso(
            function=function, d=d, T=T, lr=lr, devices=devices,
            num_runs=num_runs, optimizer_type=optimizer_type,
            init_range=init_range, max_grad_norm=max_grad_norm
        )
        # --- End ---
        
        output_path = os.path.join(output_dir, f'{func_name}_{optimizer_type}_optimization.png')
        plot_results(
            vanilla_loss_histories, 
            optex_loss_histories, 
            paso_loss_histories, # <-- Added
            f_star, 
            func_name, 
            output_path
        )


if __name__ == "__main__":
    main()