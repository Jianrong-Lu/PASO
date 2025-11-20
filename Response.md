# General Response to all Reviewers, ACs, and PCs

We sincerely thank the reviewers for their time and effort in carefully reviewing this manuscript. We are encouraged by the exceptionally positive assessment on the **strong conceptual novelty** (Reviewers bekR, ypoz, VTKr), **solid theoretical foundations** (Reviewers bekR, ypoz), and **promising speedup results** (Reviewers bekR, kux2).

The reviewers have raised common concerns regarding reproducibility, comparisons relative to existing parallel methods (such as OptEx and strong parallel methods), memory costs, the low accuracy and perplexity, and the validity of specific experimental metrics. In response, **we have some exciting theoretical and experimental results** addressing these concerns:

* **Reproducibility**: We have open-sourced our code with a detailed running script, **full environment specifications (including CUDA/NCCL versions, hardware topology), library dependencies, and random seed configurations (Reviewers VTKr, kux2)**. Our code: https://anonymous.4open.science/r/PASO-8ECE/.
* **Computation, Communication, and Space Complexity:** We have added both theoretical complexity analyses and empirical comparisons against the approximate method, OptEx. **Excitingly**, our analysis reveals that PASO achieves a linear speedup rate, whereas OptEx is limited to a sub-linear. Furthermore, PASO maintains significantly lower space complexity ($\mathcal{O}(Nd)$)  and compared to OptEx ($\mathcal{O}((T_0+N)d)$).  In our memory-efficient implementation `cv_nccl.py`, PASO maintains only one optimizer and model per device, matching the memory footprint of data parallelism. These makes PASO fundamentally scalable for large models. 
* **Competitive Performance against Fully Optimized Parallel Baselines:** We compared our naive PASO implementation against both naive Data Parallelism (DP) implementation and fully optimized industrial DP (PyTorch `DDP`). **Remarkably**, our naive PASO implementation is $2+$ times faster then the naive DP, and matches the performance of fully optimized industrial DP (achieving a $\approx 5.1\times$ speedup), while consuming the minimum number of tokens. This highlights the immense potential of PASO, particularly because its inter-step parallelism is orthogonal to—and can be combined with—intra-step DP.
* **Validation under Standard Implementation:** We have addressed the implementation issues regarding low accuracy and and perplexity calculations. Our updated results, now reported with multiple random seeds and standard deviations, **match community benchmarks**. This confirms that PASO accelerates training without compromising model quality.

**We sincerely invite you to refer to the detailed responses below and the provided source code. We are genuinely excited about these findings,** as they validate **PASO as a fundamental advancement in parallel training**—introducing **the first step-parallel training method without the loss of quality** based on equation system solving. We agree with the reviewers on the importance of scaling this approach further and are eager to pursue future explorations, such as combining PASO with industrial-grade distributed frameworks. 

**Looking ahead**, our vision is to evolve PASO into a robust, industrial-caliber  step-parallel training ecosystem. By synergizing our step-parallel approach with existing  parallelism and memory optimization techniques like ZeRO, we aim to unlock unprecedented efficiency in the training of LLMs.

**Last but not least**, we thank the PCs, ACs, and reviewers again for their invaluable time and effort. We commit that the final accepted manuscript will include  the newly added experiments and analyses. Furthermore, **we will ensure the reviews and author discussion remain public as well.**

### **Response to Reviewer bekR** （负分4）

We thank Reviewer bekR for the positive feedback.

> Comparing with parallel training on multiple GPUs.

We clarify that directly comparing our naive PASO implementation that doesn't  incorporate any optimizations on the memory with fully optimized industrial parallel methods  **would be highly unfair and unreasonable**. To make a fair comparison, we compare our naive PASO with naive data parallelism (DP)  and its fully optimized implementation. 

**The results is really excited**. It confirms our naive implementation of PASO  consumes the minimum number of tokens but achieves a 2.3× speedup over naive DP, matching industrial DP performance. Crucially, PASO is inter-step parallelism while DP uses intra-step parallelism. We believe, their complementary nature can yield more  acceleration when combined.

Table 2: Comparisons on GPT-2 over WiKiText-2 (see `llm_v2.py`). Batch size 112; learning rate 6e-5

|Method|PPL$\downarrow$|Total tokens$\downarrow$|Iters$\downarrow$|time (s)$\downarrow$|
|-|-|-|-|-|
|Adam|20.4|7344101|1k|614|
|Adam+Naive DP|20.5|7348942|1k|340 (1.8$\times$)|
|Adam+Full optimized industrial DP (Pytorch's `DDP`)|20.4| 7348942|1k|**118** (5.2$\times$)|
|Adam+Naive PASO|20.5|**7312230**|**0.16k**|146 (5.1$\times$)|

> No memory overhead analysis is discussed in the study.

Tab. 1 states the overhead  is "1 model + 1 optimizer" per device (see `cv_nccl.py`).  More analysis is shown below (full details in App. A).

Table 2: Total per-iteration complexity comparison. Denote by $N$ GPU count, $d$ model dimension, $t_{comm}$ the comm. time, $T_0$ the storage historical gradients number of OptEx.
|Metric|OptEx|PASO|
|-|-|-|
|Comp. cost|$\mathcal{O}(T_0^3 + N T_0 (d + T_0))$|$\mathcal{O}(N^2 d)$|
|Space cost|$\mathcal{O}(T_0 d + T_0^2 + N d)$|$\mathcal{O}(N d)$|
|Comm. cost|$\mathcal{O}(t_{comm})$|$\mathcal{O}(t_{comm})$|
|Speedup rate|$\mathcal{O}(\sqrt{N})$|$\mathcal{O}(N)$|

This analysis reveals a critical practical bottleneck for OptEx.

- **Space Bottleneck**: OptEx's space complexity is $\mathcal{O}((T_0 + N)d)$, scaling with its required history gradient size $T_0$. In contrast, PASO's is only $\mathcal{O}(Nd)$.
- **Practical Implication**: The $\mathcal{O}(T_0 d)$ term, which arises from storing $T_0$ historical gradients, makes OptEx **spatially infeasible for large-scale models** (where $d$ is massive), which are the primary advantage of our PASO.
- **Faster speedup**: PASO achieves a **linear speedup** rate of $\mathcal{O}(N)$ while OptEx only achieves a **sub-linear** speedup of $\mathcal{O}(\sqrt{N})$. 

> The convergence analysis  corresponding to $g$  is not presented in the main text. Prop. 2 claims the convergence but it could be arbitrary slow.

We clarify that the main text in Prop. 2 gives the convergence analysis for $g$ from different optimizers. Prop. 2 doesn't show our method is arbitrary slow. It only confirms our method never converges more slowly than sequential baselines. Our experiments demonstrate that PASO converges extremely rapidly.  

> Alg. 1 does not use $\delta$ for any stopping decision and does not update $t$ in inner iteration. In addition the role of $\delta$ is also missing in Figure 1.

We clarify that the tolerance $\delta$  are explicitly used in Alg.1 (Eq.9) and Figure 1.

1. **Alg.1**: $\delta$ is used in **Line 8** (i.e., $d(\cdot) \le \delta$ in Eq.9). 
3. **Fig. 1**: Both $\delta$ are shown in the box labeled "Check error and calculate stride" and "Adaptive tolerance update".

> What is $r$, $g_\tau$, $N$, $\alpha$?

We clarify that we have clearly defined these notations

- **$r$**  is the number of history weights used by the optimizer. For optimizers like Adam, it depends on all prior weights, so $r=t$ (see App. G).
- **$g_\tau$** is the general gradient update term for a specific optimizer. It is defined for SGD in Eq. (4) and for Adam in Eq. (15). 
- **$N$** (Table 1, lines 308-309) is the number of GPUs.
- **$\alpha$** (Table 1, lines 306-307) is the communication-to-computation time ratio.

> What is the reason of limitation the batch size to 30 for the LLaMa 3.2-1B model?

Our current implementation (see `llm_v2.py`) is memory-cost in which we adopt a master-worker multiprocessing way where the master maintains a window of model. Therefore, for 1B model, the maximum batch size is limited to 30. However, we want to highlight that this is an engineering bottleneck, not a fundamental constraint of the PASO algorithm. Our  `cv_nccl.py` explores a memory-efficient method where only a single model and optimizer are maintained per GPU.

> Do you have any ideas how to combine your approach with existing parallelism techniques? 

For a model too large for one GPU, one would *first* apply **intra-step** parallelism to shard the model *within* a single step. PASO would then operate *on top* of this, managing the **inter-step** parallelism. 

### **Response to Reviewer ypoz** （6分正分）

Thank for the very positive and encouraging review. 

> Empirical Validation Scope ... 

We agree that scaling beyond 8 GPUs  represent critical next steps. Given current computational constraints (limited to 8 GPUs), large-scale validation is reserved for future work. **Significantly, PASO introduces the first step-parallel training method  without compromising model accuracy, a fundamental advancement in parallel training.**

>  Might not generalize to industrial-scale systems.

We clarify that this limitation stems from the engineering constraints of our research prototype, not any fundamental constraint of PASO. Our `cv_nccl.py` resolves this bottleneck  by requiring each GPU to maintain only a single model and optimizer. 

>  Clearer pseudocode; Notation Complexity.

We clarify that we have included pseudocode (Alg. 1) and diagram (Fig. 1). Due to space limitation, the notations are shown in App. D.

> Theoretical comparison against classical parallel SGD variants ; more discussion on communication complexity and gradient staleness.

We clarify that our PASO method fundamentally differs from asynchronous parallel approaches. These methods achieve data parallelism by tolerating **stale gradients** which can harm or alter the convergence path.  PASO operates as a **step-parallel** framework that inherently avoids gradient staleness (Prop. 1 & 2). Thus, asynchronous optimization techniques are not directly comparable to our approach. The most relevant comparison is OptEx, which similarly employs step-parallelism. We provide a theoretical comparison  below.

Table 2: Total per-iteration complexity comparison. Denote by $N$ GPU count, $d$ model dimension, $t_{comm}$ the comm. time, $T_0$ the storage historical gradients number of OptEx.
|Metric|OptEx|PASO|
|-|-|-|
|Comp. cost|$\mathcal{O}(T_0^3 + N T_0 (d + T_0))$|$\mathcal{O}(N^2 d)$|
|Space cost|$\mathcal{O}(T_0 d + T_0^2 + N d)$|$\mathcal{O}(N d)$|
|Comm. cost|$\mathcal{O}(t_{comm})$|$\mathcal{O}(t_{comm})$|
|Speedup rate|$\mathcal{O}(\sqrt{N})$|$\mathcal{O}(N)$|

This analysis reveals a critical practical bottleneck for OptEx.

- **Space Bottleneck**: OptEx's space complexity is $\mathcal{O}((T_0 + N)d)$, scaling with its required history gradient size $T_0$. In contrast, PASO's is only $\mathcal{O}(Nd)$.
- **Practical Implication**: The $\mathcal{O}(T_0 d)$ term, which arises from storing $T_0$ historical gradients, makes OptEx **spatially infeasible for large-scale models** (where $d$ is massive), which are the primary advantage of our PASO.
- **Faster speedup**: PASO achieves a **linear speedup** rate of $\mathcal{O}(N)$ while OptEx only achieves a **sub-linear** speedup of $\mathcal{O}(\sqrt{N})$. 

> How does PASO behave under non-smooth losses or non-convex constraints?

We clarify that PASO specifically accelerates gradient-descent-based methods, which are inherently ill-suited for non-smooth objectives. Thus, non-smooth loss functions lie outside the scope of PASO. 

Regarding non-convex constraints, our experiments demonstrate PASO's compatibility, having validated it on the highly non-convex model. Extending the PASO to diffusion models or reinforcement learning is a promising research direction, which we defer to future work.

> Can PASO be adapted to work with second-order or implicit optimizers?

Yes. PASO is a general framework. As long as the optimizer's update rule, $g_\tau$, can be formulated，which we demonstrated for SGD, Adam, and AdamW, PASO can be applied. Adapting it to second-order methods would simply require correctly defining the more complex $g_\tau$ term.

> How does PASO handle gradient noise accumulation when the window size (p) grows large?

PASO does not suffer from noise *accumulation* in the traditional sense.  PASO converges to the exact trajectory defined by a fixed set of $p$ mini-batches. The "noise" from this specific batch set is fully incorporated. Once it converges, the window slides forward by $s$ steps, and *new* mini-batches are sampled for the new end of the window. This rapid re-sampling prevents the trajectory from diverging based on a single "unlucky" set of $p$ batches.

> Are the guarantees affected when using mixed precision or quantized gradients?

PASO is fully compatible with them. PASO's guarantees are about finding the unique solution to the equation system. If the system is defined using mixed-precision or quantized gradients, PASO will simply find the exact trajectory corresponding to that mixed-precision/quantized optimization process. 





### **Response to Reviewer kux2** (2分)

We thank Reviewer kux2 for the feedback. We believe there are key misunderstandings about our method's memory and theoretical claims.

> Prop.2 only shows one can reproduce GD in $K \le T$ outer iterations, and there is no theorem that $K \ll T$ .

We clarify that Prop. 2 guarantees only that PASO converges to the exact sequential trajectory at a speed no slower than sequential methods. Our core claim regarding speedup is an empirical finding (Lines 200–201). Given the inherent complexity of black-box neural network optimization, providing a principled proof for this speedup remains challenging, and we thus defer it to future work.

> The method ... requires $p$ simultaneous graphs ... The paper nevertheless claims "1 model + 1 optimizer" storage per device in Tab. 1, which appears optimistic. Will that implicitly change the sampling process and may require staging many batches concurrently? How are the future mini-batches produced and pinned to devices? Is the distribution i.i.d. with replacement? Does pre-sampling introduce measurable drift?

We clarify that our "1 model + 1 optimizer" claim is accurate and realized in our implementation. As shown in `cv_nccl.py`, each rank instantiates exactly one model (`model = CNN().to(device)`) and one optimizer (`paso_optimizer`). At any iteration $k$, rank $r$ is assigned a specific time step $t+i$. It loads **only** the data batch corresponding to step $t+i$ and computes gradients for that specific step. Thus, the memory cost per device is strictly  1 model and 1 optimizer, identical to standard SGD.

Regarding sampling, we implement a `PASODataLoader` that wraps standard PyTorch loaders to avoid the memory overhead of staging concurrent batches. Instead of pre-loading tensors, we pre-cache lightweight batch indices, allowing each device to deterministically retrieve the specific mini-batch required for its assigned time step $t+i$ on demand. This guarantees that the data distribution remains identical to standard sequential training (i.i.d.) and introduces no statistical drift or additional memory footprint. For instance, consider a  `PASODataLoader`  containing 3 batches, $\zeta_{0}, \zeta_{1}, \zeta_{2}$, across 6 sequential steps ($t=0$ to $t=5$).

- **Epoch 1** Steps $t=0, 1, 2 $ correspond to $\zeta_{0}, \zeta_{1}, \zeta_{2}$.
- **Epoch 2** Steps $t=3, 4, 5$ correspond to  $\zeta_{0}, \zeta_{1}, \zeta_{2}$.

When a worker computes the gradient at step $t=0$, it retrieves $\zeta_{0}$ from its local dataloader. When computing the gradient at $t=4$, it retrieves $\zeta_{1}$ since $4 \pmod 3 = 1$.

> what are the measured peak GPU memory ... for PASO vs. strong DP/PP/MP baselines? Furthermore, the paper does not include comparisons with strong system baselines such as fully optimized ZeRO...

Our `cv_nccl.py` has shown that our naive PASO has the same memory cost to naive data parallelism (DP). 

We clarify that directly comparing our naive PASO implementation that doesn't  incorporate any optimizations on the memory with fully optimized industrial parallel methods  **would be highly unfair and unreasonable**. 

To make a fair comparison, we compare our naive PASO with naive DP  and its fully optimized implementation. 

**The results is really excited**. It confirms our naive implementation of PASO  consumes the minimum number of tokens but achieves a 2.3× speedup over naive DP, matching industrial DP performance. Crucially, PASO is inter-step parallelism while DP uses intra-step parallelism. We believe, their complementary nature can yield more  acceleration when combined.

Table 2: Comparisons on GPT-2 over WiKiText-2 (see `llm_v2.py`). Batch size 112; learning rate 6e-5
|Method|PPL$\downarrow$|Total tokens$\downarrow$|Iters$\downarrow$|time (s)$\downarrow$|
|-|-|-|-|-|
|Adam|20.4|7344101|1k|614|
|Adam+Naive DP|20.5|7348942|1k|340 (1.8$\times$)|
|Adam+Full optimized industrial DP (Pytorch's `DDP`)|20.4| 7348942|1k|**118** (5.2$\times$)|
|Adam+Naive PASO|20.5|**7312230**|**0.16k**|146 (5.1$\times$)|


> The reported accuracy and perplexity results are presented without multiple seeds or error bars. 

We'll run all experiments with 5 different random seeds and update all tables to report the mean and standard deviation for accuracy and perplexity.

> The EMA-based tolerance update rule $\delta$ has no accuracy guarantee, it is tuned by Wandb sweeps rather than relying on derived principle.

We clarify that using EMA for adaptive estimation is a well-established paradigm in deep learning (e.g., the moving average of gradients in Adam). These methods prioritize adaptive smoothing over static theoretical bounds, which often fail to capture the dynamic variance of real-world training. Besides, deriving an exact principle for $\delta$ would require assumptions about the loss landscape (e.g., Lipschitz continuity constants) that are unknown or fluctuate during training. Thus, an empirical EMA adaptation is more practical and effective.



### **Response to Reviewer VTKr** （2分）

Thank you for the nice suggestions.  **We have opened our code (see the general response)**. 

> Lack of comparison with OptEx.  A lighter OptEx could be more practical. 

**Empirical comparison**: We integrate our PASO into the official OptEx codebase. On OptEx's benchmark, PASO has a higher speedup than OptEx. 
Table 1:  Comparisons on OptEx benchmarks  for Ackley function (full details in App. B and `optex_cmp.py`)
|Method|Optimality gap $\downarrow$|Iters $\downarrow$| Speedup $\uparrow$|
|-|-|-|-|
|Adam|$0.40\pm 0.18$|1000|$1.0\times$|
|OptEx|$0.40\pm 0.06$|162|$6.17\times$|
|PASO|$0.40\pm 0.11$|86|$11.63\times$|

**Theoretical comparison**: A precise cost comparison is challenging due to vastly different engineering implementations. We thus conducted a more fundamental complexity analysis (full details in App. A).
Table 2: Total per-iteration complexity comparison. $N$ GPU count, $d$ model dimension, $t_{cm}$ the comm. time, $T_0$, the number of storage historical gradients of OptEx.
|Metric|OptEx|PASO|
|-|-|-|
|Comp. cost|$\mathcal{O}(T_0^3 + N T_0 (d + T_0))$|$\mathcal{O}(N^2 d)$|
|Space cost|$\mathcal{O}(T_0 d + T_0^2 + N d)$|$\mathcal{O}(N d)$|
|Comm. cost|$\mathcal{O}(t_{cm})$|$\mathcal{O}(t_{cm})$|
|Speedup rate|$\mathcal{O}(\sqrt{N})$|$\mathcal{O}(N)$|

This analysis reveals a critical practical bottleneck for OptEx that refutes the "lighter" assumption.
- OptEx's space complexity scales with its $T_0$, making it **spatially infeasible for large-scale models** (where $d$ is massive). In contrast, PASO's is only $\mathcal{O}(Nd)$, which is the same as existing data parallelism.
- PASO achieves a **linear speedup** rate of $\mathcal{O}(N)$ while OptEx only achieves a **sub-linear** speedup of $\mathcal{O}(\sqrt{N})$. 

These evidences strongly supports that PASO is  *a lighter method with significantly higher efficiency and practical applicability than OptEx. **We highlight that PASO is the first step-parallel training method based on equation system solving. This inherently distinguishes it from OptEx and is a fundamental breakthrough in parallel training.**

> Reproducibility and soundness check issue.

Our public code provides full details like execution scripts, etc. (see `README.md`). We includes rigorous validation on learning rate & batch size sensitivity analysis (Tab.5&Fig. 3 in paper)

> Include plots using num_tokens to allow fair comparison with DDP and other baselines under the same computational budget. Also clarify how num_tokens was defined and whether it was kept consistent across all experiments.

We define token count as tokens processed by model per iteration, reporting the cumulative total consumed tokens during training. Our naive PASO implementation is compared against both naive and fully-optimized Data Parallelism (DP). **The results are really exciting**. Our naive PASO  consumes the minimum number of tokens but achieves a 2.3× speedup over the naive DP, matching industrial DP performance. 

Table 2: Comparisons on GPT-2 over WiKiText-2 (see `llm_v2.py`). Batch size 112; learning rate 6e-5
|Method|PPL $\downarrow$|Total tokens $\downarrow$|Iters $\downarrow$|time (s)$\downarrow$|
|-|-|-|-|-|
|Adam|20.4|7344101|1k|614|
|Adam+Naive DP|20.5|7348942|1k|340 (1.8$\times$)|
|Adam+Full optimized industrial DP (Pytorch's `DDP`)|20.4| 7348942|1k|**118** (5.2$\times$)|
|Adam+Naive PASO|20.5|**7312230**|**0.16k**|146 (5.1$\times$)|

We'll plot the loss and perplexity curves w.r.t consumed tokens. 

> Ambiguity in metric definitions. Is token-level accuracy?

 We'll keep them consistent. For LLM, it is (see `llm_v2.py`).

> The low accuracies/perplexity.  Clarify  setup; WikiText-2 vs.103 usage; comparisons under identical conditions; top-1 or top-5 accuracy? 

The low accuracy on CV tasks is likely because our models were trained with a fixed learning rate (lr) (see `cv_v1.py`). After using a decaying lr (see `cv_v2.py`), our accuracy matches existing benchmark.

Table 2: Top 1 Accuracy (ACC) comparison for CNN on CIFAR10 (`cv_v2.py`).  
|Method|ACC $\uparrow$|Iters $\downarrow$|Speedup $\uparrow$|
|-|-|-|-|
|Adam|82.04 $\pm$ 0.22|60k|1 $\times$|
|PASO|81.59 $\pm$ 0.13|9.8k| 6.1 $\times$        |

The low PPL results from the incorrect use of repetitive padding tokens for PPL computation (see `llm_v1.py`).  After fixing this bug (see `llm_v2.py`), our PPL shows a comparable value  v.s. community benchmarks.

Table 2: Comparisons for Llama-3.2-1B on WikiText-2 (`llm_v2.py`)
|Method|PPL $\uparrow$|Iters $\downarrow$|Speedup $\uparrow$|
|-|-|-|-|
|Adam|13.28 $\pm$ 0.36$|1000| 1 $\times$|
|PASO|13.29 $\pm$ 0.40$ |81| 12.3 $\times$|

> Direct W&B Fig.; difficult to assess differences across methods. Tab. 4–5 omit information. 

We plot Fig. 2 ourselves (see `plot.py`).  We believe different methods are comparable as we compare them on the same metric. We've fixed Tab. 4–5.

>  ViT  details.

A plain ViT model is trained *from scratch* (see `cv_v2.py`). 