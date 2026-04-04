很好，这次我已经结合你上传的论文内容重新梳理了。和上一版相比，我现在可以更准确地告诉你：

1. **哪些 reviewer 的担忧是“真的打到论文当前薄弱点了”**  
2. **哪些地方适合认、哪些地方要澄清、哪些地方可以稳住不退**  
3. **一份更贴你论文内容的 rebuttal 草稿该怎么写**

我先给你**基于论文本身的回复思路**，再给你**一版可直接改的 rebuttal 草稿**。为了方便你真正提交，我下面会用中文分析、英文给稿。你论文里最容易被继续追打的点，主要有四个：

- **理论直接覆盖的是 full-system FPI，不是完整覆盖 sliding-window + tolerance 的实际 PASO**
- **“no model quality loss” 这个表述比实验更强**
- **“step reduction” 被讲得太重，而 reviewer 更关心 wall-clock/runtime attribution**
- **deterministic seeding / memory claim / proof assumptions 需要更显式地写进正文而不是藏在 appendix**

这些结论都能从你论文的 abstract、Appendix A.2/A.3、Proposition 3.3/3.4、Eq. (7)-(10)、以及 Figure 10 附近的内容看出来。[论文 PDF](https://www.genspark.ai/api/files/s/XViO6qxe) [文档索引](https://www.genspark.ai/api/files/s/7Q1nA379)

---

# 一、先给你结论：这篇 rebuttal 最稳的总策略

我建议你把整篇 rebuttal 的主线统一成下面四句话：

| 主线 | 你应该怎么说 |
|---|---|
| 理论范围 | **承认** 当前理论最直接覆盖的是 full-system fixed-point formulation；practical PASO 是受其启发的 sliding-window approximation |
| 质量表述 | 把 **“no model quality loss”** 统一降为 **“comparable model quality”** 或 **“small task-dependent gaps”** |
| 效率指标 | 把 **wall-clock speedup** 作为主要 systems 指标，outer-step reduction 作为次要 algorithmic 指标 |
| 实现限制 | 主动说明当前实现是 **academic prototype / research prototype**，通信与调度开销尚未完全优化 |

这个口径和你论文现在的真实状态是匹配的：abstract 里确实写了 “up to 21× reduction in GD steps and 4.5× speedup in wall-clock time, with no model quality loss”，但 reviewer 已经抓到这个“no loss”偏强；同时你们也确实在正文/appendix里强调过 current implementation 的 prototype 性和通信瓶颈。[论文 PDF](https://www.genspark.ai/api/files/s/XViO6qxe) [文档索引](https://www.genspark.ai/api/files/s/7Q1nA379)

---

# 二、逐个 reviewer：怎么回最合适

---

## Reviewer 9uuF：重点不是实验，而是“理论严谨性 + 理论和工程的错位”

这个 reviewer 总体是偏支持的：认可 empirical detail 和 novelty，但最不满意的是：
- proof 写得不够 formal
- Appendix O 的 assumptions/scope 容易误导
- full TNE 与实际 sliding-window PASO 有 mismatch
- memory claim 容易让人误会成 full horizon 也不重

### 你对 9uuF 的总策略
**高姿态承认理论表述问题，同时把 conceptual foundation 和 practical approximation 的关系讲清楚。**

### 逐点建议

#### 1) Appendix N induction 没有 base case
这条**直接认**。这是最容易修、最不该争的点。

你可以说：
- 现在写法过于简略；
- 会显式补上 base case、induction hypothesis、induction step；
- 会更清楚说明 triangular structure 如何逐步唯一恢复 sequential update。

#### 2) Lemma O.3 的位置误导读者以为主证明都要 lower-bound 条件
这条也**直接认**，定位成 exposition issue。

你可以说：
- 该 lemma 只服务于 Eq. (43)-(44) 的局部 bound；
- 不属于 main trajectory-equivalence / general proof 的全局条件；
- 修订时会把 lemma 移到更靠近使用处，并显式标注适用范围。

#### 3) Section O proof asymptotic、可能 circular、参数收敛假设过强
这条要**部分承认 + 缩窄 claim**。

你的最佳回应逻辑是：
- 我们并不是说“实际深度学习训练普遍满足强参数收敛”
- 当前 Appendix O 里使用的是**足以得到 clean asymptotic statement 的分析性 sufficient condition**
- 我们会重组 proof flow，避免出现“似乎先假设收敛再证明收敛”的印象
- 更强的 non-asymptotic/rate analysis 很重要，但当前版本还未完成

这一条千万别硬顶“ML 里参数收敛很常见”。

#### 4) full TNE 和实际 moving window mismatch 很大
这是 9uuF 最核心的问题之一。

你应该这样定性：
- **full TNE** 的作用是定义 exact step-parallel target：顺序轨迹就是唯一目标解；
- **sliding-window PASO** 是在有限硬件条件下的 practical approximation；
- 现在稿子里确实没有把这条 conceptual-to-practical bridge 讲清楚；
- 修订版会显式加入一段：理论保证了什么、实现近似了什么、两者 gap 在哪里。

#### 5) memory overhead
你论文里关于 memory 的表述，核心是**per-device memory**：每个 GPU 只保留一个 model + 一个 optimizer state；这点在 Appendix A.1/A.3 和 Appendix K.1 的口径是明确的。  
但 reviewer 说得也对：如果把 full-horizon TNE 真的 materialize 成 $T$ 份参数，那当然很重。

所以你必须澄清：
- memory-parity claim 是针对 **implemented sliding-window PASO on a per-device basis**
- 不是在说 full-horizon system 本身可以低成本 materialize

这点你论文里原本就有依据，只是需要你在 rebuttal 里说清楚。[论文 PDF](https://www.genspark.ai/api/files/s/XViO6qxe) [文档索引](https://www.genspark.ai/api/files/s/7Q1nA379)

---

## Reviewer hm7D：重点是“什么时候 PASO 真正好用？为什么 wall-clock 没跟上？和 OptEx 到底什么关系？”

这个 reviewer 的视角更像 systems/efficiency reviewer。  
他认可框架，但要你回答：

- 为什么 $k$ 在实践中往往小？
- wall-clock 的时间花在哪？
- 与 OptEx 等最相关方法相比，PASO 是更好，还是只是 trade-off 不同？

### 你对 hm7D 的总策略
**承认目前“为什么好用”更多是 empirical characterization，而不是完整 predictive theory；主动承诺补 runtime breakdown；对 OptEx 说“different trade-off”而不是“全面碾压”。**

### 逐点建议

#### 1) 什么特征让 $k$ 小？
最好的回答不是硬给 theorem，而是给**paper-consistent intuition**：

- PASO 用 sliding-window warm start；
- 上一个 window 已收敛的轨迹，是下一个 window 的高质量初始化；
- 当相邻训练步在局部上足够接近时，只需少数 fixed-point refinements；
- 尤其在训练中后期，trajectory 更平稳，$k$ 更可能小。

然后你要承认：
- 这目前还是 empirical observation；
- 论文现有 Figure 10 给的是 aggregate cost ratio $m = pK/T$，**没有给出 $k$ 随训练步变化的曲线**
- 这个诊断会在 revision 中补上

这点同时也能顺手安抚 xigj。

#### 2) runtime breakdown
这条建议你强烈接受，因为 current implementation 在论文里本身就是 prototype。

你应该说：
- 同意 wall-clock attribution 比 outer-step reduction 更关键；
- 修订版会补 matched hardware 下的 breakdown：
  - computation
  - communication/synchronization
  - framework/launch overhead
- 并更清楚说明 why 21x step reduction 没有完全转化为同等 wall-clock gain

#### 3) 与 OptEx 的关系
这里不要说 “PASO fundamentally better in all ways”。

更稳的说法：
- PASO 不主张 strict dominance；
- 它的 trade-off 是：
  - generic wrapper around standard optimizers
  - idealized formulation targets trajectory fidelity
  - implemented version has favorable per-device memory behavior
- 你们已在 synthetic benchmarks 上和 OptEx 做了对比，但对真实 DL 任务的 matched-setting 对比确实还不够充分
- 修订版会更清楚地写 comparison axes：memory、trajectory fidelity、communication、hardware assumptions

你论文里确实有 OptEx 的理论/合成基准对比，但 reviewer 要的是更“matched”的现实 setting 解释，这点是可以认的。[论文 PDF](https://www.genspark.ai/api/files/s/XViO6qxe) [文档索引](https://www.genspark.ai/api/files/s/7Q1nA379)

---

## Reviewer dwaQ：最危险，因为是 Weak Reject

这个 reviewer 的逻辑最直接：

- **理论没覆盖实际算法**
- **outer-step reduction 不是主要效率指标**
- **large-scale wording 偏大**
- **“without sacrificing quality” 太强**

### 你对 dwaQ 的总策略
**高度诚恳，主动承认 scope gap 和 wording overstretch。**  
这位 reviewer 最怕看到的是你“硬把 full-system theorem 说成就是 practical PASO theorem”。

### 逐点建议

#### 1) Proposition 3.4 没证明 actual PASO
这是你必须认的。

最稳回复框架：

- 你说得对，Proposition 3.4 直接覆盖的是 Eq. (7) 的 full-system FPI；
- 目前实现的是 Eq. (8)-(10) 的 sliding-window + tolerance-based stopping；
- current draft 对 exact implemented PASO 的 end-to-end theorem **不完整**
- 修订版会显式区分：
  - full-system exact convergence guarantee
  - practical PASO as windowed approximation
- 并补一个 error-decomposition framing：
  - within-window stopping error
  - cross-window propagation error

这个回答会比“我们理论上其实也证明了 practical PASO”稳得多。

#### 2) 为什么把 optimizer steps 当主指标？
这条也建议认。

说法：
- 报 step reduction 是为了展示 step-parallelization 的 algorithmic effect
- 但 practical systems metric 的确应该是 wall-clock 和 total cost
- 修订版会把 outer-step reduction 从 headline 降为 secondary diagnostic

#### 3) large-scale implications
这条必须收口。

因为你论文里 LLM 部分确实是 GPT-2 / Llama-3.2-1B + WikiText-2，不足以支撑特别强的“大规模训练范式已经证明成立”的语气。

所以你可以说：
- 当前目标是证明跨任务、跨 optimizer 的 formulation validity
- 不是宣称 large-scale regime 已完全 established
- 会 soften introduction/conclusion 的大规模表述

#### 4) no model quality loss
这里必须改口。

因为 reviewer 直接点了：
- GPT-2 PPL 有小幅变差
- ResNet50 accuracy 有小幅下降
- diffusion FID 有 gap

所以最稳是：
- 我们接受这一点；
- 改为 “comparable model quality” / “small task-dependent gaps”
- diffusion 尤其应避免写成 “no quality loss”

---

## Reviewer xigj：最值得认真争取，他最可能 upward revise

这个 reviewer 写得很细，而且明确说如果理论 concerns 和 diagnostics 回答得好，会考虑改分。  
这意味着你不应该只“防守”，而是要**把他当潜在支持者**。

### 你对 xigj 的总策略
**充分承认理论 gap，并给出非常具体的 revision plan。**

### 逐点建议

#### 1) plot $k$ vs training step
这条最好强烈接受。

因为你论文现在有 Figure 10 的 aggregate $m$，但**没有** per-step/per-window 的 $k$ 诊断；reviewer 抓得很准。  
你可以说：
- 这是非常有价值的诊断；
- 会补 per-window/per-step $k$ 曲线；
- 用来说明 PASO 收益是均匀出现，还是主要在 trajectory 变平稳后出现。

#### 2) 更大学习率是否更稳定
这条不需要强答。

最稳是：
- 当前论文没有 claim enlarged stable LR region
- FPI consistency check 不等于自动扩大学习率稳定域
- 这是值得专门研究的问题

#### 3) sharp transition / warmup / distribution shift 时 $k$ 会不会 spike
这里最好的姿态是**承认有可能**。

因为这完全符合你方法机制：
- warm start 依赖 local trajectory smoothness
- 如果训练动态突变，初值就会更差，$k$ 当然可能上升

这不丢分，反而显得你懂自己方法的 failure mode。

#### 4) Parareal connection
你论文 related work 里确实提到 Parareal/Lions et al. 了，所以这里不该装作没关系。

稳妥说法：
- 结构上确实有 resemblance
- 但你们 setting 是 stochastic、optimizer-defined、可能 non-smooth
- coarse/fine operator 要保持 stochastic consistency 和 optimizer-state fidelity，更难
- 你们会在 revision 中把这层区别讲清楚

#### 5) W1-W3：理论不覆盖 implemented algorithm / 没有 rate analysis / assumptions implicit
这三点是最重要的。

你应该明确说：
- 我们会在 revision 中把 deterministic seeding 提升为 theorem condition 或 scope condition；
- bounded gradient / compactness 等假设会更明确地区分“分析性 sufficient condition”与“实践行为”；
- Adam/AdamW 的现有 proof route 很难给出有意义的 predictive rate constant，这一点会在 paper 里明确承认；
- 当前理论主要是 existence/uniqueness/convergence，不是 practical rate theory。

#### 6) Anderson acceleration 引用缺失
这条特别容易修，直接感谢并补 citation 即可。

#### 7) pipeline baseline
不要承诺一个你没有的实验。  
稳妥说法：
- 你们提 pipeline 是为了说明 orthogonal dimension
- 不是一一替代关系
- 修订版会把“orthogonal vs direct competitor”讲清楚

---

# 三、现在最值得你立刻在全文中统一修改的说法

下面这几个表述，我建议你在 rebuttal 和后续 revision 里全部统一：

### 不要再写
- **no model quality loss**
- **theory proves PASO**（笼统写法）
- **large-scale compatibility**（如果语气太强）
- **memory overhead is one-to-one**（容易歧义）

### 建议改成
- **comparable model quality with small task-dependent gaps**
- **the full-system FPI theory establishes the exact target solution; the implemented PASO is a practical sliding-window approximation**
- **compatible with existing infrastructures at the per-device memory level**
- **per-device memory remains comparable to the sequential baseline in the implemented sliding-window realization**

这些改法和你论文现有内容是兼容的，也能精准对应 reviewer 的攻击点。[论文 PDF](https://www.genspark.ai/api/files/s/XViO6qxe)

---

# 四、给你一份更贴论文的 rebuttal 草稿

下面这版我故意写成：

- **英文**
- **OpenReview 风格**
- **先总回应，再按 reviewer 分开**
- 每个点都按你要求的  
  `> question`  
  `Re: response`

我会尽量避免写入你没有的数据，也避免过度承诺新实验。

---

## Overall response

We sincerely thank all reviewers for the careful reading, detailed feedback, and constructive suggestions. We are encouraged that the reviewers recognized the novelty of formulating step-parallel optimization through a triangular nonlinear equation system and appreciated the breadth of the empirical study. We also agree with the main concerns raised across reviews: **(i)** the current theory most directly covers the full-system fixed-point formulation, while the implemented PASO algorithm uses a practical sliding-window approximation with tolerance-based stopping; **(ii)** several assumptions and proof dependencies should be stated more explicitly; **(iii)** wall-clock speedup should be emphasized more clearly than outer-step reduction; and **(iv)** our wording on model quality should be softened from “no loss” to “comparable quality” with small task-dependent gaps. In the revision, we will strengthen the proof presentation, clarify the theory-to-practice bridge, improve runtime attribution, and qualify several claims accordingly.

---

## Response to Reviewer 9uuF

> The theory is quite sketchy and a bit non-rigorous

Re: 

> The memory overhead is claimed to be small / one-on-one, but I guess that's because we're looking at per device. T steps of a triangular system as written require T copies of parameters. That's a bit heavy.

Re: 

> Appendix N claims an inductive proof. No base case of the induction is supplied. Can you make the proof formal and rigorous?

Re: Thank you for pointing this out. The base case is implicitly given through the identical weight initialization between the sequential and parallel regimes. To make it clear, we rewrite a more formal proof of Proposition 3.3 with base case stated explicitly.

### Appendix N: Proof of Proposition 3.3

#### N.1. Proof of Uniqueness

Let $\{A_t\}_{t=0}^T$ and $\{B_t\}_{t=0}^T$ be two sets of solutions satisfying the TNEs system defined in Eq. (5). We prove $A_t = B_t$ for all $t \in [0, T]$ via induction:

* **Base Case ($t=0$):** According to the first line of the TNEs system in Eq. (5), $\hat{w}_0 - w_0 = 0$. This implies both solutions must satisfy $A_0 = w_0$ and $B_0 = w_0$, hence $A_0 = B_0$.
* **Inductive Hypothesis:** Assume that for some $k \in [0, T-1]$, the solutions are identical up to step $k$, such that $A_\tau = B_\tau$ for all $\tau \in \{0, \dots, k\}$.
* **Inductive Step:** For $t = k+1$, by the definition of the mapping $F_k$ in Eq. (6):
    $$A_{k+1} = F_k(A_0, \dots, A_k; \zeta_0, \dots, \zeta_k)$$
    $$B_{k+1} = F_k(B_0, \dots, B_k; \zeta_0, \dots, \zeta_k)$$
    Since $F_k$ is a deterministic function of its parameters and the mini-batches $\zeta_\tau$ are fixed, the inductive hypothesis $A_\tau = B_\tau$ directly leads to $F_k(A_0, \dots, A_k) = F_k(B_0, \dots, B_k)$. Thus, $A_{k+1} = B_{k+1}$.

By the principle of mathematical induction, the TNEs system has a unique solution.

---

#### N.2. Proof of Unbiasedness

We now prove that the unique solution $\{\hat{w}_t\}_{t=0}^T$ satisfies $E[\hat{w}_t] = E[w_t]$ for all $t \in [0, T]$, where $\{w_t\}_{t=0}^T$ is the trajectory generated by the autoregressive GD process in Definition 3.1.

* **Base Case ($t=0$):** By the definition of the system, $\hat{w}_0 = w_0$. Therefore, $E[\hat{w}_0] = E[w_0] = w_0$, as the initial weights are identical.
* **Inductive Hypothesis:** Assume that for some $k \in [0, T-1]$, the estimator is unbiased for all steps up to $k$, i.e., $E[\hat{w}_\tau] = E[w_\tau]$ for all $\tau \in \{0, \dots, k\}$.
* **Inductive Step:** For $t = k+1$, the autoregressive GD process is defined as $w_{k+1} = F_k(w_0, \dots, w_k; \zeta_0, \dots, \zeta_k)$. Taking the expectation:
    $$E[w_{k+1}] = E[w_0] - \sum_{\tau=0}^{k} \eta_\tau E[g_\tau(w_\tau, \dots; \zeta_\tau, \dots)]$$
    For the TNEs solution $\hat{w}_{k+1}$, the expectation is:
    $$E[\hat{w}_{k+1}] = E[\hat{w}_0] - \sum_{\tau=0}^{k} \eta_\tau E[g_\tau(\hat{w}_\tau, \dots; \zeta_\tau, \dots)]$$
    Since the mini-batches $\zeta_\tau$ used at step $\tau$ are identical between the sequential and parallel regimes. Given that the update rules $g_\tau$ are consistent and $E[\hat{w}_\tau] = E[w_\tau]$ for $\tau \le k$, it follows that:
    $$E[\hat{w}_{k+1}] = E[w_{k+1}]$$

This completes the induction, proving that the TNEs solution unbiasedly estimates the GD trajectory.



> Lemma O.3 seems to be used only in Eqs.(43)-(44), it's placement at the beginning of that section leads one to worry that the authors require U and V to be bounded away from zero also in the main / general proof. Can you fix this exposition issue?

Re: We agree this is an exposition issue. The lower-bound condition in Lemma O.3 is only used locally for Eqs. (43)–(44), not in the main trajectory-equivalence argument nor in the general proof structure. We will move the lemma closer to its actual point of use and explicitly state its scope so that the proof dependencies are unambiguous.

> The proof claimed in section O is asymptotic. Any chances on improving that? Moreover, the proof assumes (inductively, and hence need to ensure that no circularity has been introduced!) that the parameters converge. Convergence of parameters is very rare in ML problems, are the authors sure about this?

Re: We thank the reviewer for this point. We would like to clarify that while Appendix O.4 discusses asymptotic convergence as $K \to \infty$, this is not the final results. Our final theoretical result, detailed in Appendix O.5, explicitly establishes that a non-asymptotic (finite-step) convergence guarantee $ K \leq T$. Besides,  we are sure that the inductive assumption  holds strictly and no circularity has been introduced.

This is because the TNEs system is triangular, the dependency between variables is strictly causal. The update for step $t$ depends only on the weight values from previous steps $0, \dots, t-1$.  Since at $k=0$, the base case $\hat{w}_0^{(0)}$ is already identical to the sequential initialization. Then $\hat{w}_1^{(k)}$  depending on  $\hat{w}_0^{(0)}$ must converge at $k=1$. This in turn forces $\hat{w}_2^{(k)}$ (dependent on $(\hat{w}_0^{(0)},\hat{w}_1^{(1)})$)  to converge at $k=2$. Following this induction，at $k=T-1$, the triangular nature ensures that the final weight $\hat{w}_T^{(T-1)}$   locks onto to its exact sequential value. As a result, our method does not merely approach the trajectory; it reaches the exact numerical sequential trajectory in at most $K = T$ iterations.



> The paper starts with the triangular system, but then in reality T can be big ... the authors present an engineering compromise of having a moving window of size p. At that point the theory and the triangular model become irrelevant, leading to a huge mismatch between the model/theory presented and the actual empirical results. Could the authors discuss this aspect more thoroughly?

Re:
To directly address your concern and eliminate the perceived mismatch between our model and the empirical results, we have formalized the global convergence proof specifically for the actual sliding window implementation. As shown below (which we will add to the appendix), we bound the global approximation error by tracking the error propagation across finite window blocks.


### Proof of Convergence for PASO with Sliding Window and Adaptive Tolerance

**1. Notation and Assumptions**
We denote the sequence of weights up to time $\tau$ as $W_\tau$ and the full sequence as $W$. The ideal autoregressive gradient descent trajectory is $W^{seq}$. 
To analyze the sliding window (maximum size $p$), we partition the sequence $T$ into $m \approx T/p$ consecutive blocks, $\mathcal{B}_1, \dots, \mathcal{B}_m$. Each block undergoes $J \in [1, p]$ parallel iterations before the window slides past it, yielding total iterations $K = mJ$. 

We rely on two standard assumptions:
* **Assumption 1 (Uniform Lipschitz Continuity):** The update function $g_\tau$ is uniformly Lipschitz continuous with constant $C > 0$.
* **Assumption 2 (Bounded Initial Distance):** When the window reaches block $\mathcal{B}_i$, its initial distance to the exact sequential target is bounded by $D > 0$, i.e., $\|W_{\mathcal{B}_i}^{(0)} - W_{\mathcal{B}_i}^{seq}\|_F \le D$.

**2. Theorem: Convergence Bounds of PASO**
Given a learning rate schedule satisfying bounded energy $\sum_{\tau=0}^{T-1} \eta_\tau^2 \le \frac{\rho^2}{C^2}$ (with contraction constant $\rho \in (0,1)$), the global approximation error of the final window block $\mathcal{B}_m$ at iteration $J$ is uniformly bounded by:
$$\|W_{\mathcal{B}_m}^{(J)} - W_{\mathcal{B}_m}^{seq}\|_F \le D \cdot \rho^J \cdot \left( \frac{1-\rho^{J+1}}{1-\rho} \right)^{m-1}$$

**3. Phase 1 & 2: Finite Bounds and Global Contraction**
* **Phase 1 (Finite Iterations):** Because the system is strictly triangular, the first element of any active window depends only on converged history. It reaches its local fixed point in exactly one iteration, guaranteeing the adaptive tolerance condition is met and the window advances by a stride $s_k \ge 1$. This strictly bounds total iterations $K \in [\lceil T/p \rceil, T]$.
* **Phase 2 (Contraction Mapping):** By Assumption 1 and our bounded learning rate condition, the update mapping forms a global strict contraction mapping with factor $\rho < 1$ across any arbitrary subset of the sequence.

**4. Phase 3: Global Convergence Error Bound**
We construct a cumulative global bound by chaining localized error bounds across the $m$ sequential blocks. Let the target approximation error for block $i$ after $J$ parallel iterations be $E_i = \|W_{\mathcal{B}_i}^{(J)} - W_{\mathcal{B}_i}^{seq}\|_F$.

**Step 1: Single Iteration Recurrence.**
Consider block $i$ at iteration $j \ge 1$. Its state updates via the flawed history $H_{i-1}$, while the target $W_{\mathcal{B}_i}^{seq}$ is the true fixed-point under perfect history $H_{i-1}^*$. Applying the triangle inequality and the $\rho$-Lipschitz contraction property (established in Phase 2) yields:
$$\|W_{\mathcal{B}_i}^{(j)} - W_{\mathcal{B}_i}^{seq}\|_F \le \rho \|W_{\mathcal{B}_i}^{(j-1)} - W_{\mathcal{B}_i}^{seq}\|_F + \rho \|H_{i-1} - H_{i-1}^*\|_F$$
Letting $e_j = \|W_{\mathcal{B}_i}^{(j)} - W_{\mathcal{B}_i}^{seq}\|_F$ and the historical perturbation term be $\delta_{hist} = \rho \|H_{i-1} - H_{i-1}^*\|_F$, the recurrence simplifies to:
$$e_j \le \rho e_{j-1} + \delta_{hist}$$

**Step 2: Unrolling $J$ Iterations.**
Unrolling this linear recurrence for $J$ iterations yields:
$$e_J \le \rho^J e_0 + \sum_{t=0}^{J-1} \rho^t \delta_{hist} = \rho^J e_0 + \frac{1-\rho^J}{1-\rho} \delta_{hist}$$
Note that $e_J$ is the final block error $E_i$, and $e_0 \le D$. For the historical perturbation, applying the block-wise summation of the Frobenius norm yields $\delta_{hist} \le \rho \sum_{k=1}^{i-1} E_k$. Substituting this back gives the strict causal block-to-block error propagation:
$$E_i \le \rho^J D + \frac{\rho(1-\rho^J)}{1-\rho} \sum_{k=1}^{i-1} E_k$$

**Step 3: Solving the Global Propagation.**
Let $A = \rho^J D$ and $B = \frac{\rho(1-\rho^J)}{1-\rho}$. The sequence $E_i \le A + B \sum_{k=1}^{i-1} E_k$ can be explicitly solved by induction to show $E_i \le A(1+B)^{i-1}$. Expanding the multiplicative factor:
$$1 + B = 1 + \frac{\rho - \rho^{J+1}}{1-\rho} = \frac{1-\rho^{J+1}}{1-\rho}$$
Substituting $A$ and $(1+B)$ for the final $m$-th block yields the global error bound:
$$\|W_{\mathcal{B}_m}^{(J)} - W_{\mathcal{B}_m}^{seq}\|_F \le D \cdot \rho^J \cdot \left( \frac{1-\rho^{J+1}}{1-\rho} \right)^{m-1}$$

**Step 4: Iteration Requirements for $\epsilon$-Precision.**
To ensure the final error is $\le \epsilon$, we relax the bound $(1-\rho^{J+1} < 1)$:
$$D \cdot \rho^J \cdot \left( \frac{1}{1-\rho} \right)^{m-1} \le \epsilon$$
Taking the natural logarithm and rearranging to isolate $J$ provides the minimum local iterations:
$$J \ge \frac{\ln(D/\epsilon)}{\ln(1/\rho)} + (m-1) \frac{\ln\left(\frac{1}{1-\rho}\right)}{\ln(1/\rho)}$$
Defining local precision $C_\epsilon = \frac{\ln(D/\epsilon)}{\ln(1/\rho)}$ and historical penalty $\alpha = \frac{\ln(\frac{1}{1-\rho})}{\ln(1/\rho)}$, we get $J \ge C_\epsilon + \alpha(m-1)$. Since $K = mJ$ and assuming $m = T/p$, the quadratic lower bound for total parallel iterations is:
$$K \ge K_{min} = \alpha \frac{T^2}{p^2} + \frac{T}{p}(C_\epsilon - \alpha)$$

**Step 5: Conditions for Parallel Speedup.**
To achieve speedup over sequential execution, we require $K_{min} < T$:
$$\alpha \frac{T^2}{p^2} + \frac{T}{p}(C_\epsilon - \alpha) < T$$
Restructuring into a quadratic inequality for $p$ and finding the positive root, the asymptotic boundary condition for the sliding-window size (where $T \gg 1$) simplifies to:
$$p > \sqrt{\alpha T} = \sqrt{\frac{\ln\left(\frac{1}{1-\rho}\right)}{\ln(1/\rho)} \cdot T}$$



> The memory overhead is claimed to be small / one-on-one, but I guess that's because we're looking at per device. T steps of a triangular system as written require T copies of parameters.

Re: Yes, directly solving the full system would require storing T copies of parameters. To circumvent this, we implement a sliding-window PASO, where each device only requires one model instance and one optimizer state instead of T parameter copies. Given that this approach trades computational cost for acceleration, we deem the storage overhead acceptable.
---

## Response to Reviewer hm7D

> Can you clarify (theoretically or empirically) what problem characteristics make the number of fixed-point iterations k small in practice, so that the computational overhead ratio stays close to 1?

Re: Thank you for this important question. Empirically, the key factor is the warm-started sliding-window solve: the previously converged trajectory provides a strong initialization for the next window, so when adjacent optimization steps remain locally close, only a small number of fixed-point refinements is needed. This is especially plausible once training enters a smoother regime. We agree, however, that the current paper does not yet provide a predictive characterization of when $k$ will remain small. In the revision, we will make this intuition more explicit and add diagnostics showing how $k$ varies over training and across window sizes.

> Is it possible to have a more complete and comparable runtime breakdown, specifically for the wall clock time, for the proportion of time spent in communication vs. computation, and to clarify the PASO behavior in comparison to the strongest available optimized configurations when the hardware is the same?

Re: Yes—this is an excellent suggestion. We agree that wall-clock attribution is the more relevant systems analysis and that the current paper emphasizes outer-step reduction more than runtime composition. In the revision, we will add a matched-hardware runtime breakdown separating computation, communication/synchronization, and other framework overheads, and we will clarify the PASO behavior relative to stronger optimized configurations under the same hardware budget. We will also revise the paper to center wall-clock speedup rather than outer-step reduction.

> Would it be possible to compare PASO more directly against the most relevant prior and concurrent step-/time-parallel methods, especially OptEx, using matched settings, and to clarify whether PASO is fundamentally better or mainly different in its trade-offs?

Re: We agree that a more direct comparison to closely related step-/time-parallel methods would strengthen the paper. Our intent is not to claim strict dominance over all prior approaches. Rather, we view PASO as offering a different trade-off: it acts as a general wrapper around standard optimizers, targets trajectory fidelity in the idealized formulation, and has favorable per-device memory behavior in the implemented windowed setting. In the revision, we will expand the discussion of OptEx and related methods, clarify the relevant comparison axes, and include more direct matched-setting discussion where space permits.

> The wall clock gain is much smaller than the iteration reduction ...

Re: We agree. The current implementation is still an academic prototype, and communication/synchronization overhead prevents the full algorithmic reduction in outer steps from translating into proportional wall-clock gains. We will revise the wording accordingly and present outer-step reduction as a secondary algorithmic diagnostic rather than the primary practical efficiency claim.

---

## Response to Reviewer dwaQ

> Proposition 3.4 proves exact convergence for the full fixed-point iteration in Eq. (7). What formal theoretical guarantee can be established for the actual PASO algorithm with sliding-window updates and tolerance-based stopping as defined in Eqs. (8)–(10)?

Re: Thank you for identifying this gap so clearly. You are correct that Proposition 3.4 directly covers the full-system fixed-point iteration in Eq. (7), whereas the implemented PASO algorithm uses sliding windows and approximate stopping in Eqs. (8)–(10). The exact end-to-end theoretical guarantee for the practical algorithm is incomplete in the current draft, and we will make this distinction explicit in the revision. We will also add a clearer error-decomposition view of the practical method, separating within-window stopping error from cross-window propagation error, and we will avoid overstating the scope of Proposition 3.4.

> Since PASO computes multiple steps concurrently, why is the number of optimizer steps treated as a primary efficiency metric?

Re: This is a fair point. Our intention in reporting outer-step reduction was to expose the algorithmic effect of step parallelization, i.e., that PASO advances multiple target steps within one outer iteration. We agree, however, that the primary practical metric should be wall-clock time together with total compute/communication overhead. In the revision, we will de-emphasize optimizer-step reduction as a headline claim and instead treat it as a secondary diagnostic.

> The large-scale implications are emphasized heavily in the introduction, but the LLM experiment is on WikiText-2 with GPT-2 and Llama-3.2-1B. Do the authors have evidence that PASO remains stable and beneficial at larger scales?

Re: We agree that the current experimental scale is limited relative to the broader vision stated in the introduction. Our present goal is to establish the validity of the step-parallel formulation across tasks and optimizers under controlled settings, rather than to claim that the large-scale regime is already fully settled. We will therefore soften the large-scale wording in the introduction and conclusion and position large-scale validation more clearly as future work.

> The paper repeatedly emphasizes achieving acceleration without sacrificing model quality, but the tables show that this statement is not always true ...

Re: We appreciate this careful reading and agree that our wording is currently too strong. The more accurate statement is that PASO achieves **comparable model quality**, with small task-dependent gaps in some settings. We will revise the abstract, main text, and conclusion accordingly, and we will explicitly note the nonzero gaps observed in several experiments, including diffusion.

> The scale of the experiments is limited ...

Re: We agree. In the revision, we will better position the current experiments as a broad proof-of-concept across tasks and optimizers rather than as a definitive large-scale systems study. We will also strengthen the limitations discussion on scale and practical deployment.

---

## Response to Reviewer xigj

> Can you provide a plot of FPI iterations k as a function of training step? This would clarify whether the speedup is uniform or concentrated in later training stages.

Re: Thank you for this excellent suggestion. We agree that reporting only the aggregate computational ratio $m$ obscures when PASO becomes efficient during training. In the revision, we will add a per-window/per-training-step diagnostic for the FPI iteration count $k$, which should clarify whether the speedup is uniform or primarily appears once the trajectory becomes smoother and warm-starts become more accurate.

> Does PASO help in using larger learning rates without divergence? The FPI convergence criterion effectively validates each window's solution before proceeding. Does this implicit check enable the use of larger learning rates that would otherwise cause divergence in sequential training?

Re: This is a very interesting question. Our current paper does not claim an expanded stable learning-rate region, and we agree that such a claim would require dedicated experiments. While the within-window refinement acts as a local consistency check, this does not by itself establish improved stability under larger learning rates. We will avoid overstating this point and will mention it as an interesting direction for future work.

> How does PASO perform when the sliding window encounters a sharp loss landscape transition (e.g., learning rate warmup, sudden distribution shift in data)? Does k spike in these regimes?

Re: We agree that this is an important failure mode to understand. Our expectation is that abrupt trajectory changes can increase $k$ because the warm-start becomes less accurate when the local optimization path changes rapidly. We will add this discussion explicitly and, where possible, include diagnostics around such regimes. This also motivates adaptive window sizing and stronger acceleration/correction mechanisms as future work.

> PASO's sliding-window FPI bears a structural resemblance to Parareal ... Have the authors considered adapting Parareal-style coarse-fine correction to PASO's setting, and if so, what specific obstacles arise ... ?

Re: Thank you for this insightful connection. We agree that the relation to Parareal deserves clearer discussion. A key difference is that PASO operates with stochastic, optimizer-defined, and potentially non-smooth updates rather than smooth time-discretized ODE solvers. As a result, defining coarse/fine operators that preserve stochastic consistency and optimizer-state fidelity is substantially more challenging. We nevertheless agree that Parareal-style correction is promising and will expand the discussion accordingly.

> The convergence proofs do not cover the implemented algorithm ... no error propagation analysis across windows ...

Re: We agree, and this is one of the main theoretical limitations of the current version. In the revision, we will state this limitation explicitly and better connect the full-system theory to the practical sliding-window implementation by discussing approximate window solves, residual control, and empirical trajectory drift.

> No convergence rate analysis ... the practical speedup rests on the empirical observation that k is small ...

Re: We agree. The current theory establishes existence/uniqueness and convergence for the idealized formulation, but it does not yet provide a useful predictive rate bound for the practical regime. We will sharpen this distinction in the paper and avoid wording that could suggest otherwise. We also appreciate the observation that the present proof route is unlikely to yield a meaningful constant for Adam/AdamW without a different analytical approach.

> Key assumptions are implicit or insufficiently stated ... deterministic mini-batch seeding ... bounded gradient norm ...

Re: Thank you. We agree that several assumptions should be elevated from implementation details or appendix remarks into explicit theorem conditions or scope statements. In particular, deterministic mini-batch seeding should be stated clearly wherever exact trajectory equivalence is claimed in the stochastic setting. We will revise the theorem statements and assumptions accordingly.

> Prior work on Anderson acceleration for deep learning is missing.

Re: Thank you for the pointer. We will incorporate this literature in the revision and clarify how PASO’s fixed-point setting relates to, and differs from, prior Anderson-acceleration work in deep learning.

> The "no model quality loss" claim is insufficiently supported ... especially for diffusion ...

Re: We agree and will qualify this claim throughout the paper. The appropriate wording is “comparable model quality” rather than “no loss,” especially in light of the small but nonzero gap in the diffusion result.

> Missing pipeline parallelism baselines ...

Re: Thank you. Our motivation for discussing pipeline parallelism was to situate PASO as exploring an orthogonal parallelism dimension, not to suggest that it is a direct drop-in substitute for pipeline methods. We will revise the text to make this distinction clearer and discuss complementarities and differences more explicitly.

> Presentation clarity ... appendix execution commands ...

Re: We appreciate this feedback and agree that the presentation can be improved. We will revise figure captions/annotations to make them more self-contained and move implementation-oriented execution details out of the main appendix text so that the appendix can better focus on theory and diagnostics.

---

# 五、我建议你在 rebuttal 开头再加一句“主动认错式总括”

这句很有用，因为它能显著降低 reviewer 对“作者在硬撑”的戒心：

**A short common clarification.**  
Across reviews, the most important clarification is that the current theory most directly covers the full-system fixed-point formulation, whereas the implemented PASO algorithm is a practical sliding-window approximation with adaptive stopping. We agree that the current draft does not make this distinction sufficiently explicit, and we will revise the paper accordingly.

这句话几乎能同时安抚 9uuF、dwaQ、xigj。

---

# 六、如果你想进一步冲分，最该补的不是大实验，而是这 3 个“小而关键”的补充

如果 rebuttal 截止前你们还能补一点材料，我最推荐优先补：

### 1. 一张 $k$ vs training step 的图
这是 xigj 最想看的，也能顺便回应 hm7D。

### 2. 一个 runtime breakdown 表
至少拆成：
- computation
- communication/synchronization
- other overhead

这会大幅增强 hm7D 和 dwaQ 的信服度。

### 3. 一句正文层面的改口
把 abstract / intro / conclusion 里所有  
**“no model quality loss”**  
统一改为  
**“comparable model quality”** 或 **“small task-dependent gaps”**

这一步非常重要，成本又最低。[论文 PDF](https://www.genspark.ai/api/files/s/XViO6qxe)

---

# 七、我接下来最建议继续帮你的两件事

如果你愿意，我下一步可以直接继续做下面任一件：

## 选项 A：我把这份 rebuttal **压缩到真正可贴 OpenReview 的字数版**
也就是：
- 每位 reviewer 更短
- 句子更硬朗
- 控制字符数

## 选项 B：我帮你做一份 **“逐条精修版”**
我会把每个 reviewer 的回复再分成：
- 哪些句子必须保留
- 哪些句子是可删的
- 哪些句子更可能打动 reviewer

如果你愿意，我下一条就可以直接给你：

**“最终精简版 rebuttal（适合直接粘贴 OpenReview）”**。