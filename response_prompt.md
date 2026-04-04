你是一位专业的学术写作助手和经验丰富的研究员，精通AI和大模型训练相关领域。你的任务是协助我起草一份论文ICML顶会评审意见的回复稿（Rebuttal/Response）。

核心任务:

根据我提供的论文摘要、主要贡献以及审稿人的意见，为每一位审稿人生成一份专业、谦逊且具有建设性的回复。回复需要清晰地回应审稿人的每一个问题，澄清误解，并说明将如何根据其建议修改论文。



**回复的总体策略与语气:**

1. **态度:** 始终保持礼貌、专业和感激。首先感谢审稿人付出的时间和提出的宝贵意见。
2. **结构:**
   - 首先，生成一个对所有审稿人的总体感谢段落。
   - 然后，针对每一位审稿人进行单独回复。
   - 在回复每个审稿人时，采用“逐点回应”的方式。清晰地引用审稿人的问题，然后给出你的回复。
3. **内容:**
   - **对于积极评价:** 表示感谢。
   - **对于误解:** 礼貌地澄清，并引用论文中的证据（如“在论文的第X页/第Y节中，我们其实已经讨论了...”）。同时反思是否是论文的表述不够清晰，并承诺会“在修改版中，我们将在第Y节中进一步强调/澄清这一点，以避免歧义”。
   - **对于建设性意见/指出的不足:**
     - 如果同意，请真诚地接受并说明你将如何修改论文（例如，“这是一个非常有价值的建议。我们将在最终版本中加入对...的讨论/补充...实验。”）。
     - **如果需要补充实验:** 如果审稿人要求补充实验来验证某个观点，而你无法立即完成，你可以**生成一些符合逻辑预期的、合理的模拟实验数据**，并将其以Markdown表格的形式呈现。同时说明，“感谢您的建议，为了回应您的关切，我们进行了一项初步的模拟/补充实验，结果如下表所示。这些结果进一步支持了我们的结论。我们将在最终版本中加入完整的实验细节。”
     - 如果不同意，请提供充分、严谨的论据和证据来礼貌地为自己的观点辩护。避免直接说“你是错的”，可以说“我们非常理解您的顾虑，但我们想从另一个角度解释我们的设计/选择...”。

**严格遵守以下 OpenReview 格式化要求 (最重要!):**

1. **字符数限制:** 每个审稿人的回复内容（不包括引用的问题）严格限制在 **xxxx 字符**以内。请务必精简语言。

2. **回复模板:** 严格按照以下格式组织每个回复点：

   > [这里是引用的审稿人问题]
   >
   > Re: [这里是你的具体回复]

3. **文本格式:**

   - 加粗、斜体、列表等文本格式**必须**使用 Markdown 语法（例如 `**加粗**`, `*斜体*`, `- 列表项`）。
   - **绝对不能**使用 TeX 的文本宏，如 `\emph` 或 `\begin{enumerate}`。

4. **数学公式:**

   - 所有数学公式和符号必须使用 MathJax 渲染。
   - 行内公式使用 `$...$` 包裹，块级公式使用 `$$...$$` 包裹。
   - **关键规则:** 在 Markdown 环境中，所有 TeX 的反斜杠 `\` **必须转义**为 `\\`。
     - **错误示例:** `$\sum_{i=1}^{n}$`
     - **正确示例:** `$y = \\sum_{i=1}^{n} x_i$`
     - **正确示例:** `$$L(x, y) = \\|x - y\\|_2^2$$`

5. **特殊字符:**

   - 当在公式或文本中使用 `<` 或 `>` 符号时，请确保它们前后有空格，以避免被误认为HTML标签。例如，写 `$a < b$` 而不是 `$a<b>$`。

6. **表格:** 如果需要生成补充实验数据的表格，请使用标准的 Markdown 表格语法。

------


**背景信息:**

- **我的论文pdf:** 请看我上传的pdf

- **审稿人意见原文:**

  ```
    Official Review of Submission1615 by Reviewer 9uuF
    Official Reviewby Reviewer 9uuF12 Mar 2026, 18:41 (modified: 24 Mar 2026, 20:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 9uuFRevisions
    Summary:
    SUMMARY
    The paper looks at the GD family of iterations written out in an unrolled manner. Then, it suggests viewing these unrolled iterations as a triangular system, which the authors then propose to approximately "solve" in parallel. The paper claims strong theoretical and empirical results justifying the modeling choice.

    NOTE: In the reviewer's opinion, this paper is more of an ML systems paper (please see note in the review between the "mismatch")

    Strengths And Weaknesses:
    The empirical results look detailed and good
    The idea of exploiting the triangular system in parallel seems to be novel (to my knowledge)
    Weakenesses

    The theory is quite sketchy and a bit non-rigorous
    The memory overhead is claimed to be small / one-on-one, but I guess that's because we're looking at per device. T steps of a triangular system as written require T copies of parameters. That's a bit heavy.
    Soundness: 2: fair
    Presentation: 2: fair
    Significance: 3: good
    Originality: 3: good
    Key Questions For Authors:
    Appendix N claims an inductive proof. No base case of the induction is supplied. Can you make the proof formal and rigorous?

    Lemma O.3 seems to be used only in (43)-(44), it's placement at the beginning of that section leads one to worry that the authors require U and V to be bounded away from zero also in the main / general proof. Can you fix this exposition issue?

    The proof claimed in section O is asymptotic. Any chances on improving that? Moreover, the proof assumes (inductively, and hence need to ensure that no circularity has been introduced!) that the parameters converge. Convergence of parameters is very rare in ML problems, are the authors sure about this?

    The paper starts with the triangular system, but then in reality T can be big (and when T corresponds to stochastic iterations as opposed to epochs, then it can be gigantic), and the authors present an engineering compromise of having a moving window of size 'p'. At that point the theory and the triangular model become irrelevant, leading to a huge mismatch between the model/theory presented and the actual empirical results. While such mismatches are common across ML papers (unfortunate, in my opinion), could the authors discuss this aspect more thoroughly?

    Limitations:
    Limited theory, not yet fully vetted
    At best asymptotic analysis
    Potentially heavier memory footprint and its limitations not discussed
    Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
    Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
    Compliance With LLM Reviewing Policy: Affirmed.
    Code Of Conduct Acknowledgement: Affirmed.
    Official Review of Submission1615 by Reviewer hm7D
    Official Reviewby Reviewer hm7D12 Mar 2026, 12:47 (modified: 24 Mar 2026, 20:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer hm7DRevisions
    Summary:
    The paper proposes PASO, a new training framework that parallelizes optimization across training steps, rather than only within a single step as in data, model, or pipeline parallelism. The key idea is to reinterpret the usual autoregressive update sequence of optimizers like SGD, Adam, and AdamW as a system of triangular nonlinear equations whose unknowns are the parameters at different training steps. By solving this system in parallel, PASO aims to preserve the same optimization trajectory as the original sequential optimizer while exposing a new axis of parallelism.

    The paper has two main contributions. 1. It develops the theoretical view that the sequential gradient-descent trajectory corresponds to the unique solution of this triangular system, so the step-parallel problem is well-defined. 2. It introduces a practical algorithm that solves only a window of consecutive steps at a time, using a sliding-window fixed-point procedure with an adaptive stopping rule, so the method can operate under limited hardware rather than requiring parallelization over the full horizon. This makes PASO a general wrapper that can be combined with standard optimizers instead of replacing them with a new update rule.

    Strengths And Weaknesses:
    Strengths:

    This paper uses a novel idea of solving the triangular nonlinear equation (TNE) system to reframe sequential stochastic optimization, which enables parallelism across training steps.
    On the theory parts, the paper proves that the TNE solution uniquely matches the sequential GD trajectory and that the fixed-point iteration converges to that trajectory.
    On the empirical side, the paper reports large reductions in optimization steps and moderate wall-clock speedups while maintaining similar model quality.
    Weaknesses:

    The wall clock gain is much smaller than the iteration reduction for the runtime, which varies slightly depending on the configuration.They have PASO reducing the number of optimization steps in the range 
    , however, the wall clock speedup is in the range 
    . This implies that the current gains are capped by resource utilization in their implementation or the cost of communication.
    The theory mainly establishes trajectory equivalence and convergence, but not a predictive characterization of when PASO will deliver large speedups. In particular, the appendix gives a worst-case bound of 
    , which is compatible with essentially sequential behavior, while the efficiency analysis assumes 
    based on empirical observation rather than theory. As a result, the strongest practical claims appear to depend more on experiments than on a theory explaining the favorable regimes.
    Soundness: 2: fair
    Presentation: 2: fair
    Significance: 2: fair
    Originality: 3: good
    Key Questions For Authors:
    Can you clarify (theoretically or empirically) what problem characteristics make the number of fixed-point iterations 
    small in practice, so that the computational overhead ratio 
    stays close to 
    ?
    Is it possible to have a more complete and comparable runtime breakdown, specifically for the wall clock time, for the proportion of time spent in communication vs. computation, and to clarify the PASO behavior in comparison to the strongest available optimized configurations when the hardware is the same?
    Would it be possible to compare PASO more directly against the most relevant prior and concurrent step-/time-parallel methods, especially OptEx, using matched settings, and to clarify whether PASO is fundamentally better or mainly different in its trade-offs?
    Limitations:
    Yes.

    Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
    Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    Compliance With LLM Reviewing Policy: Affirmed.
    Code Of Conduct Acknowledgement: Affirmed.
    Official Review of Submission1615 by Reviewer dwaQ
    Official Reviewby Reviewer dwaQ08 Mar 2026, 14:40 (modified: 24 Mar 2026, 20:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer dwaQRevisions
    Summary:
    The work studies whether the autoregressive dependency across GD steps can be broken without changing the underlying optimization trajectory. The authors claim to explore a fundamental challenge by recasting sequential GD-style optimization as a system of triangular nonlinear equations (TNEs), and then solving these equations with fixed-point iteration to enable step-parallel training. Within this generic framework, the authors establish that the TNE system has a unique solution corresponding to the original GD trajectory and that fixed-point iteration recovers that trajectory in at most the equal iterations. The proposed PASO optimizer achieves up to 21x fewer optimizer steps and 4.5× wall-clock speedup on language modeling and diffusion tasks, with no model quality loss.

    Strengths And Weaknesses:
    Strengths

    The paper is clearly written and well structured.

    Recasting autoregressive GD as a triangular nonlinear system is conceptually novel and provides a clean perspective for understanding inter-step dependencies.

    The experiments section covers a broad range of tasks. The authors evaluate on GPT-2 and Llama-3.2-1B over WikiText-2, CIFAR-10 image classification with ResNet50 and ViT, and diffusion training on LSUN Church.

    Weaknesses

    The main formal guarantee is for the full fixed-point system in Eq. (7), but the proposed PASO algorithm actually uses a sliding-window approximation and an adaptive stopping rule in Eqs. (8)-(10). I did not find the theoretical guarantee for the exact PASO method.

    The paper repeatedly emphasizes achieving acceleration without sacrificing model quality, but the tables show that this statement is not always true. On GPT-2, Adam degrades from 20.4 to 20.7 PPL and AdamW from 20.4 to 20.6; on CIFAR-10 ResNet50, Adam degrades from 90.1 to 89.5 and AdamW from 90.3 to 89.8; on diffusion, FID degrades from 9.08 to 9.55 even though training is faster. The paper’s wording is stronger than the experimental results.

    It seems that PASO computes multiple gradients concurrently per outer iteration. The paper acknowledges that the more relevant gain is the wall-clock speedup (3-4.5x), and Table 1 already analyzes PASO in terms of total gradient computation 
    , not simply reduced iteration count. In other words, the major empirical contribution is the runtime speedup, not a 21x reduction in GD steps.

    The scale of the experiments is limited. The LLM result is on WikiText-2 with GPT-2 and Llama-3.2-1B, rather than on a more larger scale setup. Similarly, the classification result is only on CIFAR-10, and the strongest diffusion claim still involves a FID gap. Since this work argues a new training paradigm, it would be better to include some larger-scale experiments.

    Soundness: 2: fair
    Presentation: 3: good
    Significance: 2: fair
    Originality: 2: fair
    Key Questions For Authors:
    Proposition 3.4 proves exact convergence for the full fixed-point iteration in Eq. (7). What formal theoretical guarantee can be established for the actual PASO algorithm with sliding-window updates and tolerance-based stopping as defined in Eqs. (8)–(10)?

    Since PASO computes multiple steps concurrently, why is the number of optimizer steps treated as a primary efficiency metric?

    The large-scale implications are emphasized heavily in the introduction, but the LLM experiment is on WikiText-2 with GPT-2 and Llama-3.2-1B. Do the authors have evidence that PASO remains stable and beneficial at larger scales?

    Limitations:
    The limitations are discussed in the weaknesses and questions sections above. There is no potential negative societal impact of this work.

    Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
    Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    Compliance With LLM Reviewing Policy: Affirmed.
    Code Of Conduct Acknowledgement: Affirmed.
    Official Review of Submission1615 by Reviewer xigj
    Official Reviewby Reviewer xigj02 Mar 2026, 12:14 (modified: 24 Mar 2026, 20:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer xigjRevisions
    Summary:
    This paper tackles a fundamental bottleneck in deep learning optimization: the sequential dependency of gradient descent steps; and proposes a principled mathematical framework for an under-explored parallelism dimension. Specifically, PASO recasts the autoregressive nature of gradient descent (SGD, Adam, AdamW) as a system of triangular nonlinear equations (TNEs). By treating the weight vectors across 
    time steps as unknowns and solving the system via fixed-point iteration (FPI), PASO enables parallel computation of gradient updates across steps within a sliding window. The paper provides convergence proofs for the full-system FPI, demonstrates trajectory equivalence to sequential training via deterministic seeding, and evaluates on language modeling (GPT-2, Llama-3.2-1B), image classification (CIFAR-10), and diffusion tasks (LSUN Church), reporting wall-clock speedups of 3.0 to 4.5
    with comparable model quality. The direction is promising, though the current version would benefit from stronger theoretical foundations and additional diagnostics.

    Strengths And Weaknesses:
    Strengths
    Creative problem formulation. The idea of parallelizing across GD steps by reformulating the training trajectory as a system of triangular nonlinear equations is genuinely interesting and opens a research direction that is distinct from data parallelism, model parallelism, and pipeline parallelism. The TNE perspective provides a clean mathematical framework for thinking about step-level parallelism.

    Experimental coverage. PASO is evaluated across three distinct domains (language modeling, image classification, diffusion), multiple optimizers (SGD, Adam, AdamW), and multiple architectures (GPT-2, Llama-3.2-1B, CNN, ViT, ResNet50, UNet). This breadth strengthens the claim of generality. The deterministic seeding mechanism (Appendix A.2) is a clean engineering solution for maintaining trajectory equivalence across parallel workers.

    Wall-clock speedups demonstrated. The empirical results show consistent wall-clock speedups (3.0-4.5 x) across tasks, with model quality metrics that are comparable to sequential baselines. The trajectory divergence analysis (Table 11) provides additional evidence that PASO closely tracks the sequential optimization path.

    Honest limitations discussion. Section 5.3 is commendably candid about the gap between step-level speedup (6-21x) and actual runtime speedup (3-4.5 x), attributing it to engineering constraints. The authors clearly identify Anderson acceleration as a promising future direction.

    Memory parity with sequential training. Per-device memory remains constant regardless of window size 
    , matching the sequential baseline at 1 model + 1 optimizer (Table 1, Tables 2-3). This is a practical advantage over methods like OptEx that require storing gradient histories.

    Weaknesses
    The convergence proofs do not cover the implemented algorithm. The convergence analysis (Appendix O) addresses the full-system FPI where all 
    components are solved simultaneously. This is a reasonable theoretical starting point. However, the implemented algorithm uses a sliding window of size 
    , where each window's initial condition depends on the previous window's output. No proof is provided that this sequential-windows-of-parallel-FPI converges to the same trajectory as sequential GD. Crucially, there is no error propagation analysis across windows: when convergence within each window is only approximate (tolerance-based), does the accumulated error drift over many windows? Any effort to bridge this gap between the full-system theory and the sliding-window implementation would substantially strengthen the contribution.

    No convergence rate analysis. The convergence proof establishes that FPI converges to the correct solution but provides only a worst-case bound of 
    iterations, which yields zero speedup over sequential GD. The entire practical speedup rests on the empirical observation that 
    with good initial guesses (the last converged weight from the sliding window, Algorithm 1 Line 9), but this is never theoretically bounded. Since this is a fixed-point problem, a contraction factor analysis (i.e., establishing that the contraction coefficient is strictly less than 1) or a bound on 
    as a function of the initialization quality and problem parameters would substantially strengthen the theoretical contribution. Without such analysis, the behavior in adversarial regimes is unclear: for instance, when SGD steps exhibit large oscillations near a basin of attraction, the FPI might hop around without converging even if the true solution exists nearby. Furthermore, the Lipschitz bound for Adam's update function (Appendix O.7, Eq. 47) involves terms of order 
    , which with standard 
    produces astronomically large constants, suggesting the existing proof framework cannot yield a meaningful rate bound for Adam without a different analytical approach.

    Key assumptions are implicit or insufficiently stated. The unbiasedness result (Proposition 3.3) silently depends on deterministic mini-batch seeding (
    ), which is only described in Appendix A.2. This should be a formal condition in the theorem statement. The bounded gradient norm assumption (Assumption O.2) implies the entire optimization trajectory lies in a compact set, which is a strong condition that may fail during early training. Additionally, the theorems do not exclude degenerate cases (e.g., if 
    and 
    , the entire trajectory trivially stays at zero, since zero is a fixed point too). Appropriate conditions on the loss landscape and initialization should be stated explicitly.

    Prior work on Anderson acceleration for deep learning is missing. The paper mentions Anderson acceleration as future work (Section 5.3) but does not cite existing applications of Anderson acceleration to deep learning, such as: Lupo Pasini et al., "Stable Anderson Acceleration for Deep Learning,". Engaging with this existing literature would better contextualize the proposed future directions and clarify what additional challenges PASO's FPI setting introduces beyond what has already been studied.

    The "no model quality loss" claim is insufficiently supported for all tasks. For language modeling (Table 2) and image classification (Table 3), model quality is indeed comparable. However, for diffusion (Table 4), PASO achieves FID 9.55 versus the Adam baseline's 9.08, a gap of 0.47 from a single run with no error bars or confidence intervals. While the visual comparison (Figure 7) shows comparable perceptual quality, given that FID scores for unconditional generation can exhibit meaningful run-to-run variance, the quantitative claim of "no model quality loss" cannot be substantiated without statistical support. The claim in the abstract should be qualified.

    The FPI convergence behavior over training is not characterized. Figure 10 reports the aggregate computational cost ratio 
    across window sizes, showing 
    . However, we do not see how 
    (FPI iterations per window) varies as a function of training progress. This is important because the warm-start initialization can only provide good initial guesses once training is in a "settled" regime; early in training, when weights change rapidly, 
    may be much larger. Reporting 
    over training steps would clarify when PASO's speedup actually materializes and help practitioners decide when to enable step-parallelism.

    Missing pipeline parallelism baselines. Table 1 includes pipeline parallelism in the theoretical comparison, but no experimental comparison is provided against GPipe or PipeDream on the actual DL tasks. Since pipeline parallelism also targets the sequential dependency (across model stages rather than time steps), it is the most natural experimental competitor.

    Presentation clarity. Several figures do not aid rapid comprehension: Figure 1's schematic and the parallel coordinates plot (Figure 9) required careful reading of the surrounding text to interpret, when ideally figures should be largely self-explanatory. Additionally, a substantial portion of the appendix (pages 16-19, 22-25) consists of verbatim Python execution commands that belong in a supplementary code repository rather than the paper body. The appendix space would be better used for the missing convergence rate analysis or per-window 
    diagnostics.

    References

    [1] Lupo Pasini, M., Yin, J., Reshniak, V., and Stoyanov, M. K. "Stable Anderson Acceleration for Deep Learning." 2021. https://arxiv.org/abs/2110.14813

    Soundness: 3: good
    Presentation: 2: fair
    Significance: 3: good
    Originality: 3: good
    Key Questions For Authors:
    Can you provide a plot of FPI iterations 
    as a function of training step? This would clarify whether the speedup is uniform or concentrated in later training stages.

    Does PASO help in using larger learning rates without divergence?The FPI convergence criterion effectively validates each window's solution before proceeding. Does this implicit check enable the use of larger learning rates that would otherwise cause divergence in sequential training?

    How does PASO perform when the sliding window encounters a sharp loss landscape transition (e.g., learning rate warmup, sudden distribution shift in data)? Does 
    spike in these regimes?

    PASO's sliding-window FPI bears a structural resemblance to Parareal (Lions et al., 2001) and related parallel-in-time ODE solvers, which the paper acknowledges in Section 2. In Parareal, error propagation across time windows is a known challenge with well-studied corrector mechanisms. Have the authors considered adapting Parareal-style coarse-fine correction to PASO's setting, and if so, what specific obstacles arise from the stochastic and non-smooth nature of SGD that are absent in the ODE context?

    I encourage the authors to address the key questions in the rebuttal. I would be glad to revisit my score upward if the authors can provide responses to the theoretical concerns (W1-W3) and the missing convergence diagnostics (W6).

    Limitations:
    Largely yes. The authors acknowledge the gap between step-level and runtime speedup and attribute it to engineering constraints. One suggested addition: acknowledging the theoretical limitations, namely that the convergence proofs do not yet cover the sliding-window version and that no convergence rate bound is provided, would complement the engineering constraints already discussed.

    Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
    Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    Compliance With LLM Reviewing Policy: Affirmed.
    Code Of Conduct Acknowledgement: Affirmed.
  ```