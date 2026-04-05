## Reviewer 1

**Dear Reviewer TpA3,**

We thank the reviewer for the positive feedback, in particular for highlighting: **(i) the interesting motivation of addressing flow model collapse; (ii) the theoretical analysis on increasing a volume surrogate while preserving the base behavior; and (iii) the comprehensive empirical evaluation.** We address the remaining concerns point by point below.

> Q1: First-order theory only

We thank the reviewer for this important point. We do not claim exact distribution preservation. Rather, beyond the local first-order result, our theory also provides a **global bounded-deviation guarantee**: theorem B 13 shows that the end-time deviation is bounded by $O(\Delta t)+O(B)$, with $B=\int_0^1 \beta(t)\,dt$. Hence, the distribution shift induced by OSCAR is controlled rather than arbitrary.

For very large $m$, the toy samples can appear closer to a uniform within-support coverage because our log-det objective is equivalent to increasing the set volume; Theorem 2 further shows that this reduces pairwise correlations and makes the singular values more uniform. This creates a **set-level repulsive spreading effect**, encouraging particles to spread across feasible regions within and across modes. In Fig. 13, this effect is amplified by the intentionally extreme choice $m=2000$, which is not representative of practical image-generation usage.

At the same time, under otherwise identical settings, Fig. 12 with $m=200$ already shows that OSCAR can still maintain the overall target distribution much better in a more moderate regime. We will clarify that Fig. 13 is intended to illustrate the limiting behavior of OSCAR under extreme $m$, whereas Fig. 12 better reflects its ability to preserve distributional structure in realistic settings.

> Q2: Non-negligible runtime overhead

We thank the reviewer for pointing this out. We agree that our previous wording was not sufficiently precise: given the reported increase from 237 to 451, describing the overhead as *modest* is too optimistic. OSCAR introduces additional cost because the diversity gradient requires pairwise interactions, leading to an $O(n^2)$ dependence on batch size, so higher overhead at larger batch sizes is expected. At the same time, this cost decreases substantially in smaller-batch regimes that are often more relevant in practice; for example, with batch size 8, OSCAR incurs only about **30--40\%** additional runtime over the base model. Importantly, OSCAR remains effective even at **batch size 4**, as shown in the table below. We will revise the paper to use more accurate wording and make the batch-size/runtime tradeoff more explicit.

| Method   | Vendi Score Inception | Vendi Score Pixel | CLIP Score | 1 − MS-SSIM(%) | BLIPvqa |
|----------|----------------------:|------------------:|-----------:|----------------:|--------:|
| FM-SD3.5 | 3.14                  | 2.20              | 38.93      | 0.8868          | 0.8242  |
| CADS     | 3.16                  | 2.21              | 38.98      | 0.8872          | 0.8257  |
| PG       | 3.15                  | 2.32              | 39.28      | 0.8653          | 0.8265  |
| DPP      | 3.11                  | 2.40              | 38.97      | 0.8750          | 0.8083  |
| OSCAR    | 3.24                  | 2.53              | 39.19      | 0.9025          | 0.8369  |

> Q3: Missing DiverseFlow baseline

We thank the reviewer for pointing this out. The baseline referred to as  **“DPP” in our tables is in fact DiverseFlow** [1] . We used the shorthand “DPP” because the method is fundamentally based on Determinantal Point Processes, but we agree this naming can be confusing. To avoid ambiguity, we will revise the paper to consistently refer to this baseline as **DiverseFlow** in both the main text and all tables.

[1] Morshed et al. "DiverseFlow" arXiv:2504.07894, 2025

**Best regards,** 

**Authors**



## Reviewer 2

**Dear Reviewer MVWf,**

We thank the reviewer for the positive feedback, in particular for highlighting: **(i) the novelty and completeness of the proposed method; (ii) the intuition behind the orthogonal design; and (iii) the overall strength of the empirical study.** and we will report the tradeoff more transparently.

> Q1: Missing runtime evaluation, how to balance efficiency and quality

We thank the reviewer for raising this important point. We do provide a runtime analysis in **Appendix E.3 (Fair-Budget Comparison)**, where we report the inference cost under a batch size of 32. Since OSCAR involves pairwise interactions in the diversity gradient, the additional overhead is expected due to its $O(n^2)$ dependence on batch size. That said, this cost decreases noticeably in smaller-batch regimes that are often more relevant in practice: for example, when using a batch size of 8, OSCAR incurs only about **30–40%** additional runtime over the base model. We agree that efficiency–quality tradeoffs are important for real-world deployment, and we will make this discussion more explicit in the main paper and clarify how the overhead scales with batch size.

> Q2: Limited qualitative diversity evidence

We thank the reviewer for this helpful suggestion. We agree that the originally shown qualitative examples were relatively limited in scenario diversity. To better assess whether OSCAR improves diversity beyond fixed settings such as buses or restaurants, we conducted additional experiments on T2I-CompBench [1], where we randomly sampled 80 prompts from each of the **color**, **complex**, and **spatial** subsets and generated images for comparison across methods using the same evaluation protocol.

In addition, to provide more direct qualitative evidence, we selected one representative prompt from each subset and included the generated samples in the linked results: **“a blue bench and a green boat”** (color), **“a bicycle on the left of a bird”** (spatial), and **“The black camera was mounted on the silver tripod”** (complex) in **section 1:Qualitative Comparison: Diversity**. These examples were chosen to cover a broader range of compositional and attribute-binding scenarios than the original examples. The aggregated quantitative results over the 80 sampled prompts per subset are reported in the table below. Together, the expanded T2I-Bench evaluation and these qualitative cases provide stronger evidence that OSCAR improves diversity beyond the previously shown fixed scenarios. Due to space limits, more results can be found in the https://anonymous.4open.science/r/2026-ICML-rebuttal-BB38.

| Method   | Vendi Score Inception | Vendi Score Pixel | CLIP Score | 1 − MS-SSIM     | BLIPvqa |
|----------|----------------------:|------------------:|-----------:|----------------:|--------:|
| FM-SD3.5 | 3.14                  | 2.20              | 38.93      | 0.8868          | 0.8242  |
| CADS     | 3.16                  | 2.21              | 38.98      | 0.8872          | 0.8257  |
| PG       | 3.15                  | 2.32              | 39.28      | 0.8653          | 0.8265  |
| DPP      | 3.11                  | 2.40              | 38.97      | 0.8750          | 0.8083  |
| OSCAR    | 3.24                  | 2.53              | 39.19      | 0.9025          | 0.8369  |

> Q3: Lack of challenging human/portrait examples (maintain image quality)

We thank the reviewer for this helpful suggestion. To better demonstrate OSCAR’s ability to preserve image quality, we added a new set of challenging **portrait-focused qualitative examples** specifically designed to stress fine-grained visual fidelity, including **facial plausibility, skin and hand details, material rendering, and complex lighting conditions**. 

Concretely, we constructed several detailed prompts involving human subjects, such as:  
- *“A studio portrait of a person in 1920s vintage Gatsby-style formal wear, intricate lace and feather details, monochrome noir lighting.”*  
- *“A portrait of an artist in a paint-splattered apron standing in front of a large abstract canvas, holding a brush, messy hair, warm studio light.”*  
- *“A medium-shot of a person wearing a glossy silk evening gown, standing on a rainy city balcony at night with vibrant neon reflections, high fashion style.”*

For each prompt, we provide 4 generated samples in the same link. These examples were chosen because they are particularly sensitive to quality degradation under diversity enhancement, especially in terms of facial plausibility, fine local details, and material rendering. We include them to visually demonstrate that OSCAR does not noticeably degrade image quality in these challenging portrait settings. 

[1] Huang et al. "T2I-CompBench" NeurIPS 2023.

**Best regards,** 

**Authors**



## Reviewer 3

**Dear Reviewer 79u4,**

We thank the reviewer for the positive feedback, in particular for highlighting: **(i) the feature-space volume maximization formulation for improving diversity; (ii) the empirical evidence that diversity and quality can improve simultaneously; and (iii) the clear and easy-to-follow presentation.** and we will report the tradeoff more transparently.

> Q1: Missing APG

We thank the reviewer for pointing this out. We agree that APG is a relevant baseline, and we have now implemented it in our framework for direct comparison. On the simplest setting, APG is indeed a strong baseline: its quality metrics are generally comparable to ours, as shown in the table below. However, on diversity-related metrics, our method consistently outperforms APG across all three cases. 

| Method | Guidance Scale (CFG) | Vendi Score Pixel | Vendi Score Inception | FID | CLIP Score | 1 − MS-SSIM(%) | Brisque |
|--------|---------------------:|------------------:|----------------------:|----:|-----------:|----------------:|--------:|
| APG    | 3.0                  | 2.61 ± 0.28       | 4.76 ± 0.67           | 166.62 ± 1.11 | 27.30 ± 0.55 | 85.16 ± 2.14 | 24.89 ± 2.23 |
| APG    | 5.0                  | 2.40 ± 0.30       | 4.27 ± 0.40           | 165.55 ± 1.48 | 26.73 ± 0.50 | 83.53 ± 1.81 | 27.97 ± 2.04 |
| APG    | 7.5                  | 2.31 ± 0.30       | 4.05 ± 0.29           | 165.77 ± 1.61 | 26.67 ± 0.47 | 82.93 ± 2.55 | 30.09 ± 2.20 |

Regarding the similarity to Eq. (4) of APG, we agree that the two methods share a similar projection form, but they are not the same orthogonal guidance. APG applies orthogonal decomposition to the CFG update to mitigate oversaturation at high CFG, whereas our method is derived from the objective of maximizing the ellipsoid volume, with the guidance signal given by the component orthogonal to the local tangent direction. Thus, the objective and guidance signal are different. We will revise the paper to cite APG explicitly and revised our novelty claim.

> Q2: One image per concept

We thank the reviewer for this important point. We agree that OSCAR, in its current form, relies on a jointly active set of samples. In the single-image-per-concept setting, however, diversity itself is inherently difficult to define and may be confounded with distribution shift. Still, OSCAR remains effective even at small $m$: as shown in the table below, with $m=4$ our method continues to lead across the evaluated metrics. More generally, an offline extension is possible by replacing the fully online active set with a memory bank of cached endpoint features, so that diversity guidance can be applied without requiring multiple images to be generated simultaneously. We will clarify this limitation and extension direction in the revision.

| Method   | Vendi Score Inception | Vendi Score Pixel | CLIP Score | 1 − MS-SSIM     |
|----------|----------------------:|------------------:|-----------:|----------------:|
| FM-SD3.5 | 1.84                  | 1.64              | 37.75      | 0.8686          |
| CADS     | 1.83                  | 1.64              | 37.82      | 0.8686          |
| PG       | 1.83                  | 1.58              | 37.81      | 0.8391          |
| DPP      | 1.87                  | 1.67              | 37.13      | 0.8654          |
| OSCAR    | 1.88                  | 1.76              | 37.95      | 0.8825          |

> Q3: Limited evaluation

We thank the reviewer for the suggestion and conducted additional experiments on the color and complex subsets of T2I-CompBench [1], using 80 randomly sampled prompts per subset and 16 images per prompt. Our method performs favorably against most baselines on both subsets under the same diversity and quality metrics as in the paper. Full results are provided in the anonymous repository: https://anonymous.4open.science/r/2026-ICML-rebuttal-BB38.

> Q4: Baseline details

We thank the reviewer for the suggestion and agree that the baseline implementation details should be stated more clearly. For **CADS**, we used `tau1=0.6`, `tau2=0.9`, `cads_s=0.10`, and `noise_scale=1.0`, with noise-related settings aligned as much as possible to those in our method for fairness. We found CADS to be sensitive to noise magnitude: for example, increasing `noise_scale` to `2.0` often introduced colored spots and visible artifacts. We therefore adopted the above setting as a fair and stable configuration. For **DiverseFlow**, we used `gamma_sched=sqrt`, `kernel_spread=3.0`, and `gamma_max=0.12`. These choices follow the method’s design: stronger diversification earlier in sampling and weaker guidance later, with moderate kernel width and conservative guidance strength. Overall, these settings were chosen to provide a stable and faithful baseline implementation. We will include these implementation details explicitly in the revision for clarity.

[1] Huang et al. "T2I-CompBench" NeurIPS 2023.

**Best regards,** 

**Authors**



## Reviewer 4

**Dear Reviewer PecY,**

We thank the reviewer for the positive feedback, in particular for highlighting:
**(i) the novel motivation of broadening generative flow to improve diversity; (ii) the reasonable geometric design; and (iii) the rich and detailed analysis, including the extensive supplementary experiments.** and we will report the tradeoff more transparently.

> Q1: Limited evaluation

We thank the reviewer for the suggestion and conducted additional experiments on the **spatial** **color** and **complex** subsets of T2I-CompBench [1]. For each subset, we randomly sampled **80 prompts** and generated 16 images per prompt, using the same diversity and quality metrics as in the main paper. The results show that our method achieves the best overall performance on both subsets, with a consistently stronger diversity–quality trade-off than the baselines. In contrast, while the mix SDE/ODE baseline is competitive on some diversity metrics, it shows clear degradation on quality-related metrics. Due to space limits, we show only the **complex subset** results here; additional results can be found in the anonymous repository: https://anonymous.4open.science/r/2026-ICML-rebuttal-BB38.

| Method      | Vendi Score Inception | Vendi Score Pixel | CLIP Score | 1 − MS-SSIM | BLIPvqa |
|-------------|----------------------:|------------------:|-----------:|------------:|--------:|
| FM-SD3.5    | 2.95                  | 1.89              | 36.03      | 0.8092      | 0.7384  |
| CADS        | 3.04                  | 1.86              | 35.20      | 0.8231      | 0.7245  |
| PG          | 3.02                  | 1.94              | 35.37      | 0.7835      | 0.7811  |
| DPP         | 2.93                  | 1.99              | 35.55      | 0.7984      | 0.7914  |
| Mix ODE/SDE | 3.02                  | 2.16              | 34.04      | 0.8201      | 0.7353  |
| OSCAR       | 3.15                  | 2.13              | 35.53      | 0.8285      | 0.7952  |

> Q2: Missing baseline

| method      | guidance | FID           | CLIP Score      | Vendi Pixel    | Vendi Inception | 1 - MS-SSIM    | BRISQUE        |
|-------------|----------|---------------|-----------------|----------------|-----------------|----------------|----------------|
| Mix ODE/SDE | 3.0      | 166.55 ± 1.90 | 26.68 ± 0.23    | 4.94 ± 0.23    | 4.26 ± 0.19     |  86.61 ± 0.41  | 47.30 ± 1.88   |
| Mix ODE/SDE | 5.0      | 165.70 ± 1.57 | 26.05 ± 0.28    | 4.31 ± 0.25    | 3.43 ± 0.16     |  84.58 ± 0.66  | 44.88 ± 1.78   |
| Mix ODE/SDE | 7.5      | 166.04 ± 1.65 | 25.84 ± 0.31    | 4.06 ± 0.23    | 3.18 ± 0.15     |  83.74 ± 0.97  | 44.74 ± 1.83   |

We additionally evaluated the Early-SDE + Late-ODE baseline in both the simplest setting and more complex scenarios, and observed a consistent pattern: while it can obtain relatively strong diversity metrics, it consistently performs worse on quality-related metrics, indicating that its diversity gains come at the cost of degraded text-image alignment and visual fidelity. Due to space limits, we present only the bicycle results here, additional results are available in the same link.

> Q3: Batch-coupled design

We thank the reviewer for this insightful question. We agree that OSCAR is batch-coupled in its current form, since the Gram-matrix-based repulsive term is defined over a jointly active set of trajectories. When $m$ is small, the method remains applicable, but the diversity signal becomes more local. To address this concern, we include an additional small-batch evaluation with $m=4$. Our preliminary results show that, although the diversity gain is weaker than in the larger-batch regime, OSCAR still consistently improves over the base model in this setting.

| Method      | Vendi Score Inception | Vendi Score Pixel | CLIP Score | 1 − MS-SSIM |
|-------------|----------------------:|------------------:|-----------:|------------:|
| FM-SD3.5    | 1.84                  | 1.64              | 37.75      | 0.8686      |
| CADS        | 1.83                  | 1.64              | 37.82      | 0.8686      |
| PG          | 1.83                  | 1.58              | 37.81      | 0.8391      |
| DPP         | 1.87                  | 1.67              | 37.13      | 0.8654      |
| Mix ODE/SDE | 1.88                  | 1.74              | 36.64      | 0.9033      |
| OSCAR       | 1.88                  | 1.76              | 37.95      | 0.8825      |

More generally, a natural offline extension is to maintain a **memory bank of cached endpoint features** from previous generations. The repulsive term can then be computed not only against the currently active set, but also against these cached features, which would provide a broader semantic reference under low-VRAM or asynchronous generation settings. We view this as a promising generalization of the current online formulation and will clarify this point in the revision.

[1] Huang et al. "T2I-CompBench" NeurIPS 2023.

**Best regards,** 

**Authors**

**Dear Reviewer PecY,**

Thank you very much for your thoughtful follow-up and supportive remarks regarding our work. We are glad to see that your previous concerns have been fully resolved.

We will add the implementation details of the Mix ODE/SDE baseline in the revision. Concretely, we implement it as a simple **early-SDE / late-ODE hybrid sampler** with `t_gate = 0.7` and `eta_sde = 1.0`: when `t_norm > t_gate`, we inject Gaussian noise into the latent with standard deviation `eta_sde * sqrt(dt)`; otherwise, sampling proceeds with the standard ODE solver. 

Regarding why this baseline is commonly used in flow-matching RL works yet shows relatively weak visual quality in our experiments, we believe the key reason is the difference in **intended use**. In recent FM-RL methods such as **Flow-GRPO** [1], ODE-to-SDE conversion is introduced primarily to provide **stochastic exploration** for online RL. **MixGRPO** [2] further adopts a mixed ODE/SDE strategy with a sliding window, using SDE sampling only within the window and ODE sampling outside, in order to **reduce GRPO training overhead and improve alignment performance**. By contrast, our evaluation is purely **training-free inference-time** and directly measures the final diversity–quality trade-off of generated images. In this setting, a simple Mix ODE/SDE sampler indeed behaves as expected: it can improve diversity-related metrics, but because it introduces stochasticity without any explicit quality-preserving mechanism, the resulting gain often comes at the cost of degraded visual fidelity. We believe this explains why mixed ODE/SDE sampling can be beneficial in RL optimization, yet still serve as a relatively weak visual baseline in our setting.

We hope this clarification is helpful, and please let us know if you have any further questions or concerns.

[1] Liu et al. "Flow-GRPO" NeurIPS 2026.
[2] Li et al. "MixGRPO" arXiv:2507.21802, 2025.

**Best regards,** 

**Authors**

**Dear Reviewer PecY,**

Thank you very much for your thoughtful follow-up and supportive remarks regarding our work. We are glad to see that your previous concerns have been fully resolved.

We will add the implementation details of the Mix ODE/SDE baseline in the revision. Concretely, we implement it as a simple **early-SDE / late-ODE hybrid sampler** with `t_gate = 0.7` and `eta_sde = 1.0`: when `t_norm > t_gate`, we inject Gaussian noise into the latent with standard deviation `eta_sde * sqrt(dt)`; otherwise, sampling proceeds with the standard ODE solver.

Regarding why this baseline is commonly used in flow-matching RL works yet shows relatively weak visual quality in our experiments, we believe the key reason is the difference in **intended use**. In recent FM-RL methods such as **Flow-GRPO** [1], ODE-to-SDE conversion is introduced primarily to provide **stochastic exploration** for online RL. **MixGRPO** [2] further adopts a mixed ODE/SDE strategy with a sliding window, using SDE sampling only within the window and ODE sampling outside, in order to **reduce GRPO training overhead and improve alignment performance**. Similarly, **DanceGRPO** [3] also introduces stochastic sampling for GRPO optimization, while using ODE-based samplers for evaluation and visualization. By contrast, our evaluation is purely **training-free inference-time** and directly measures the final diversity–quality trade-off of generated images. In this setting, a simple Mix ODE/SDE sampler indeed behaves as expected: it can improve diversity-related metrics, but because it introduces stochasticity without any explicit quality-preserving mechanism, the resulting gain often comes at the cost of degraded visual fidelity. We believe this explains why mixed ODE/SDE sampling can be beneficial in RL optimization, yet still serve as a relatively weak visual baseline in our setting.

We hope this clarification is helpful, and please let us know if you have any further questions or concerns.

[1] Liu et al. "Flow-GRPO" NeurIPS 2025. 
[2] Li et al. "MixGRPO" arXiv:2507.21802, 2025. 
[3] Xue et al. "DanceGRPO" arXiv:2505.07818, 2025. 

**Best regards,**  
**Authors**

**Dear Reviewer 79u4,**

Thank you very much for your thoughtful follow-up comments. We find your suggestions very helpful and agree that the additional evaluation in our earlier rebuttal was still not sufficiently comprehensive, mainly due to the limited rebuttal timeline.

Following your suggestion, we have now conducted a more complete evaluation on the **color** and **complex** subsets of **CompT2I-Bench**. In total, we use **300 prompts** across these two subsets, and generate **32 images per prompt** for each method. We also report **mean ± std** for the evaluation metrics to better reflect the statistical stability of the results. In addition, we now include **KID** in the evaluation, as you suggested. The main reason we did not include it earlier in the rebuttal is that KID requires a reference dataset. In the updated evaluation, we use **5,000 images from COCO** as the reference set for KID computation.

| Method      | Vendi-Inc↑     | Vendi-Pix↑     | CLIP↑          | 1-MS-SSIM↑      | KID↓  |
|-------------|-----------------:|---------------:|----------------:|----------------:|------:|
| OSCAR       | 3.96 ± 0.13      | 2.92 ± 0.09    | 39.15 ± 0.30   | 0.890 ± 0.029   | 25.15 |
| Mix ODE/SDE | 3.62 ± 0.24      | 2.89 ± 0.16    | 38.10 ± 0.71   | 0.903 ± 0.012   | 31.99 |
| CADS        | 3.45 ± 0.11      | 2.43 ± 0.07    | 39.04 ± 0.20   | 0.891 ± 0.025   | 28.50 |
| FM-SD3.5    | 3.44 ± 0.22      | 2.42 ± 0.13    | 38.96 ± 0.23   | 0.889 ± 0.026   | 28.52 |
| DPP         | 3.41 ± 0.09      | 2.56 ± 0.09    | 38.96 ± 0.46   | 0.877 ± 0.017   | 29.17 |
| PG          | 3.42 ± 0.15      | 2.46 ± 0.07    | 39.17 ± 0.40   | 0.861 ± 0.029   | 30.30 |
| APG         | 3.34 ± 0.10      | 2.54 ± 0.12    | 39.08 ± 0.28   | 0.882 ± 0.027   | 30.34 |

Regarding your question about the earlier **m = 4** results, we would also like to clarify that the standard deviation reported in **Table 1** of the paper and the variation in our rebuttal **m = 4** experiment are computed under two different protocols, and are therefore not directly comparable. In Table 1, the reported std is measured across **many different prompts** (together with different seeds), so it includes substantial **across-prompt variation**. Since the metric values can vary significantly from one prompt to another, this leads to a relatively large std. In contrast, the rebuttal **m = 4** experiment was conducted by fixing the **same prompt** and varying only the random seeds. Therefore, it reflects **within-prompt variation**, which is much smaller than the across-prompt std reported in Table 1. This is why the variance in that setup is much smaller. Also, for the baseline methods, the generation process itself is not affected by **m**; here, **m** only refers to how many samples from the same prompt are used in this specific evaluation setup.

| Method      | Vendi-Inc↑     | Vendi-Pix↑     | CLIP↑          | 1-MS-SSIM↑      | KID↓  |
|-------------|-----------------:|---------------:|----------------:|----------------:|------:|
| OSCAR       | 3.96 ± 0.13      | 2.92 ± 0.09    | 39.15 ± 0.30   | 0.890 ± 0.029   | 25.15 |
| Mix ODE/SDE | 3.62 ± 0.24      | 2.89 ± 0.16    | 38.10 ± 0.71   | 0.903 ± 0.012   | 31.99 |
| CADS        | 3.45 ± 0.11      | 2.43 ± 0.07    | 39.04 ± 0.20   | 0.891 ± 0.025   | 28.50 |
| FM-SD3.5    | 3.44 ± 0.22      | 2.42 ± 0.13    | 38.96 ± 0.23   | 0.889 ± 0.026   | 28.52 |
| DPP         | 3.41 ± 0.09      | 2.56 ± 0.09    | 38.96 ± 0.46   | 0.877 ± 0.017   | 29.17 |
| PG          | 3.42 ± 0.15      | 2.46 ± 0.07    | 39.17 ± 0.40   | 0.861 ± 0.029   | 30.30 |
| APG         | 3.34 ± 0.10      | 2.54 ± 0.12    | 39.08 ± 0.28   | 0.882 ± 0.027   | 30.34 |

To make this point more transparent and to address your concern more directly, we further adopt a unified setup where each method generates **32 samples** under the same seed ordering, and we then evaluate the **first 4 samples** (consistently for all methods) as the **m = 4** case. We will include these updated **m = 4** results below for completeness.

We also supplement the rebuttal with more systematic qualitative comparisons in the same link. 

We hope these additions address your concerns regarding evaluation scale, statistical variation, and the use of KID under limited sample settings.

**Best regards,**  

**Authors**