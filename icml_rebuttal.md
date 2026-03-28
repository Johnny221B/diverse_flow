## Reviewer 1

**Dear Reviewer TpA3,**

We thank the reviewer for the positive feedback, in particular for highlighting: **(i) the interesting motivation of addressing flow model collapse; (ii) the theoretical analysis on increasing a volume surrogate while preserving the base behavior; and (iii) the comprehensive empirical evaluation.** We address the remaining concerns point by point below.

> Weakness 1: First-order theory only

> Weakness 2: Non-negligible runtime overhead
We thank the reviewer for pointing this out. We agree that our previous wording was not sufficiently precise: given the reported increase from 237 to 451, describing the overhead as *modest* is too optimistic. OSCAR introduces additional cost because the diversity gradient requires pairwise interactions, leading to an \(O(n^2)\) dependence on batch size, so higher overhead at larger batch sizes is expected. At the same time, this cost decreases substantially in smaller-batch regimes that are often more relevant in practice; for example, with batch size 8, OSCAR incurs only about **30--40\%** additional runtime over the base model. We will revise the paper to use more accurate wording and make the batch-size/runtime tradeoff more explicit.

> Weakness 3: Missing DiverseFlow baseline
We thank the reviewer for pointing this out. The baseline referred to as  **“DPP” in our tables is in fact DiverseFlow** . We used the shorthand “DPP” because the method is fundamentally based on Determinantal Point Processes, but we agree this naming can be confusing. To avoid ambiguity, we will revise the paper to consistently refer to this baseline as **DiverseFlow** in both the main text and all tables.

**Best regards,** 
**Authors**



## Reviewer 2

**Dear Reviewer MVWf,**

We thank the reviewer for the positive feedback, in particular for highlighting: **(i) the novelty and completeness of the proposed method; (ii) the intuition behind the orthogonal design; and (iii) the overall strength of the empirical study.** and we will report the tradeoff more transparently.

> Weakness 1: Missing runtime evaluation, how to balance efficiency and quality

We thank the reviewer for raising this important point. We do provide a runtime analysis in **Appendix E.3 (Fair-Budget Comparison)**, where we report the inference cost under a batch size of 32. Since OSCAR involves pairwise interactions in the diversity gradient, the additional overhead is expected due to its \(O(n^2)\) dependence on batch size. That said, this cost decreases noticeably in smaller-batch regimes that are often more relevant in practice: for example, when using a batch size of 8, OSCAR incurs only about **30–40%** additional runtime over the base model. We agree that efficiency–quality tradeoffs are important for real-world deployment, and we will make this discussion more explicit in the main paper and clarify how the overhead scales with batch size.

> Weakness 2: Limited qualitative diversity evidence

> Weakness 3: Lack of challenging human/portrait examples (maintain image quality)



**Best regards,** 
**Authors**



## Reviewer 3

**Dear Reviewer 79u4,**

We thank the reviewer for the positive feedback, in particular for highlighting: **(i) the feature-space volume maximization formulation for improving diversity; (ii) the empirical evidence that diversity and quality can improve simultaneously; and (iii) the clear and easy-to-follow presentation.** and we will report the tradeoff more transparently.

> Weakness 1: Missing APG related work / baseline

We thank the reviewer for pointing this out. We agree that APG is a relevant baseline, and we have now implemented it in our framework for direct comparison. On the simplest truck / bus / bicycle setting, APG is indeed a strong baseline: its quality metrics are generally comparable to ours. However, on diversity-related metrics, our method consistently outperforms APG across all three cases. We also evaluated APG on more complex prompts from T2I-Bench, and observed the same overall trend, as shown in the table below.

| method | guidance | vendi_pixel   | vendi_inception | fid           | clip_score    | one_minus_ms_ssim | brisque      |
|--------|----------|---------------|-----------------|---------------|---------------|-------------------|--------------|
| apg    | 3.0      | 2.61 ± 0.28   | 4.76 ± 0.67     | 166.62 ± 1.11 | 27.30 ± 0.55  | 85.16 ± 2.14      | 24.89 ± 2.23 |
| apg    | 5.0      | 2.40 ± 0.30   | 4.27 ± 0.40     | 165.55 ± 1.48 | 26.73 ± 0.50  | 83.53 ± 1.81      | 27.97 ± 2.04 |
| apg    | 7.5      | 2.31 ± 0.30   | 4.05 ± 0.29     | 165.77 ± 1.61 | 26.67 ± 0.47  | 82.93 ± 2.55      | 30.09 ± 2.20 |

> Weakness 2: Inefficient for one image per concept

> Weakness 3: Limited concept coverage in evaluation

We thank the reviewer for the suggestion and have conducted additional experiments on the color and complex subsets of T2I-Bench. Specifically, for each subset, we randomly sampled 80 prompts from the benchmark and generated 16 images per prompt. We then evaluated all methods using the same diversity and quality metrics as in the main paper. The results show that our method performs favorably against most baselines on both subsets, demonstrating a consistently strong overall performance across the evaluated metrics. These results further support our main claim that our method can improve output diversity while maintaining competitive generation quality.

| method    | vendi_inception | vendi_pixel | clip_score | one_minus_ms_ssim | BLIPvqa |
|-----------|----------------:|------------:|-----------:|------------------:|--------:|
| base      | 3.14            | 2.20        | 38.93      | 0.8868            | 0.8242  |
| cads      | 3.16            | 2.21        | 38.98      | 0.8872            | 0.8257  |
| pg        | 3.15            | 2.32        | 39.28      | 0.8653            | 0.8265  |
| dpp       | 3.11            | 2.40        | 38.97      | 0.8750            | 0.8083  |
| apg       | 3.09            | 2.38        | 39.14      | 0.8897            | 0.8371  |
| ourmethod | 3.24            | 2.53        | 39.19      | 0.9025            | 0.8369  |

| method    | vendi_inception | vendi_pixel | clip_score | one_minus_ms_ssim | BLIPvqa |
|-----------|----------------:|------------:|-----------:|------------------:|--------:|
| base      | 2.95            | 1.89        | 36.03      | 0.8092            | 0.7384  |
| cads      | 3.04            | 1.86        | 35.20      | 0.8231            | 0.7245  |
| pg        | 3.02            | 1.94        | 35.37      | 0.7835            | 0.7811  |
| dpp       | 2.93            | 1.99        | 35.55      | 0.7984            | 0.7914  |
| apg       | 2.97            | 1.99        | 35.61      | 0.8039            | 0.7953  |
| ourmethod | 3.15            | 2.13        | 35.53      | 0.8185            | 0.7952  |

> Weakness 4: Insufficient baseline implementation details

We thank the reviewer for this helpful suggestion and agree that the implementation details of the baselines should be made more explicit.

For **CADS**, we used the hyperparameters `tau1=0.6`, `tau2=0.9`, `cads_s=0.10`, and `noise_scale=1.0`. To ensure a fair comparison, we aligned the **noise-related settings in CADS** with those used in our method as much as possible. We also found that CADS is relatively sensitive to the noise magnitude: for example, increasing `noise_scale` to `2.0` often leads to noticeable colored spots and artifacts in the generated images. Based on this observation, we chose the above setting as a fair and stable configuration for the baseline.

For **DiverseFlow**, we used `gamma_sched=sqrt`, `kernel_spread=3.0`, and `gamma_max=0.12`. We believe these choices are well motivated by the original method design. In particular, DiverseFlow explicitly formulates the diversity strength as a time-varying term, so using a `sqrt` schedule is consistent with the method’s intended behavior of applying stronger diversification earlier and reducing it later. The `kernel_spread=3.0` setting corresponds to a moderate DPP kernel width, which provides effective repulsion without making the kernel overly sharp or unstable. Finally, we set `gamma_max=0.12` as a conservative upper bound so that the diversity guidance remains effective while avoiding excessive deviation that could harm image quality. Overall, these settings were chosen to provide a stable and faithful implementation of the baseline, rather than an under-tuned one.

**Best regards,** 
**Authors**



## Reviewer 4

**Dear Reviewer PecY,**

We thank the reviewer for the positive feedback, in particular for highlighting:
**(i) the novel motivation of broadening generative flow to improve diversity; (ii) the reasonable geometric design, especially the endpoint-feature-based diversity gradient; and (iii) the rich and detailed analysis, including the extensive supplementary experiments.** and we will report the tradeoff more transparently.

> Weakness 1: Limited evaluation on complex prompts

We thank the reviewer for the suggestion and have conducted additional experiments on the color and complex subsets of T2I-Bench. Specifically, for each subset, we randomly sampled 80 prompts from the benchmark and generated 16 images per prompt. We then evaluated all methods using the same diversity and quality metrics as in the main paper. The results show that our method performs favorably against most baselines on both subsets, demonstrating a consistently strong overall performance across the evaluated metrics. These results further support our main claim that our method can improve output diversity while maintaining competitive generation quality.

We also observe that the hybrid SDE/ODE method is relatively competitive on several diversity-related metrics and is the closest baseline to ours in that regard. However, this comes with a noticeable degradation in quality: its CLIP score and BLIP-VQA are substantially lower than those of the other methods. This suggests that its diversity gains are achieved at the expense of text-image alignment and semantic correctness of the generated content. In contrast, our method achieves a better trade-off between diversity and quality, which we believe makes it more practically useful.

| method    | vendi_inception | vendi_pixel | clip_score | one_minus_ms_ssim | BLIPvqa |
|-----------|----------------:|------------:|-----------:|------------------:|--------:|
| base      | 3.14            | 2.20        | 38.93      | 0.8868            | 0.8242  |
| cads      | 3.16            | 2.21        | 38.98      | 0.8872            | 0.8257  |
| pg        | 3.15            | 2.32        | 39.28      | 0.8653            | 0.8265  |
| dpp       | 3.11            | 2.40        | 38.97      | 0.8750            | 0.8083  |
| mix       | 3.23            | 2.54        | 38.27      | 0.9375            | 0.8016  |
| ourmethod | 3.24            | 2.53        | 39.19      | 0.9025            | 0.8369  |

| method    | vendi_inception | vendi_pixel | clip_score | one_minus_ms_ssim | BLIPvqa |
|-----------|----------------:|------------:|-----------:|------------------:|--------:|
| base      | 2.95            | 1.89        | 36.03      | 0.8092            | 0.7384  |
| cads      | 3.04            | 1.86        | 35.20      | 0.8231            | 0.7245  |
| pg        | 3.02            | 1.94        | 35.37      | 0.7835            | 0.7811  |
| dpp       | 2.93            | 1.99        | 35.55      | 0.7984            | 0.7914  |
| mix       | 3.02            | 2.16        | 34.04      | 0.8201            | 0.7353  |
| ourmethod | 3.15            | 2.13        | 35.53      | 0.8185            | 0.7952  |

> Weakness 2: Missing Early-SDE + Late-ODE baseline

| method      | guidance | FID           | CLIP Score      | Vendi Pixel    | Vendi Inception | 1 - MS-SSIM    | BRISQUE        |
|-------------|----------|---------------|-----------------|----------------|-----------------|----------------|----------------|
| mix_sde_ode | 3.0      | 166.55 ± 1.90 | 26.68 ± 0.23    | 4.94 ± 0.23    | 4.26 ± 0.19     |  86.61 ± 0.41  | 47.30 ± 1.88   |
| mix_sde_ode | 5.0      | 165.70 ± 1.57 | 26.05 ± 0.28    | 4.31 ± 0.25    | 3.43 ± 0.16     |  84.58 ± 0.66  | 44.88 ± 1.78   |
| mix_sde_ode | 7.5      | 166.04 ± 1.65 | 25.84 ± 0.31    | 4.06 ± 0.23    | 3.18 ± 0.15     |  83.74 ± 0.97  | 44.74 ± 1.83   |

We additionally evaluated the Early-SDE + Late-ODE baseline on the simplest truck / bus / bicycle setting. We observed a consistent pattern: although this baseline tends to achieve relatively strong diversity metrics, its quality metrics are almost always the worst among all compared methods. In other words, its diversity gains come with a substantial degradation in generation quality. Due to the space limit, we only present the results for the bicycle case here.

We further tested the same baseline on more complex scenarios, as shown in the table above, and observed the same trend. While the method remains competitive on some diversity-related metrics, it consistently underperforms in quality-related metrics such as CLIP Score and BRISQUE, indicating weaker text-image alignment and poorer visual fidelity. These results are consistent with our broader findings: compared with such baselines, our method achieves a more favorable trade-off between diversity and quality.

> Weakness 3: Batch-coupled design / no offline alternative

**Best regards,** 
**Authors**