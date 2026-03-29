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

We thank the reviewer for this helpful suggestion. We agree that the originally shown qualitative examples were relatively limited in scenario diversity. To better assess whether OSCAR improves diversity beyond fixed settings such as buses or restaurants, we conducted additional experiments on **T2I-CompBench**, where we randomly sampled 80 prompts from each of the **color**, **complex**, and **spatial** subsets and generated images for comparison across methods using the same evaluation protocol.

In addition, to provide more direct qualitative evidence, we selected one representative prompt from each subset and included the generated samples in the linked results: **“a blue bench and a green boat”** (color), **“a bicycle on the left of a bird”** (spatial), and **“The black camera was mounted on the silver tripod”** (complex) in **section 1:Qualitative Comparison: Diversity**. These examples were chosen to cover a broader range of compositional and attribute-binding scenarios than the original examples. The aggregated quantitative results over the 80 sampled prompts per subset are reported in the table below. Together, the expanded T2I-Bench evaluation and these qualitative cases provide stronger evidence that OSCAR improves diversity beyond the previously shown fixed scenarios. Due to space limits, results can be found in the https://anonymous.4open.science/r/2026-ICML-rebuttal-BB38 under **Section 3: Quantitative Results on T2I-CompBench**.

| method    | vendi_inception | vendi_pixel | clip_score | one_minus_ms_ssim | BLIPvqa |
|-----------|----------------:|------------:|-----------:|------------------:|--------:|
| base      | 3.14            | 2.20        | 38.93      | 0.8868            | 0.8242  |
| cads      | 3.16            | 2.21        | 38.98      | 0.8872            | 0.8257  |
| pg        | 3.15            | 2.32        | 39.28      | 0.8653            | 0.8265  |
| dpp       | 3.11            | 2.40        | 38.97      | 0.8750            | 0.8083  |
| ourmethod | 3.24            | 2.53        | 39.19      | 0.9025            | 0.8369  |

> Weakness 3: Lack of challenging human/portrait examples (maintain image quality)

We thank the reviewer for this helpful suggestion. To better demonstrate OSCAR’s ability to preserve image quality, we added a new set of challenging **portrait-focused qualitative examples** specifically designed to stress fine-grained visual fidelity, including **facial plausibility, skin and hand details, material rendering, and complex lighting conditions**. 

Concretely, we constructed several detailed prompts involving human subjects, such as:  
- *“A studio portrait of a person in 1920s vintage Gatsby-style formal wear, intricate lace and feather details, monochrome noir lighting.”*  
- *“A portrait of an artist in a paint-splattered apron standing in front of a large abstract canvas, holding a brush, messy hair, warm studio light.”*  
- *“A professional portrait of a master artisan working in a sunlit woodcarving workshop, fine dust in the air, highly detailed skin textures and hands.”*  
- *“A medium-shot of a person wearing a glossy silk evening gown, standing on a rainy city balcony at night with vibrant neon reflections, high fashion style.”*

For each prompt, we provide 4 generated samples in the linked results. These examples were chosen because they are particularly sensitive to quality degradation under diversity enhancement, especially in terms of facial plausibility, fine local details, and material rendering. We include them to visually demonstrate that OSCAR does not noticeably degrade image quality in these challenging portrait settings. The qualitative results are available in the anonymous repository at https://anonymous.4open.science/r/2026-ICML-rebuttal-BB38/readme.md under **Section 2. Qualitative Comparison: Quality Preservation on Challenging Portrait Prompts**.

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

We thank the reviewer for the suggestion and have conducted additional experiments on the color and complex subsets of T2I-Bench. Specifically, for each subset, we randomly sampled 80 prompts from the benchmark and generated 16 images per prompt. We then evaluated all methods using the same diversity and quality metrics as in the main paper. The results show that our method performs favorably against most baselines on both subsets, demonstrating a consistently strong overall performance across the evaluated metrics. These results further support our main claim that our method can improve output diversity while maintaining competitive generation quality. Due to space limits, results can be found in the https://anonymous.4open.science/r/2026-ICML-rebuttal-BB38 under **Section 3: Quantitative Results on T2I-CompBench**.

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

We thank the reviewer for the suggestion and conducted additional experiments on the **color** and **complex** subsets of T2I-Bench. For each subset, we randomly sampled **80 prompts** and generated **16 images per prompt**, using the same diversity and quality metrics as in the main paper. The results show that our method compares favorably with most baselines on both subsets, indicating consistently strong overall performance. We also find that the hybrid SDE/ODE baseline is competitive on some diversity metrics, but suffers clear drops in **CLIP score** and **BLIP-VQA**, suggesting weaker text-image alignment and semantic correctness. In contrast, our method achieves a better diversity–quality trade-off. Due to space limits, we show only the **complex** results here; additional quantitative results can be found in the https://anonymous.4open.science/r/2026-ICML-rebuttal-BB38 under **Section 3: Quantitative Results on T2I-CompBench**.

| method    | vendi_inception | vendi_pixel | clip_score | one_minus_ms_ssim | BLIPvqa |
|-----------|----------------:|------------:|-----------:|------------------:|--------:|
| base      | 2.95            | 1.89        | 36.03      | 0.8092            | 0.7384  |
| cads      | 3.04            | 1.86        | 35.20      | 0.8231            | 0.7245  |
| pg        | 3.02            | 1.94        | 35.37      | 0.7835            | 0.7811  |
| dpp       | 2.93            | 1.99        | 35.55      | 0.7984            | 0.7914  |
| mix       | 3.02            | 2.16        | 34.04      | 0.8201            | 0.7353  |
| oscar     | 3.15            | 2.13        | 35.53      | 0.8185            | 0.7952  |

> Weakness 2: Missing Early-SDE + Late-ODE baseline

| method      | guidance | FID           | CLIP Score      | Vendi Pixel    | Vendi Inception | 1 - MS-SSIM    | BRISQUE        |
|-------------|----------|---------------|-----------------|----------------|-----------------|----------------|----------------|
| mix_sde_ode | 3.0      | 166.55 ± 1.90 | 26.68 ± 0.23    | 4.94 ± 0.23    | 4.26 ± 0.19     |  86.61 ± 0.41  | 47.30 ± 1.88   |
| mix_sde_ode | 5.0      | 165.70 ± 1.57 | 26.05 ± 0.28    | 4.31 ± 0.25    | 3.43 ± 0.16     |  84.58 ± 0.66  | 44.88 ± 1.78   |
| mix_sde_ode | 7.5      | 166.04 ± 1.65 | 25.84 ± 0.31    | 4.06 ± 0.23    | 3.18 ± 0.15     |  83.74 ± 0.97  | 44.74 ± 1.83   |

We additionally evaluated the Early-SDE + Late-ODE baseline in both the simplest setting and more complex scenarios, and observed a consistent pattern: while it can obtain relatively strong diversity metrics, it consistently performs worse on quality-related metrics, indicating that its diversity gains come at the cost of degraded text-image alignment and visual fidelity. Due to space limits, we present only the bicycle results here. Overall, these results support our main conclusion that OSCAR achieves a more favorable diversity–quality trade-off than this simple hybrid-sampling baseline.

> Weakness 3: Batch-coupled design / no offline alternative

We thank the reviewer for this insightful question. We agree that OSCAR is batch-coupled in its current form, since the Gram-matrix-based repulsive term is defined over a jointly active set of trajectories. When \(m\) is small, the method remains applicable, but the diversity signal becomes more local. To address this concern, we include an additional small-batch evaluation with \(m=4\). Our preliminary results show that, although the diversity gain is weaker than in the larger-batch regime, OSCAR still consistently improves over the base model in this setting.

| method | vendi_inception | vendi_pixel | clip_score | one_minus_ms_ssim |
|--------|----------------:|------------:|-----------:|------------------:|
| base   | 1.84            | 1.64        | 37.75      | 0.8686            |
| cads   | 1.83            | 1.64        | 37.82      | 0.8686            |
| pg     | 1.83            | 1.58        | 37.81      | 0.8391            |
| dpp    | 1.87            | 1.67        | 37.13      | 0.8654            |
| mix    | 1.88            | 1.74        | 36.64      | 0.9033            |
| oscar  | 1.88            | 1.76        | 37.95      | 0.8825            |

More generally, a natural offline extension is to maintain a **memory bank of cached endpoint features** from previous generations. The repulsive term can then be computed not only against the currently active set, but also against these cached features, which would provide a broader semantic reference under low-VRAM or asynchronous generation settings. We view this as a promising generalization of the current online formulation and will clarify this point in the revision.

**Best regards,** 
**Authors**