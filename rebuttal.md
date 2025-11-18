## Reviewer 1

**Dear Reviewer Q6XH,**

**Thank you for your thoughtful and encouraging review. We sincerely appreciate your positive assessment of our work and your recognition of its potential. Below, we address your specific comments in more detail:**

> Weakness 1 Missing simplest FM baselines  / compute-matched comparison

1. **Simple FM baseline with \(N\) independent samples.**  
   In the revised manuscript, Table 1 now includes a “FM-SD3.5 (N particles)” baseline that generates \(N\) independent samples from the same Stable Diffusion 3.5 flow-matching model used by OSCAR, with identical NFE, prompts, and model settings. This isolates the effect of our orthogonal stochastic control from simply increasing the number of samples drawn from the base model.

2. **Compute-matched comparison.**  
   We further include a compute-matched study in Appendix Table A.x, where we compare OSCAR to two strengthened FM-SD3.5 variants under approximately matched FLOPs: (i) increasing the number of particles, and (ii) increasing the NFE. The results are summarized below:

   | Method        | NFE | Particles | FLOPs (T/run) | CLIP ↑           | Vendi (Pixel) ↑   | FID ↓             | BRISQUE ↓         |
   |--------------|-----|-----------|---------------|------------------|-------------------|-------------------|-------------------|
   | FM-SD3.5     | 30  | 32        | 4.09          | 28.24 ± 0.18     | 2.45 ± 0.13       | 164.4 ± 1.8       | 23.4 ± 1.4        |
   | +K particles | 30  | 48        | 6.14          | 28.15 ± 0.23     | 2.60 ± 0.21       | **149.7 ± 1.3**   | 23.5 ± 1.7        |
   | +N NFE       | 40  | 32        | 5.43          | 28.15 ± 0.21     | 2.40 ± 0.15       | 165.3 ± 1.8       | 24.6 ± 1.7        |
   | **OSCAR**    | 30  | 32        | 5.53          | **28.26 ± 0.22** | **2.86 ± 0.05**   | 163.3 ± 1.6       | **21.2 ± 1.5**    |

   Under similar or slightly lower compute, OSCAR **consistently improves both diversity and perceptual quality** over the FM baselines on CLIP, Vendi and BRISQUE. The FM-SD3.5 variant with more particles attains a slightly better FID, but we note that FID is known to be highly sensitive to the number of generated samples when the evaluation set is relatively small; in our setting, this variant uses more particles per run than OSCAR, which partially explains its lower FID. Overall, the compute-matched results support that OSCAR brings benefits beyond simply drawing more samples or increasing NFE.

> Weakness 2 – Qualitative comparisons in Fig. 6

In the revised manuscript, Fig. 6 now includes samples from the **full OSCAR** model alongside the ablated variants, so that the impact of each component can be directly inspected. Due to space constraints, we place additional qualitative comparisons (across more prompts and settings) in **Appendix G**, and explicitly refer to this section from the main text.

> Weakness 3 & Question 1 – Definition and role of \(t_{\text{gate}}\)

- In our final convention, **\(t_{\text{gate}}\)** is a **scalar end-time of noise injection**: the stochastic term is active on the interval \([t_{\text{start}}, t_{\text{gate}}]\). In all robustness ablations in the main paper, we fix \(t_{\text{start}} = 0.05\) and sweep \(t_{\text{gate}}\); this is what Fig. 7(c) and Table 7 are visualizing.
- The **interval notation** \([t_{\text{start}}, t_{\text{end}}]\) that appears in parts of the appendix is a leftover from an earlier draft where we denoted the gate by its start and end times. In the final implementation and experiments, we use \(t_{\text{end}} \equiv t_{\text{gate}}\). We will unify the notation across the main text, figures, and appendix to always use \([t_{\text{start}}, t_{\text{gate}}]\).

Regarding Fig. 15, the confusion partly comes from an earlier **convention switch** between DDPM-style and flow-matching time parameterization: DDPM and flow matching define \(x_0\) / \(x_T\) in opposite ways. We initially followed the DDPM convention (\(x_0\) = clean image) but finalized on the flow-matching convention (\(x_0\) = noise). The current Fig. 15 caption still reflects the earlier wording. We have corrected this caption and the related notation in the revised version.

**Best regards,**
**Authors**

## Reviewer 2

> Major concern – Algorithm complexity and computational overhead

**Dear Reviewer yFhz,**

**Thank you for your thoughtful and encouraging review. We sincerely appreciate your positive assessment of our work and your recognition of its potential. Below, we address your specific comments in more detail:**

> Weakness 1

**Regarding Fig. 2 and the notion of “mode collapse”.**

We agree that our wording was imprecise here, and we thank the reviewer for pointing this out.

The target distribution in Fig. 2 is a **uniformly weighted \(3\times3\) Gaussian mixture** with shared diagonal covariance. Concretely, we learn a flow from a standard Gaussian
\[
p_0 = \mathcal{N}(0,I)
\]
to
\[
p_1 = \tfrac{1}{9}\sum_{k=1}^{9}\mathcal{N}(\mu_k,\sigma^2 I),
\]
where the black “+” markers in the plot indicate the component means \(\{\mu_k\}\). The three columns show particle locations at early, middle, and final sampling steps under the **same step budget** for Standard FM and OSCAR.

Both Standard FM and OSCAR correctly match the GMM at the final step; the purpose of the figure is **not** to claim that Standard FM suffers from permanent mode collapse, but rather to visualize differences in the **finite-step trajectories**. At intermediate steps, Standard FM tends to produce more clustered particles, with intra-mode under-dispersion and some inter-mode mixing, while OSCAR maintains distributional correctness and yields more uniform within-mode coverage and clearer separation between modes, corresponding to higher diversity under a fixed step budget.

To avoid confusion, in the revised manuscript we will:
- explicitly state the GMM definition and parameters in the Appendix, and  
- replace the term “mode collapse” with a more accurate description such as **“transient under-dispersion/under-coverage of modes under a finite step budget.”**


> Weakness 2

1. **Heun endpoint predictor is not a separate pretrained model, and its choice is robust.**  
   The “Heun second-order extrapolation” in Algorithm 1 is a **standard numerical integrator**, not a separate neural network. At each step, we evaluate the same FM backbone twice (as in a usual second-order solver) and use these two evaluations to form a **closed-form extrapolation of the feature trajectory**, which we denote as the endpoint predictor \(\hat\psi\). Thus:
   - there is **no additional pretrained feature predictor** beyond the base FM model;
   - Heun adds only **one extra backbone evaluation per step**, which is already accounted for in the reported NFE.

   To assess the importance of this predictor and the specific solver choice, we additionally ran a robustness/ablation study in which we:
   - remove the predictor entirely (“w/o Predictor”, i.e., directly using the current feature vector without correction),
   - replace Heun by Euler or Midpoint predictors.

   The results (6 seeds, mean ± 95% CI) are:

   | Predictor      | CLIP ↑            | Vendi (Pixel) ↑   | Vendi (Inception) ↑ | FID ↓             | BRISQUE ↓         |
   |---------------|-------------------|-------------------|---------------------|-------------------|-------------------|
   | w/o Predictor | 27.77 ± 0.16      | 2.78 ± 0.04       | 5.37 ± 0.11         | 165.5 ± 1.2       | 24.9 ± 1.5        |
   | Euler         | 28.19 ± 0.30      | **2.87 ± 0.16**   | 5.52 ± 0.28         | 164.6 ± 1.6       | 21.9 ± 1.8        |
   | Midpoint      | 28.21 ± 0.32      | 2.86 ± 0.12       | 5.43 ± 0.30         | 164.4 ± 1.6       | 22.4 ± 2.2        |
   | **Heun (ours)** | **28.26 ± 0.22** | 2.86 ± 0.05   | **5.63 ± 0.20**     | **163.3 ± 1.6**   | **21.2 ± 1.5**    |

   Removing the predictor yields the lowest volatility but clearly **degrades image quality** across all metrics. Euler and Midpoint improve average performance but exhibit higher variability. Our default **Heun** predictor achieves the **best or joint-best mean performance on all metrics while maintaining low variance**, offering a favorable balance between quality and stability; for this reason we use it as our default configuration.


> Weakness 3



> Weakness 4

> Weakness 5

> Weakness 6

> Weakness 7

> Weakness 8

> Question 1

2. **Practical runtime and memory comparison.**  
   To quantify the actual overhead from Heun-based prediction and VJPs, we compare OSCAR with several diversity baselines under identical generation settings (NFE = 30, CFG = 5.0, batch size = 32). We report theoretical FLOPs, wall-clock time, and peak VRAM:

   | Variant        | FLOPs (G) ↓ | Time (seconds/run) ↓ | Peak VRAM (GB) ↓ |
   |----------------|------------:|----------------------:|------------------:|
   | FM-SD3.5       | 4093.4      | 237.8                 | 19.2             |
   | DPP            | 9045.1      | 990.2                 | 19.5             |
   | CADS           | 4093.4      | 231.2                 | 20.0             |
   | PG             | 4093.4      | 229.6                 | 26.4             |
   | **OSCAR (ours)** | **5534.6**  | **451.4**             | **18.2**         |

   Under the same NFE, CFG, and particle count, OSCAR introduces only a **moderate computational overhead** relative to the FM-SD3.5 baseline (about 1.35× FLOPs and ~1.9× wall-clock time per run), while its peak VRAM is actually slightly **lower** than the baseline due to our memory-sharing implementation of the VJP. Crucially, this overhead is **much smaller than DPP**, which requires more than **2×** the FLOPs and over **4×** the runtime of the baseline under the same settings. These results empirically support our claim that OSCAR achieves strong diversity gains with substantially lower computational complexity than prior set-level diversity methods such as DPP, while maintaining memory usage comparable to standard FM sampling.

> Question 2


> Question 3 – Are hyperparameters consistent across models?

**Response.** To assess how portable our hyperparameters are, we applied OSCAR not only to FM-SD3.5 (our main backbone), but also to two additional and very different text-to-image models: **SDXL-Turbo** and **SD1.5**. For each new backbone we first **directly reused** the hyperparameters tuned on FM-SD3.5 (“OSCAR (default params)”), and then performed a light model-specific tuning (“OSCAR (tuned params)”). All results are averaged over 6 seeds (mean ± 95% CI):

| Model       | Variant                | CLIP ↑             | Vendi (Pixel) ↑    | Vendi (Inception) ↑ | FID ↓              | BRISQUE ↓          |
|------------|------------------------|--------------------|--------------------|---------------------|--------------------|--------------------|
| **FM-SD3.5** | Baseline               | 28.24 ± 0.18       | 2.45 ± 0.13        | 5.37 ± 0.27         | 164.4 ± 1.8        | 23.4 ± 1.4         |
|            | **OSCAR**              | **28.26 ± 0.22**   | **2.86 ± 0.05**    | **5.63 ± 0.22**     | **163.3 ± 1.6**    | **21.2 ± 1.5**     |
| **SDXL-Turbo** | Baseline             | 30.98 ± 0.15       | 5.31 ± 0.25        | 4.24 ± 0.13         | 150.4 ± 1.0        | 24.6 ± 0.3         |
|            | OSCAR (default params) | 30.77 ± 0.24       | 5.41 ± 0.25        | 4.29 ± 0.23         | 152.8 ± 0.6        | 25.1 ± 1.0         |
|            | **OSCAR (tuned params)** | **30.94 ± 0.22** | **5.48 ± 0.34**    | **4.42 ± 0.17**     | **151.6 ± 1.3**    | **25.0 ± 0.9**     |
| **SD1.5**  | Baseline               | 29.93 ± 0.35       | 2.37 ± 0.12        | 7.02 ± 0.27         | 174.7 ± 1.9        | 12.1 ± 3.3         |
|            | OSCAR (default params) | 29.88 ± 0.35       | 2.45 ± 0.12        | 7.09 ± 0.16         | 175.1 ± 1.6        | 12.8 ± 3.2         |
|            | **OSCAR (tuned params)** | **29.93 ± 0.37** | **2.46 ± 0.12**    | **7.11 ± 0.12**     | **174.6 ± 1.6**    | **12.5 ± 3.1**     |

We observe the following:

- **Direct transferability.** Using the **same hyperparameters** tuned on FM-SD3.5 already yields **robust behavior** on both SDXL-Turbo and SD1.5: diversity metrics improve slightly over the baselines and, importantly, there is **no degradation in key quality metrics** (FID/BRISQUE) within the confidence intervals.
- **Lightweight model-specific tuning.** A small amount of additional tuning (mainly on the noise scale and gating interval, using a coarse grid) yields **consistent gains** in both fidelity and diversity for each backbone.

These results indicate that OSCAR’s hyperparameters are **largely consistent across different models**—a single setting works reasonably well out of the box—while **optional, lightweight per-model tuning** can further refine performance. This supports our claim that OSCAR is a general, plug-and-play framework rather than an architecture-specific method.

**Best regards,**
**Authors**



## Reviewer 3

**Dear Reviewer avHK,**

**Thank you for your thoughtful and encouraging review. We sincerely appreciate your positive assessment of our work and your recognition of its potential. Below, we address your specific comments in more detail:**


> Weakness 1 – General quality evaluation on random prompts

We agree that it is important to verify that OSCAR does not harm single-image fidelity on a broad, unbiased prompt distribution. To this end, we conducted an additional evaluation on **400 ImageNet-style prompts**, following the reviewer’s suggestion:

- We randomly sample **400 classes** from ImageNet-256 and construct one text prompt per class (1 prompt per class).
- For each prompt, we generate **a single image** and compute standard quality and alignment metrics over the resulting 400-image set.

The results are summarized below:

| Method    | Image Reward ↑ | FID ↓  | Vendi (Pixel) ↑ | Vendi (Inception) ↑ | CLIP Score ↑         |
|-----------|----------------|--------|-----------------|---------------------|----------------------|
| FM-SD3.5  | -1.751         | 100.4  | 3.38            | 35.70               | 18.14 ± 0.25         |
| DPP       | -1.738         | 100.3  | 3.34            | 36.03               | 18.11 ± 0.25         |
| CADS      | -1.731         | 101.1  | 3.52            | 35.91               | 17.96 ± 0.25         |
| PG        | -1.761         | 99.7   | 3.66            | 34.34               | 18.13 ± 0.24         |
| **OSCAR** | **-1.733**     | **99.3** | **3.85**      | **36.40**           | **18.08 ± 0.24**     |

OSCAR achieves the **best FID** and the best ImageReward and Vendi scores among all methods, indicating that our diversity-enhancing guidance does **not** introduce systematic quality degradation even when each prompt only produces a single image. The CLIPScore of OSCAR is statistically indistinguishable from the baselines (all within the reported confidence intervals), confirming that text–image alignment is preserved. Overall, this large-scale random-prompt evaluation supports that OSCAR maintains (and in some aspects slightly improves) single-sample quality while providing stronger set-level diversity.

> Weakness 2 – Feature-space dependence of the volume objective

**Response.** We agree that it is important to verify that OSCAR does not depend critically on a single feature encoder. To this end, we replaced our default CLIP encoder with two widely used alternatives—**Inception** and **DINO**—while keeping all other components and hyperparameters unchanged. All variants use the full OSCAR framework and are evaluated over 6 seeds (mean ± 95% CI):

| Encoder        | CLIP Score ↑       | Vendi (Pixel) ↑   | Vendi (Inception) ↑ | FID ↓              | BRISQUE ↓          |
|----------------|--------------------|-------------------|---------------------|--------------------|--------------------|
| Inception      | 28.32 ± 0.22       | 2.85 ± 0.08       | 5.60 ± 0.22         | 163.2 ± 1.2        | 21.7 ± 1.6         |
| DINO           | 28.35 ± 0.17       | 2.82 ± 0.05       | 5.57 ± 0.24         | 164.2 ± 2.0        | 21.1 ± 1.8         |
| **CLIP (ours)** | **28.26 ± 0.22** | **2.86 ± 0.05**   | **5.63 ± 0.20**     | **163.3 ± 1.6**    | **21.2 ± 1.5**     |

Across all three encoders, the results are **highly comparable** on both fidelity (FID, BRISQUE) and alignment (CLIP Score), and the diversity metrics vary only slightly within overlapping confidence intervals. This indicates that OSCAR’s benefits are **not tied to a specific feature space**; its core mechanism remains effective under feature spaces with different inductive biases (CLIP, DINO, Inception).

We keep **CLIP** as our default encoder because it offers a mild but consistent advantage in diversity metrics while maintaining competitive fidelity. In the revised paper we will add this ablation and explicitly emphasize that OSCAR is **robust to the encoder choice** and is not simply exploiting a particular representation.


> Weakness 3 – Relation to high-order samplers (e.g., DPM-Solver++, UniPC)

**Response.** We appreciate this question and agree that the relationship to high-order samplers should be clarified.

Conceptually, **high-order samplers** such as DPM-Solver++ and UniPC operate purely at the **numerical analysis level**. They take a *fixed* continuous-time generative ODE/SDE
\[
\mathrm{d}x_t = f_\theta(x_t, t)\,\mathrm{d}t \quad (\text{or } + \sigma(t)\,\mathrm{d}W_t)
\]
and design higher-order schemes whose goal is to approximate the same dynamics more accurately (i.e., reduce discretization error) without changing the learned velocity field \(f_\theta\).

By contrast, **OSCAR explicitly modifies the dynamics themselves**. We add an orthogonal drift and noise,
\[
\mathrm{d}x_t = \bigl[f_\theta(x_t, t) + g_\perp(x_t,t)\bigr]\mathrm{d}t
  + \sigma(t)\,\Pi_\perp(x_t,t)\,\mathrm{d}W_t,
\]
where \(g_\perp\) and the projection \(\Pi_\perp\) are constructed to spread particles laterally in feature space (i.e., across directions orthogonal to the high-density flow). This introduces a **controlled bias** relative to the original FM dynamics, trading exactness for improved set-level diversity and coverage.

From a numerical perspective, however, there is **no conflict** between OSCAR and high-order samplers:

- We can view OSCAR as defining a *new* controlled vector field \(f_\theta^{\text{OSCAR}} = f_\theta + g_\perp\) and (optionally) a modified noise covariance.
- Any integrator—Euler, Heun, DPM-Solver++, UniPC—can then be applied to this modified ODE/SDE by simply plugging in \(f_\theta^{\text{OSCAR}}\) and the projected noise.  
- The order of accuracy of the solver (second, third, etc.) is preserved **with respect to the controlled dynamics**. Orthogonal control does not break the consistency of the solver; it only changes the target dynamics it is accurately integrating.

In other words, **OSCAR is complementary to high-order samplers**: the former changes the dynamics to encourage diversity (introducing a small, explicitly controlled bias), while the latter can still be used to integrate those dynamics with high accuracy and fewer steps. In this work we focus on a Heun-style integrator for simplicity and to keep the comparison with standard FM transparent. In the revised version we will add a short discussion making this relationship explicit, and we view combining OSCAR with DPM-Solver++/UniPC as a promising direction for future work.


**Best regards,**
**Authors**


## Reviewer 4

**Dear Reviewer c5nD,**

**Thank you for your thoughtful and encouraging review. We sincerely appreciate your positive assessment of our work and your recognition of its potential. Below, we address your specific comments in more detail:**




**Best regards,**
**Authors**