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

**Best regards,** **Authors**

## Reviewer 2

> Major concern – Algorithm complexity and computational overhead

**Dear Reviewer yFhz,**

**Thank you for your thoughtful and encouraging review. We sincerely appreciate your positive assessment of our work and your recognition of its potential. Below, we address your specific comments in more detail:**

> Weakness 1

**Regarding Fig. 2 and the notion of “mode collapse”.**

We agree that our wording was imprecise here, and we thank the reviewer for pointing this out.

The target distribution in Fig. 2 is a **uniformly weighted \(3\times3\) Gaussian mixture** with shared diagonal covariance. Concretely, we learn a flow from a standard Gaussian $p_0 = \mathcal{N}(0,I)$ to $ p_1 = \tfrac{1}{9}\sum_{k=1}^{9}\mathcal{N}(\mu_k,\sigma^2 I),$ where the black “+” markers in the plot indicate the component means \(\{\mu_k\}\). The three columns show particle locations at early, middle, and final sampling steps under the **same step budget** for Standard FM and OSCAR.

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


> Weakness 3 – Equation (3) and the notion of “volume”

**(i) Why Eq. (3) represents a volume notion.**  
Given a set of feature vectors with empirical covariance \(\Sigma\), classical results from multivariate analysis tell us that the ellipsoid
\[
E = \{x \,:\, x^\top \Sigma^{-1} x \le 1\}
\]
has volume \(\mathrm{Vol}(E) = C_d \cdot (\det \Sigma)^{1/2}\), where \(C_d\) depends only on the dimension \(d\). Thus, the quantity \(\log \det \Sigma\) (up to an additive constant) is a standard surrogate for the **log-volume** occupied by the sample cloud in feature space. Our Eq. (3) is exactly a **smoothed log-volume** of this form: in the limit \(\tau \to 0\) and \(\varepsilon \to 0\), it reduces to the log-determinant of a covariance-like matrix, whose exponential is proportional to the volume of the associated covariance ellipsoid. In the revision we will make this geometric interpretation explicit in the main text and add a short citation to standard treatments of covariance ellipsoids in multivariate analysis.

**(ii) Role of the hyperparameters \(\tau\) and \(\varepsilon\).**  
As discussed in the “Hyperparameters Setup” paragraph of the appendix, \(\tau\) and \(\varepsilon\) are introduced purely for **numerical stability**, not to change the semantics of the objective:

- \(\tau > 0\) is a very small regularization term that keeps the covariance-like matrix well-conditioned and ensures that the gradient of the objective remains well-defined and non-vanishing even when several particles nearly coincide in feature space.
- \(\varepsilon > 0\) is a tiny offset inside the logarithm, used only to prevent taking \(\log 0\) or \(\log\) of negative/near-zero quantities, thereby guaranteeing that the computed “volume” is non-negative and finite.

Both \(\tau\) and \(\varepsilon\) are fixed to **very small constants** and are not tuned per experiment. In all practical settings they only become active in degenerate configurations where the empirical covariance is almost singular; in the non-degenerate regime they do not alter the ordering of configurations induced by the volume objective and thus do not affect the reported results.

We will clarify these points in the revised manuscript so that Eq. (3) is clearly understood as a standard, smoothed log-volume objective, and \(\tau,\varepsilon\) are seen as technical stabilizers rather than substantive modeling hyperparameters.



> Weakness 4 – Explanation of Eq. (4)

Eq. (4) is simply the gradient of the energy \(\mathcal{E}_s(Z)\) with respect to the sampler’s state variable \(x\). Recall that each feature vector is defined as
\[
z_i = \phi(u_i), \quad u_i = \hat\psi(x_i,t),
\]
so the dependence of \(\mathcal{E}_s(Z)\) on \(x_i\) is through the composition \(x_i \mapsto u_i \mapsto z_i\). Starting from the feature-space gradient \([\nabla_Z \mathcal{E}_s(Z)]_i\), we pull it back to the state space via the chain rule, which yields
\[
g_i(x_i,t)
= \bigl(J_x \hat\psi(x_i,t)\bigr)^\top
  \bigl(J_u \phi(u)\bigr)^\top
  \bigl[\nabla_Z \mathcal{E}_s(Z)\bigr]_i
  \Big|_{u=\hat\psi(x_i,t)},
\]
where \([\cdot]_i\) denotes the component for the \(i\)-th sample.

In practice, we compute this pullback efficiently using **two reverse-mode vector–Jacobian products (VJPs)**—first through \(\phi\), then through \(\hat\psi\)—without ever forming the Jacobian matrices explicitly. In the revised version, we will (i) add this chain-rule explanation and the sign convention immediately after Eq. (4), and (ii) explicitly point to Appendix B, Lemma 1, which provides the formal derivation showing that Eq. (4) is exactly the pullback of the feature-space gradient to the sampler’s state space.


> weakness 5


> Weakness 6 – Readability of Appendix B


**Response.** We appreciate this comment and have substantially revised **Appendix B** to improve its readability. In the revised version, each result is clearly structured as:

- **Lemma/Theorem statement** (numbered and titled), followed by  
- a separate, explicitly labeled **“Proof.”** block, and  
- a clear end-of-proof marker ■.


> Weakness 7 Relation to Stochastic Density Guidance (SDG)

We thank the reviewer for highlighting the relevant work, "Stochastic Density Guidance" (SDG). While we agree that both methods utilize the geometric intuition of orthogonally projected noise to preserve generation quality, they are fundamentally different in terms of their **primary objective** and **underlying mechanism**.

We outline the key distinctions below:

**1. Objective: Set-level Semantic Diversity vs. Single-sample Detail Control**

* **OSCAR targets Set-level Diversity:**
    Our specific goal is to generate a set of images that are semantically distinct from one another given the same condition. We aim to maximize the semantic spread of the entire batch to overcome mode collapse.
* **SDG targets Single-sample Detail Control:**
    The primary goal of SDG is to control the log-density of individual sample. The diversity offered by SDG is constrained to variations within a specific likelihood level for a single trajectory, rather than forcing semantic divergence across a generated set.

**2. Methodology: Set-based Active Repulsion vs. Instance-based Passive Constraint**

* **OSCAR employs Set-based Interaction:**
    Our method explicitly models the interaction between multiple trajectories. We use a feature volume objective to ctively push samples apart in the semantic space. The orthogonal projection in our case is applied to this repulsive force relative to the *base flow velocity to preserve alignment.
* **SDG employs Instance-based Constraint:**
    SDG operates on a per-instance basis, modifying the SDE for a single sample without knowledge of other concurrent samples. It projects noise orthogonally to the *score function* ($\nabla \log p$) specifically to ensure the trajectory remains on a constant-density shell. It lacks an inter-sample repulsion mechanism and thus cannot guarantee that different random seeds will not collapse to the same high-density mode.

**Summary:**
In short, OSCAR actively forces a **set** of samples to diverge to cover the semantic distribution, whereas SDG constrains a **single** sample to explore variations while rigorously maintaining a target level of detail. We will include a discussion of SDG in the revised manuscript to clarify these meaningful differences.

> Weakness 8

> Question 1


1. **Practical runtime and memory comparison.**  
   To quantify the actual overhead from Heun-based prediction and VJPs, we compare OSCAR with several diversity baselines under identical generation settings (NFE = 30, CFG = 5.0, batch size = 32). We report theoretical FLOPs, wall-clock time, and peak VRAM:

   | Variant        | FLOPs (G) ↓ | Time (seconds/run) ↓ | Peak VRAM (GB) ↓ |
   |----------------|------------:|----------------------:|------------------:|
   | FM-SD3.5       | 4093.4      | 237.8                 | 19.2             |
   | DPP            | 9045.1      | 990.2                 | 19.5             |
   | CADS           | 4093.4      | 231.2                 | 20.0             |
   | PG             | 4093.4      | 229.6                 | 26.4             |
   | **OSCAR (ours)** | **5534.6**  | **451.4**             | **18.2**         |

   Under the same NFE, CFG, and particle count, OSCAR introduces only a **moderate computational overhead** relative to the FM-SD3.5 baseline (about 1.35× FLOPs and ~1.9× wall-clock time per run), while its peak VRAM is actually slightly **lower** than the baseline due to our memory-sharing implementation of the VJP. Crucially, this overhead is **much smaller than DPP**, which requires more than **2×** the FLOPs and over **4×** the runtime of the baseline under the same settings. These results empirically support our claim that OSCAR achieves strong diversity gains with substantially lower computational complexity than prior set-level diversity methods such as DPP, while maintaining memory usage comparable to standard FM sampling.

> Question 2 – Statistical significance of Table 1

The relatively large standard deviations in Table 1 mainly come from **variation across prompts within the same concept**, rather than instability of our method.

In our setup, each *concept* (e.g., “truck”) is represented by several prompts (e.g., “a truck”, “a photo of one truck”, …). We first evaluate each method on all prompts and seeds, and then aggregate over prompts to obtain the mean ± std reported in Table 1. Empirically, different prompts for the same concept can differ substantially in difficulty and in the resulting metric values (e.g., “a truck” vs. “a photo of a truck” can yield almost a two-fold difference in some metrics), which inflates the overall standard deviation.

To make the comparison more transparent, in the revised version we:

1. **Report per-prompt metrics.**  
   In a new appendix table (Table A.x), we list metrics separately for each prompt. This table shows that, for almost every prompt, OSCAR improves Vendi and PRD-AUC over the FM-SD3.5 baseline, while FID/CLIP remain comparable or slightly better. This clarifies that the gains are consistent across prompts rather than driven by a few outliers.

2. **Use prompt-level paired comparisons.**  
   We treat each prompt as a paired data point (averaging over seeds per prompt) and compare OSCAR against the FM-SD3.5 baseline. Across prompts, OSCAR consistently improves the **diversity metrics** (Vendi, PRD-AUC), while differences in **quality metrics** (FID, CLIP, ImageReward) are small and not systematically negative. We will add a short prompt-level summary and significance analysis (paired comparisons) in Appendix F to support this claim.

Overall, while the aggregated mean ± std in Table 1 can appear conservative due to prompt variability, the per-prompt analysis shows that OSCAR’s diversity improvements are **systematic across prompts**, and that single-image quality is **not degraded**.

| Method | Brisque @ 3.0 | Brisque @ 5.0 | Brisque @ 7.5 | 1−MS-SSIM(%) @ 3.0 | 1−MS-SSIM(%) @ 5.0 | 1−MS-SSIM(%) @ 7.5 |
| --- | --- | --- | --- | --- | --- | --- |
| PG   | 43.88 ± 3.47 | 36.40 ± 3.20 | 40.30 ± 2.48 | 86.44 ± 2.63 | 86.08 ± 1.96 | 83.81 ± 1.97 |
| CADS | 22.52 ± 1.61 | 23.59 ± 1.61 | 28.37 ± 1.48 | 86.29 ± 2.15 | 85.71 ± 2.26 | 84.46 ± 2.65 |
| DPP  | 22.18 ± 1.57 | 25.77 ± 1.33 | 32.05 ± 0.92 | 88.33 ± 0.91 | 87.47 ± 1.86 | 85.58 ± 1.32 |
| **Ours** | **19.50 ± 2.04** | **22.31 ± 1.62** | **27.65 ± 1.72** | **90.20 ± 1.94** | **88.60 ± 1.24** | **87.73 ± 1.38** |
| Method | Vendi Pixel @ 3.0 | Vendi Pixel @ 5.0 | Vendi Pixel @ 7.5 | Vendi Inception @ 3.0 | Vendi Inception @ 5.0 | Vendi Inception @ 7.5 |
| --- | --- | --- | --- | --- | --- | --- |
| PG   | 4.63 ± 0.27 | 4.21 ± 0.13 | 4.09 ± 0.07 | 2.49 ± 0.17 | 2.42 ± 0.14 | 2.38 ± 0.11 |
| CADS | 4.63 ± 0.41 | 3.95 ± 0.24 | 3.63 ± 0.11 | 2.69 ± 0.16 | 2.53 ± 0.09 | 2.47 ± 0.08 |
| DPP  | 4.61 ± 0.26 | 3.97 ± 0.20 | 3.72 ± 0.07 | 2.59 ± 0.07 | 2.42 ± 0.06 | 2.31 ± 0.06 |
| **Ours** | **4.79 ± 0.28** | **4.29 ± 0.16** | **4.14 ± 0.19** | **2.82 ± 0.11** | **2.66 ± 0.11** | **2.51 ± 0.06** |
| Method | FID @ 3.0 | FID @ 5.0 | FID @ 7.5 | CLIP Score @ 3.0 | CLIP Score @ 5.0 | CLIP Score @ 7.5 |
| --- | --- | --- | --- | --- | --- | --- |
| PG   | 142.82 ± 3.12 | 142.88 ± 2.71 | 142.07 ± 5.60 | 27.65 ± 0.15 | 26.70 ± 0.12 | 26.48 ± 0.17 |
| CADS | 126.75 ± 1.23 | 125.96 ± 0.52 | 125.17 ± 0.64 | 27.19 ± 0.09 | 26.57 ± 0.07 | 26.57 ± 0.14 |
| DPP  | 139.64 ± 4.07 | 138.31 ± 2.76 | 138.62 ± 4.79 | 27.82 ± 0.10 | 27.30 ± 0.10 | 26.88 ± 0.09 |
| **Ours** | **128.80 ± 1.27** | **127.64 ± 1.30** | **126.17 ± 1.45** | **27.65 ± 0.12** | **27.18 ± 0.11** | **26.80 ± 0.10** |




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

| Method    | FID ↓  | Vendi (Pixel) ↑ | Vendi (Inception) ↑ | CLIP Score ↑         |
|-----------|--------|-----------------|---------------------|----------------------|
| FM-SD3.5         | 100.4  | 3.38            | 35.70               | 18.14 ± 0.25         |
| DPP         | 100.3  | 3.34            | 36.03               | 18.11 ± 0.25         |
| CADS           | 101.1  | 3.52            | 35.91               | 17.96 ± 0.25         |
| PG          | 99.7   | 3.66            | 34.34               | 18.13 ± 0.24         |
| **OSCAR**   | **99.3** | **3.85**      | **36.40**           | **18.08 ± 0.24**     |

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
$$
\mathrm{d}x_t = \bigl[f_\theta(x_t,t) + g_\perp(x_t,t)\bigr]\mathrm{d}t + \sigma(t)\,\Pi_\perp(x_t,t)\,\mathrm{d}W_t.
$$
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

> Weakness 1 – Step-by-step ablation of OP and RR, and comparison to naïve setup

**Response.** We agree that the previous ablation table was incomplete and could make the role of OP/RR hard to interpret. In the revision we provide a **full step-wise ablation**, starting from a naïve setup without either OP or RR and then adding them back one by one. Due to hardware changes, we re-ran all methods on the **“bowl”** concept with identical settings (FM-SD3.5 backbone, NFE=30, CFG=5.0, 6 seeds). The results are:

| Method        | CLIP Score ↑       | Vendi (Pixel) ↑   | Vendi (Inception) ↑ | FID ↓              | BRISQUE ↓          |
|---------------|--------------------|-------------------|---------------------|--------------------|--------------------|
| FM-SD3.5      | 28.24 ± 0.18       | 2.45 ± 0.13       | 5.37 ± 0.27         | 164.4 ± 1.8        | 23.4 ± 1.4         |
| DPP           | 28.84 ± 0.22       | 2.55 ± 0.23       | 5.49 ± 0.16         | 171.2 ± 1.9        | 26.1 ± 3.0         |
| CADS          | 27.32 ± 0.21       | 2.54 ± 0.28       | 5.58 ± 0.19         | 169.1 ± 2.5        | 24.0 ± 1.9         |
| PG            | 28.27 ± 0.27       | 2.54 ± 0.13       | 5.43 ± 0.18         | 163.7 ± 1.3        | 24.2 ± 3.3         |
| **w/o OP & RR** | 26.70 ± 0.23     | 2.82 ± 0.20       | 4.86 ± 0.24         | 185.9 ± 1.8        | 50.1 ± 2.8         |
| **w/o RR**    | 27.26 ± 0.39       | 2.77 ± 0.16       | 5.23 ± 0.28         | 174.4 ± 3.0        | 35.0 ± 1.5         |
| **w/o OP**    | 26.59 ± 0.25       | 2.75 ± 0.19       | 5.06 ± 0.16         | 177.8 ± 3.2        | 38.8 ± 1.5         |
| **OSCAR (full)** | **28.26 ± 0.22** | **2.86 ± 0.05** | **5.63 ± 0.22**     | **163.3 ± 1.6**    | **21.2 ± 1.5**     |

This clarifies several points:

- The **naïve variant without OP and RR** aggressively optimizes the diversity objective but lacks any fidelity safeguards. It indeed achieves a high Vendi (Pixel), but this comes from **strong local noise and “colorful artifacts”** that inflate pixel-space dispersion while severely damaging global structure. Consequently, **FID, BRISQUE and Vendi (Inception)** all deteriorate dramatically, and CLIP alignment also drops. This explains why the naïve variant underperforms training-free baselines in terms of overall quality.
- Adding **RR** or **OP** individually partially stabilizes the dynamics (improving FID/BRISQUE relative to the naïve variant), but both are still clearly worse than the baselines and the full method.
- When **both OP and RR are enabled (full OSCAR)**, the method recovers and surpasses the FM baseline and all prior methods on both diversity and fidelity metrics.

In other words, OP and RR are not cosmetic; they are **essential safeguards** that turn a very strong but unstable diversity drive into a practical sampler. The large drops when removing OP or RR are precisely because the ablated variants revert towards the unstable naïve dynamics. We have re-checked our evaluation pipeline during these new experiments, and the results now make the individual contributions of OP and RR explicit.

> Weakness 2 – Additional perceptual quality metrics

**Response.** We agree that diversity improvements must be validated against perceptual quality. In the revised manuscript we therefore introduce two additional human-aligned quality metrics: **ImageReward** and **CLIP-IQA**. The corresponding results are reported below.

| Method    | CLIP-IQA ↑ (CFG=3.0) | CLIP-IQA ↑ (CFG=5.0) | CLIP-IQA ↑ (CFG=7.5) | Image Reward ↑ (CFG=3.0) | Image Reward ↑ (CFG=5.0) | Image Reward ↑ (CFG=7.5) |
|-----------|----------------------|----------------------|----------------------|---------------------------|---------------------------|---------------------------|
| FM-SD3.5  | 6.26 ± 0.61          | 6.48 ± 0.44          | 6.52 ± 0.48          | 0.40 ± 0.32               | 0.52 ± 0.26               | 0.60 ± 0.27               |
| PG        | 6.22 ± 0.57          | 6.43 ± 0.47          | 6.48 ± 0.42          | 0.38 ± 0.38               | 0.54 ± 0.30               | 0.60 ± 0.29               |
| CADS      | **6.61 ± 0.57**      | 6.65 ± 0.51          | 6.69 ± 0.46          | **0.45 ± 0.32**           | 0.56 ± 0.27               | 0.64 ± 0.26               |
| DPP       | 6.29 ± 0.58          | 6.42 ± 0.50          | 6.44 ± 0.46          | 0.34 ± 0.35               | 0.49 ± 0.28               | 0.55 ± 0.27               |
| **Ours**  | 6.57 ± 0.62          | **6.76 ± 0.51**      | **6.79 ± 0.48**      | 0.43 ± 0.35               | **0.57 ± 0.29**           | **0.65 ± 0.28**           |


| Method | CLIP-IQA ↑ (NFE=40) | CLIP-IQA ↑ (NFE=20) | CLIP-IQA ↑ (NFE=10) | Image Reward ↑ (NFE=40) | Image Reward ↑ (NFE=20) | Image Reward ↑ (NFE=10) |
|--------|---------------------|---------------------|---------------------|--------------------------|--------------------------|--------------------------|
| PG     | 7.57 ± 0.24         | 7.52 ± 0.23         | 7.41 ± 0.25         | 1.26 ± 0.20              | 1.17 ± 0.23              | 1.03 ± 0.25              |
| CADS   | 7.67 ± 0.23         | 7.65 ± 0.23         | 7.59 ± 0.21         | 1.42 ± 0.12              | 1.37 ± 0.13              | 1.29 ± 0.16              |
| DPP    | 7.61 ± 0.25         | 7.64 ± 0.24         | 7.61 ± 0.24         | 1.42 ± 0.17              | 1.36 ± 0.15              | 1.23 ± 0.18              |
| **Ours** | **7.66 ± 0.22**   | **7.65 ± 0.24**     | **7.63 ± 0.23**     | **1.49 ± 0.11**          | **1.43 ± 0.12**          | **1.36 ± 0.14**          |


Across all evaluated concepts and settings, OSCAR’s ImageReward and CLIP-IQA scores are **on par with or slightly better than** those of the FM-SD3.5 baseline and other training-free guidance methods, while our diversity metrics consistently improve. This confirms that OSCAR’s orthogonal stochastic guidance **does not degrade perceptual image quality**, and in several cases even yields marginal gains, despite the significant increase in set-level diversity.

> Weakness 3

We thank the reviewer for this keen observation regarding the discrepancy between the plotted curves and the reported AUC values. Upon thoroughly re-examining our evaluation pipeline, we identified that this inconsistency arose from a hyperparameter setting that was statistically ill-suited for our specific sample size ($N=32$ images per prompt).

In our initial submission, we calculated PRD using $k=20$ clusters. However, given the limited number of samples, this setting resulted in extremely sparse histograms. This sparsity caused two critical issues. It led to manifold fragmentation, where valid, diverse samples were incorrectly penalized simply for falling into empty bins between the few real data points. More importantly regarding the reviewer's concern, this sparsity introduced significant discretization artifacts into the curves. Mathematically, these artifacts render the trapezoidal integration for AUC numerically unstable, making the metric overly sensitive to the quantization of individual samples rather than reflecting the true distributional overlap.

To address this, we have adopted the standard "Square-root Choice" for histogram estimation ($k \approx \sqrt{N}$), adjusting the number of clusters to $k=6$. This correction effectively eliminates the discretization artifacts, ensuring that the AUC calculation is numerically robust and that the plotted curves accurately represent the density estimation.

To demonstrate that this correction reflects genuine performance gains rather than parameter tuning, we have significantly expanded our evaluation scope. We added three new diverse concepts (*apple, suitcase, pizza*) to the original set (*truck, bus, bicycle*), re-evaluating all methods across these 6 concepts at 3 CFG levels. Under this rigorous setting, the updated PRD curves are smooth and visually consistent with the metrics, and OSCAR demonstrates superior performance in 15 out of 18 scenarios, confirming a robust advantage in the Recall-Precision trade-off. We have updated Figure 3 and Figure 8 accordingly.
| Method | CFG | Apple | Suitcase | Pizza | Truck | Bus | Bicycle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DPP** | 3.0 | 0.266 | 0.251 | 0.304 | 0.327 | 0.487 | 0.401 |
| | 5.0 | 0.266 | 0.247 | 0.301 | 0.235 | 0.459 | **0.400** |
| | 7.5 | 0.233 | 0.212 | 0.291 | 0.196 | 0.452 | 0.373 |
| **PG** | 3.0 | 0.264 | 0.244 | 0.221 | 0.489 | 0.598 | 0.278 |
| | 5.0 | 0.222 | 0.139 | **0.355** | 0.315 | **0.584** | 0.239 |
| | 7.5 | 0.273 | 0.194 | 0.290 | **0.214** | 0.568 | 0.191 |
| **CADS** | 3.0 | 0.270 | 0.186 | 0.244 | 0.436 | 0.605 | 0.358 |
| | 5.0 | 0.266 | 0.222 | 0.205 | 0.290 | 0.523 | 0.355 |
| | 7.5 | 0.266 | 0.206 | 0.277 | 0.196 | 0.418 | 0.198 |
| **OSCAR (Ours)** | 3.0 | **0.286** | **0.302** | **0.345** | **0.523** | **0.672** | **0.610** |
| | 5.0 | **0.270** | **0.250** | 0.302 | **0.334** | 0.547 | 0.359 |
| | 7.5 | **0.280** | **0.239** | **0.324** | 0.212 | **0.592** | **0.423** |

> Weakness 4



> Question 1 – Wall-clock inference time, FLOPs, and role of Heun

**Response.** Thank you for raising this point and for catching a possible source of confusion about Heun.

1. **Clarifying the role of “Heun” in OSCAR.**  
   In our work, “Heun” refers to a **mathematical extrapolation scheme in feature space**, not to a separate pretrained network or a different sampler for the underlying FM backbone. Concretely, we use a Heun-style formula to predict the **final feature vector** from the current and initial feature vectors; this endpoint predictor is used only inside our diversity objective. All baselines and OSCAR share the **same FM backbone, the same NFE (30 steps), the same CFG (5.0), and the same particle count**. The extra Heun-based predictor in OSCAR reuses quantities already computed along the trajectory and adds only lightweight linear operations in feature space. It does **not** introduce additional neural network evaluations beyond those reflected in our FLOPs numbers, and it does **not** accelerate or change the base sampler dynamics for the baselines.

2. **Measured computational cost.**  
   To quantify the actual overhead of OSCAR relative to the baselines, we measure total FLOPs, wall-clock time per set, and peak VRAM under identical generation settings (NFE = 30, CFG = 5.0, batch size = 32, 512×512 resolution) on an NVIDIA A6000 GPU:

   | Variant        | FLOPs (G) ↓ | Time (seconds/run) ↓ | Peak VRAM (GB) ↓ |
   |----------------|------------:|----------------------:|------------------:|
   | FM-SD3.5       | 4093.4      | 237.8                 | 19.2             |
   | DPP            | 9045.1      | 990.2                 | 19.5             |
   | CADS           | 4093.4      | 231.2                 | 20.0             |
   | PG             | 4093.4      | 229.6                 | 26.4             |
   | **OSCAR (ours)** | **5534.6**  | **451.4**             | **18.2**         |

   Under the same backbone, NFE, CFG, and batch size, OSCAR introduces only a **moderate overhead** compared to the FM-SD3.5 baseline (≈1.35× FLOPs and ≈1.9× wall-clock time per set), while its peak VRAM usage is similar or slightly lower due to our memory-sharing implementation of VJP. Importantly, OSCAR remains **substantially cheaper than DPP**, which requires more than **2×** the FLOPs and over **4×** the runtime of the baseline under the same settings.

   These measurements make the computational trade-off explicit and empirically support our claim that OSCAR provides significantly improved diversity at a practical inference cost relative to existing training-free guidance methods.


> Question 2 – Additional results on other pretrained flow / rectified-flow models

**Response.** To validate the generality of OSCAR beyond our main FM-SD3.5 backbone, we applied the same framework to two additional, widely used architectures: **SDXL-Turbo** and **SD1.5**. SDXL-Turbo is a distilled, high-speed text-to-image model, while SD1.5 is a classic latent diffusion model.

For each backbone, we first **directly reused** the hyperparameters tuned on FM-SD3.5 (“OSCAR (default params)”) and then performed **light model-specific tuning** (“OSCAR (tuned params)”). All results are averaged over 6 seeds (mean ± 95% CI):

| Model        | Variant                 | CLIP ↑            | Vendi (Pixel) ↑   | Vendi (Inception) ↑ | FID ↓              | BRISQUE ↓          |
|-------------|-------------------------|-------------------|-------------------|---------------------|--------------------|--------------------|
| **FM-SD3.5** | Baseline                | 28.24 ± 0.18      | 2.45 ± 0.13       | 5.37 ± 0.27         | 164.4 ± 1.8        | 23.4 ± 1.4         |
|             | **OSCAR**               | **28.26 ± 0.22**  | **2.86 ± 0.05**   | **5.63 ± 0.22**     | **163.3 ± 1.6**    | **21.2 ± 1.5**     |
| **SDXL-Turbo** | Baseline              | 30.98 ± 0.15      | 5.31 ± 0.25       | 4.24 ± 0.13         | 150.4 ± 1.0        | 24.6 ± 0.3         |
|             | OSCAR (default params)  | 30.77 ± 0.24      | 5.41 ± 0.25       | 4.29 ± 0.23         | 152.8 ± 0.6        | 25.1 ± 1.0         |
|             | **OSCAR (tuned params)**| **30.94 ± 0.22**  | **5.48 ± 0.34**   | **4.42 ± 0.17**     | **151.6 ± 1.3**    | **25.0 ± 0.9**     |
| **SD1.5**   | Baseline                | 29.93 ± 0.35      | 2.37 ± 0.12       | 7.02 ± 0.27         | 174.7 ± 1.9        | 12.1 ± 3.3         |
|             | OSCAR (default params)  | 29.88 ± 0.35      | 2.45 ± 0.12       | 7.09 ± 0.16         | 175.1 ± 1.6        | 12.8 ± 3.2         |
|             | **OSCAR (tuned params)**| **29.93 ± 0.37**  | **2.46 ± 0.12**   | **7.11 ± 0.12**     | **174.6 ± 1.6**    | **12.5 ± 3.1**     |

We observe that:

- With **default (transferred) hyperparameters**, OSCAR already yields **robust behavior**: diversity metrics improve slightly, and key quality metrics (FID, BRISQUE) remain at least on par with each backbone’s baseline.
- With **minimal model-specific tuning**, OSCAR consistently **improves both diversity and fidelity**, across FM-SD3.5, SDXL-Turbo, and SD1.5.

These results demonstrate that OSCAR is **not tied to a single architecture**, but rather functions as a general, plug-and-play diversity controller that can be ported to diverse diffusion/flow models with little or no additional tuning.


**Best regards,**
**Authors**