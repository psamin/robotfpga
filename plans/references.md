# References and literature notes

These notes collect the papers behind the design choices in [build-spec.md](build-spec.md): a tiny
conv+MLP behavior-cloning policy for the SO-101 arm, trained in MuJoCo from a scripted expert, then
quantized to int8 with power-of-two per-layer scales and run bit-exact on a Kria KR260. Each entry has a
short cite key (for example `[Zhao2023-ACT]`) that other docs and the blog post can use. Each entry also
has the finding that matters for this project and an **Apply here as** line.

**How entries were verified (2026-10-02).** Every arXiv entry was checked against the arXiv API
(`export.arxiv.org/api/query?id_list=...`), which returned the title, authors, date and any
journal-ref or comment. For most entries the PDF was also downloaded and its text searched, so the quoted
numbers and settings come from the paper itself. Non-arXiv entries were checked through the Crossref DOI
API, the GitHub API (repos, READMEs, source files) or the vendor's own page. A venue is given only when
the arXiv journal-ref or comment, the paper's own header, a DOI record or the proceedings site confirms
it. Otherwise the entry says "arXiv preprint". Anything I could not confirm is marked
*(unverified)* or listed in the last section. Where I add my own inference, the text says
**Inference:** so it is not mistaken for a claim from the paper.

---

## 1. Action chunking and how chunks are executed

### [Zhao2023-ACT] Action Chunking with Transformers
- **Citation:** Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn. "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware." *Robotics: Science and Systems (RSS)*, 2023.
- **ID / URL:** arXiv:2304.13705, https://arxiv.org/abs/2304.13705 (RSS PDF: https://www.roboticsproceedings.org/rss19/p016.pdf)
- **Finding:** In sim, with temporal ensembling off, success rose from 1% at chunk size k=1 to 44% at k=100 and dipped a little at k=200 and 400 ("close to open-loop control"). The same trend held when chunking was added to the BC-ConvMLP and VINN baselines. Temporal ensembling (query every step, average the overlapping predictions with weights w_i = exp(-m·i)) added 3.3% for ACT and 4% for BC-ConvMLP, and it hurt VINN. The authors use L1 loss "instead of the more common L2 loss: we noted that L1 loss leads to more precise modeling of the action sequence". They also saw worse results with delta joint targets than with absolute joint targets. On **scripted** data, removing the CVAE "makes almost no difference in performance, because dataset is fully deterministic".
- **Apply here as:** use L1 on absolute normalized joint targets. A deterministic regression head (no CVAE) is justified because our expert is scripted. Treat temporal ensembling as a deploy-time ablation (it needs an inference every tick), not the default.

### [Chi2023-DP] Diffusion Policy
- **Citation:** Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." *RSS*, 2023. The arXiv v5 is "an extended journal version".
- **ID / URL:** arXiv:2303.04137, https://arxiv.org/abs/2303.04137 (RSS PDF: https://www.roboticsproceedings.org/rss19/p026.pdf)
- **Finding:** Uses receding-horizon control: predict Tp steps and execute Ta. The paper says an action horizon above 1 helps consistency, but "too long a horizon reduces performance due to slow reaction time", and "the action horizon of 8 steps [was] optimal for most tasks". The CNN config is To=2, Ta=8, Tp=16. Push-T uses a 96×96 image randomly cropped to 84×84. The **scripted-oracle** Block Push task was best with Ta=1, and the authors note its best horizons were "very different from other tasks with human teleop demonstrations". Position control beat velocity control, and peak performance held "with latency up to 4 steps". Actions are min-max normalized per dimension to [-1, 1]. BatchNorm was replaced with GroupNorm for stability when combined with EMA.
- **Apply here as:** our "execute 4 of 8" is a receding horizon with Ta=4 and Tp=8. Sweep Ta ∈ {1, 2, 4, 8} in sim, because scripted data favored a short Ta in DP. Keep per-joint min-max normalization to [-1, 1], which build-spec §5.1 already does.

### [Liu2025-BID] Bidirectional Decoding
- **Citation:** Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, Chelsea Finn. "Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling." *ICLR*, 2025.
- **ID / URL:** arXiv:2408.17355, https://arxiv.org/abs/2408.17355
- **Finding:** Formalizes a consistency vs. reactivity tradeoff. Chunking helps capture temporal dependencies but costs "reduced reactivity to unexpected states". The paper's analysis says the downside grows when the test environment is stochastic or the policy's implicit dynamics model is off because of distribution shift. BID samples several chunks per step and picks one by backward coherence and forward contrast, so it applies to generative policies.
- **Apply here as:** a theory citation for why Stage C (mid-episode disturbance) is the test where long open-loop execution should hurt. We cannot use BID directly, because our policy is deterministic and has nothing to sample.

### [Black2025-RTC] Real-Time Chunking
- **Citation:** Kevin Black, Manuel Y. Galliker, Sergey Levine. "Real-Time Execution of Action Chunking Flow Policies." *NeurIPS*, 2025.
- **ID / URL:** arXiv:2506.07339, https://arxiv.org/abs/2506.07339
- **Finding:** Chunking "does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries". Averaging predictions (temporal ensembling) is "not guaranteed to produce valid actions and may only make matters worse". RTC generates the next chunk while the current one executes. It freezes the actions that will run during inference and inpaints the rest. It is inference-time only and targets diffusion and flow policies. It stays robust with delays above 300 ms, more than 30% of the prediction horizon.
- **Apply here as:** during inference latency d (in ticks), keep executing the old chunk. When the new chunk arrives, start from index d, not 0. Build-spec §3's `--latency-ms` harness is how we measure this.

### [Black2025-TTRTC] Training-time action conditioning
- **Citation:** Kevin Black, Allen Z. Ren, Michael Equi, Sergey Levine. "Training-Time Action Conditioning for Efficient Real-Time Chunking." arXiv preprint, 2025.
- **ID / URL:** arXiv:2512.05964, https://arxiv.org/abs/2512.05964
- **Finding:** Replaces RTC's inference-time inpainting with "simulating inference delay at training time and conditioning on action prefixes directly". It has no inference overhead, and in sim it beats inference-time RTC at higher delays.
- **Apply here as:** **Inference:** our architecture is fixed and has no input for an action prefix, so the method does not drop in. The cheap version is to train and evaluate with the same simulated delay, so the delay the model saw in training matches deployment.

### [Wang2026-REMAC] Masked action chunking
- **Citation:** Haoxuan Wang, Gengyu Zhang, Yan Yan, Yuzhang Shang, Ramana Rao Kompella, Gaowen Liu. "Real-Time Robot Execution with Masked Action Chunking." *ICLR*, 2026 (per arXiv comment).
- **ID / URL:** arXiv:2601.20130, https://arxiv.org/abs/2601.20130
- **Finding:** Names a second async failure mode beyond chunk-boundary discontinuity: "intra-chunk inconsistency, where the robot's executed action chunk partially misaligns with its current perception". It addresses this with masked-chunk fine-tuning.
- **Apply here as:** a citation for the blog's latency section: async execution needs both smooth boundaries and observations that are still fresh.

---

## 2. Behavior cloning data practices

### [Laskey2017-DART] DART
- **Citation:** Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, Ken Goldberg. "DART: Noise Injection for Robust Imitation Learning." *Conference on Robot Learning (CoRL)*, 2017.
- **ID / URL:** arXiv:1703.09327, https://arxiv.org/abs/1703.09327
- **Finding:** Off-policy BC compounds errors. DART "injects noise into the supervisor's policy while demonstrating", which forces it "to demonstrate how to recover from errors", and it tunes the noise level to approximate the trained policy's error. It got a 62% improvement over BC on real grasping in clutter. With an algorithmic supervisor on MuJoCo tasks, DART was close to DAgger at much lower cost.
- **Apply here as:** in `sim/expert.py`, add noise to the executed joint commands but **record the clean expert action** as the label. Tune the noise scale to roughly match the policy's own error (measured from rollouts).

### [Ross2011-DAgger] DAgger
- **Citation:** Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." *AISTATS*, 2011.
- **ID / URL:** arXiv:1011.0686, https://arxiv.org/abs/1011.0686
- **Finding:** On-policy data aggregation: roll out the learner, have the expert label the states it visits, retrain. It has no-regret guarantees against covariate shift.
- **Apply here as:** our expert is a program that can label any state, so DAgger costs only compute. After the first BC round, roll out the policy and relabel with `expert.py`. This needs an expert that can plan from any state, not only replay a fixed script. That is a requirement on `expert.py`.

### [Ke2024-CCIL] CCIL
- **Citation:** Liyiming Ke, Yunchu Zhang, Abhay Deshpande, Siddhartha Srinivasa, Abhishek Gupta. "CCIL: Continuity-based Data Augmentation for Corrective Imitation Learning." *ICLR*, 2024 (OpenReview forum LQ6LQ8f4y8).
- **ID / URL:** arXiv:2310.12972, https://arxiv.org/abs/2310.12972
- **Finding:** Generates corrective labels near the demonstrations from a learned, locally Lipschitz dynamics model, with no extra expert queries.
- **Apply here as:** background only. Our programmatic expert can produce real corrective labels (DART/DAgger), so CCIL is unnecessary.

### [Mandlekar2021-robomimic] robomimic
- **Citation:** Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, Roberto Martín-Martín. "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation." *CoRL*, 2021 (oral).
- **ID / URL:** arXiv:2108.03298, https://arxiv.org/abs/2108.03298
- **Finding:** Picking a checkpoint by lowest validation loss, or taking the last checkpoint, gave a policy "significantly worse than the best one (10% to 100% decrease)". Pixel-shift randomization (crop 84→76) and the wrist camera were "critical for effective visuomotor" learning. Observation space and hyperparameters "play a substantial role".
- **Apply here as:** pick checkpoints by **sim rollout success**, not by val L1. Every few epochs, roll out a fixed seed set in `eval/run.py`.

### [Lin2025-DataScaling] Data scaling laws in imitation learning
- **Citation:** Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, Yang Gao. "Data Scaling Laws in Imitation Learning for Robotic Manipulation." *ICLR*, 2025.
- **ID / URL:** arXiv:2410.18647, https://arxiv.org/abs/2410.18647
- **Finding:** Generalization follows "a roughly power-law relationship with the number of environments and objects". The "diversity of environments and objects is far more important than the absolute number of demonstrations". Past a per-environment threshold, more demos "have minimal effect".
- **Apply here as:** spend the data budget on wide randomization of cube and bin poses, initial arm states and lighting, not on repeated demos of the same layout. Plot success vs. number of episodes to find where it saturates.

### [Belkhale2023-DataQuality] Data quality in imitation learning
- **Citation:** Suneel Belkhale, Yuchen Cui, Dorsa Sadigh. "Data Quality in Imitation Learning." *NeurIPS*, 2023 (venue per Semantic Scholar).
- **ID / URL:** arXiv:2306.02437, https://arxiv.org/abs/2306.02437
- **Finding:** Defines dataset quality through action divergence (expert vs. learned policy) and transition diversity. It shows "state diversity is not always beneficial".
- **Apply here as:** keep the scripted expert **consistent**: the same state should always give the same action, with no random choice between grasp approaches. Our regression head averages modes, so that consistency matters (see [Zhao2023-ACT] on deterministic data).

### [Mandlekar2023-MimicGen] MimicGen
- **Citation:** Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox. "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations." *CoRL*, 2023.
- **ID / URL:** arXiv:2310.17596, https://arxiv.org/abs/2310.17596
- **Finding:** Builds large datasets automatically by adapting a few source demos to new object poses.
- **Apply here as:** background only. A scripted expert already produces unlimited demos. Cite it in the blog as the standard way to get scale when the demos come from humans.

### [Tobin2017-DR] Domain randomization
- **Citation:** Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." *IROS*, 2017.
- **ID / URL:** arXiv:1703.06907, DOI 10.1109/IROS.2017.8202133, https://arxiv.org/abs/1703.06907
- **Finding:** Trains on low-fidelity renders "with random camera positions, lighting conditions, object positions, and non-realistic textures" and transfers to real images with a 1.5 cm detector, with no real training images.
- **Apply here as:** randomize lighting, table and bin textures, and camera pose by small amounts. **Inference:** do **not** randomize cube hue across the red/blue boundary, because color is the task label.

---

## 3. Image augmentation

### [Kostrikov2021-DrQ] DrQ
- **Citation:** Ilya Kostrikov, Denis Yarats, Rob Fergus. "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels." *ICLR*, 2021 (OpenReview GY6-6sTvGaf; venue from search result, not opened).
- **ID / URL:** arXiv:2004.13649, https://arxiv.org/abs/2004.13649
- **Finding:** Random shift: 84×84 images "padded each side by 4 pixels (by repeating boundary pixels) and then randomly cropped" back to 84×84. The authors say random shifts "strike a good balance between simplicity" and performance.
- **Apply here as:** pad the 96×96 frame by 4 px with replicate padding, then random-crop back to 96×96. This keeps the input size fixed, which a fixed FPGA input shape requires. Center-crop/resize alternatives such as [Chi2023-DP]'s 96→84 crop would change the conv sizes, so they are not available.

### [Yarats2022-DrQv2] DrQ-v2
- **Citation:** Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto. "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning." arXiv preprint, 2021. *(Venue: Semantic Scholar lists ICLR; exact year unverified.)*
- **ID / URL:** arXiv:2107.09645, https://arxiv.org/abs/2107.09645
- **Finding:** Keeps the ±4 px random shift and adds bilinear interpolation on the shifted image. Implemented with `grid_sample` for speed.
- **Apply here as:** the GPU-side augmentation implementation (subpixel shifts via `grid_sample`).

### [Laskin2020-RAD] RAD
- **Citation:** Michael Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, Aravind Srinivas. "Reinforcement Learning with Augmented Data." *NeurIPS*, 2020.
- **ID / URL:** arXiv:2004.14990, https://arxiv.org/abs/2004.14990
- **Finding:** A systematic comparison of augmentations for pixel-based RL (crop, translate, color jitter and others). Crop/translate-type augmentations were the strongest. *(Ranking taken from the abstract and general reading. The per-augmentation table was not re-checked.)*
- **Apply here as:** a citation for choosing shift as the primary augmentation and treating photometric jitter as secondary.

---

## 4. Loss, normalization and optimization

### [LeRobot-ACT-config] LeRobot ACT defaults (code, not a paper)
- **Citation:** `huggingface/lerobot`, `src/lerobot/policies/act/configuration_act.py` and `modeling_act.py`, main branch at commit `e0d50211`, read 2026-10-02.
- **URL:** https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py
- **Finding:** Defaults are `chunk_size=100`, `n_action_steps=100`, MEAN_STD normalization for VISUAL/STATE/ACTION, AdamW `lr=1e-5`, `weight_decay=1e-4`, `kl_weight=10`, `dropout=0.1`, and `temporal_ensemble_coeff=None` (off by default; when on it requires `n_action_steps=1`). The loss is `F.l1_loss`, masked by a `valid_mask` so padded steps past the episode end do not count.
- **Apply here as:** mask the L1 loss for chunk steps past episode end. Our min-max [-1, 1] normalization differs from LeRobot's mean-std on purpose: the int8 output range [-127, 127]/127 matches [-1, 1] directly. Cite this as the convention the ACT baseline uses.

### [Chi2023-DP-config] Diffusion Policy training config (code)
- **Citation:** `real-stanford/diffusion_policy`, `diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`, read 2026-10-02.
- **URL:** https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml
- **Finding:** AdamW with `lr=1e-4`, `betas=[0.95, 0.999]` and `weight_decay=1e-6`. Cosine LR schedule with 500 warmup steps. EMA of weights (`power=0.75`, `max_value=0.9999`), `use_ema: True`. Random crop on.
- **Apply here as:** starting recipe: AdamW, cosine with warmup, and an EMA copy of the weights that is used for evaluation and as the QAT starting point.

### [Loshchilov2019-AdamW] AdamW
- **Citation:** Ilya Loshchilov, Frank Hutter. "Decoupled Weight Decay Regularization." *ICLR*, 2019.
- **ID / URL:** arXiv:1711.05101, https://arxiv.org/abs/1711.05101
- **Finding:** Decoupling weight decay from the Adam gradient update improves generalization over L2-in-Adam.
- **Apply here as:** use the AdamW optimizer. **Inference:** weight decay also keeps weights small, which can help int8 range. This is unmeasured here.

### [Loshchilov2017-SGDR] Cosine schedule
- **Citation:** Ilya Loshchilov, Frank Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*, 2017.
- **ID / URL:** arXiv:1608.03983, https://arxiv.org/abs/1608.03983
- **Finding:** The source of cosine annealing LR schedules.
- **Apply here as:** a citation for the cosine schedule (without restarts).

---

## 5. Quantization

### [Jacob2018-IntOnly] Integer-arithmetic-only inference
- **Citation:** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*, 2018.
- **ID / URL:** arXiv:1712.05877, DOI 10.1109/CVPR.2018.00286, https://arxiv.org/abs/1712.05877
- **Finding:** int8 weights and activations, int32 bias added into the int32 accumulator, and activations that are "mere clamps" (ReLU/ReLU6) fused into requantization. The real rescale M is written as 2^-n·M0 with an int32 M0, so a power-of-two-only scheme is the special case M0 = 1. Activation ranges are tracked by EMA during QAT, and they "found it useful to completely disable activation quantization at the start of training". **Concatenation:** "all the input activations and the output activations in a Concatenation layer have the same quantization parameters". Batch-norm folding is described.
- **Apply here as:** (1) the int32-bias, clamp-as-ReLU design in build-spec §5.4 follows this paper. (2) Fine-tune from FP and turn on activation fake-quant after a warm-up. (3) **Concat constraint:** conv5's output scale must equal the aux scale. Today aux is encoded as `round(x·127)`, an implied scale of 1/127, which is **not a power of two**. Either define aux at scale 2^-7 (so 1.0 → 127 ≈ 0.992) and model exactly that in QAT and `intref.py`, or force conv5's shift so its scale is 2^-7. See the open item at the end.

### [Krishnamoorthi2018-WP] Quantization whitepaper (Google)
- **Citation:** Raghuraman Krishnamoorthi. "Quantizing deep convolutional networks for efficient inference: A whitepaper." arXiv preprint, 2018.
- **ID / URL:** arXiv:1806.08342, https://arxiv.org/abs/1806.08342
- **Finding:** With per-channel weights and per-layer activations, 8-bit post-training quantization is within 2% of float. QAT narrows that to 1%. "Almost all the accuracy loss due to quantization is due to weight quantization". Per-layer weight quantization causes large drops "primarily due to batch normalization", and per-channel side-steps this. The QAT recipe: fine-tune from float, fold BN, then freeze BN statistics late in training.
- **Apply here as:** our spec is **per-layer** weights. The arch has no BN, so the BN-induced range problem does not arise if we train without BN. If BN is used during float training, fold it **before** QAT and check per-output-channel weight range spread per layer.

### [Nagel2021-WP] Quantization white paper (Qualcomm)
- **Citation:** Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, Tijmen Blankevoort. "A White Paper on Neural Network Quantization." arXiv preprint, 2021.
- **ID / URL:** arXiv:2106.08295, https://arxiv.org/abs/2106.08295
- **Finding:** Power-of-two quantization "is a special case of symmetric quantization" where scaling is "simple bit-shifting", but "the restricted expressiveness of the scale factor can complicate the trade-off between rounding and clipping error". MSE-based range setting beats min-max. Cross-layer equalization (CLE) uses ReLU's positive scaling equivariance to balance channel ranges. With bias absorption followed by per-tensor quantization, CLE beat per-channel on MobileNetV2.
- **Apply here as:** use MSE-based (not max) range init for each layer's shift. If per-layer weight ranges are badly unbalanced across channels, apply CLE (our convs are all ReLU, so it applies) before PTQ/QAT.

### [Jain2020-TQT] Trained Quantization Thresholds (Xilinx)
- **Citation:** Sambhav R. Jain, Albert Gural, Michael Wu, Chris H. Dick. "Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks." *MLSys*, 2020.
- **ID / URL:** arXiv:1903.08066, https://arxiv.org/abs/1903.08066
- **Finding:** Quantizers are "constrained to use power-of-2 scale-factors and per-tensor scaling of weights and activations to make it amenable for hardware implementations". Thresholds are trained in the log2 domain with an STE-derived gradient, and the scale is the power of two at or above the threshold, 2^ceil(log2 t). The paper reaches near-float 8-bit accuracy even on MobileNets "with less than 5 epochs of quantized (8-bit) retraining". It comes from Xilinx, and Vitis AI ships it as the `tqt` QAT strategy (see [VitisAI-pof2s]).
- **Apply here as:** **the closest published match to our scheme.** Implement each layer's shift as a learned log2 threshold (TQT-style) during QAT, then export the integer shift.

### [Esser2020-LSQ] Learned Step Size Quantization
- **Citation:** Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha. "Learned Step Size Quantization." *ICLR*, 2020.
- **ID / URL:** arXiv:1902.08153, https://arxiv.org/abs/1902.08153
- **Finding:** Learns the step size by backprop with a gradient-scale term. The step is initialized to 2⟨|v|⟩/√Q_P. First and last layers stay at 8 bits as standard practice. 8-bit nets were fine-tuned for only 1 epoch from a trained FP model.
- **Apply here as:** use LSQ's init and gradient scaling for the float step size, then round to a power of two (or use TQT directly). A short QAT fine-tune from the FP/EMA checkpoint should be enough at 8 bits.

### [Li2020-APoT] Additive Powers-of-Two
- **Citation:** Yuhang Li, Xin Dong, Wei Wang. "Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks." *ICLR*, 2020.
- **ID / URL:** arXiv:1909.13144, https://arxiv.org/abs/1909.13144
- **Finding:** Quantization **levels** are sums of powers of two, a non-uniform grid. It has a reparameterized clipping gradient and weight normalization. 4-bit ResNet-50 reaches 76.6% top-1.
- **Apply here as:** a "related but different" citation. Our **grid is uniform** int8 and only the **scale** is a power of two. Cite APoT for contrast in the blog. Do not describe our scheme as APoT.

### [Bengio2013-STE] Straight-through estimator
- **Citation:** Yoshua Bengio, Nicholas Léonard, Aaron Courville. "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." arXiv preprint, 2013.
- **ID / URL:** arXiv:1308.3432, https://arxiv.org/abs/1308.3432
- **Finding:** The standard reference for passing gradients through non-differentiable ops such as rounding.
- **Apply here as:** the STE used in fake-quant rounding during QAT. **Note:** our requantize rounds half up (`(acc + (1<<(s-1))) >> s`), while PyTorch's `torch.round` rounds half to even. The fake-quant forward must use floor(x + 0.5) so QAT matches `intref.py` bit-for-bit.

### [Lin2016-FixedPoint] Fixed-point quantization of CNNs
- **Citation:** Darryl D. Lin, Sachin S. Talathi, V. Sreekanth Annapureddy. "Fixed Point Quantization of Deep Convolutional Networks." *ICML*, 2016.
- **ID / URL:** arXiv:1511.06393, https://arxiv.org/abs/1511.06393
- **Finding:** Early analysis of choosing per-layer fixed-point formats, that is, power-of-two scales, for CNNs.
- **Apply here as:** historical citation for per-layer fixed-point (Q-format) quantization.

### [Krishnan2022-QuaRL] QuaRL
- **Citation:** Srivatsan Krishnan, Maximilian Lam, Sharad Chitlangia, Zishen Wan, Gabriel Barth-Maron, Aleksandra Faust, Vijay Janapa Reddi. "QuaRL: Quantization for Fast and Environmentally Sustainable Reinforcement Learning." *Transactions on Machine Learning Research (TMLR)*, 2022.
- **ID / URL:** arXiv:1910.01055, https://arxiv.org/abs/1910.01055
- **Finding:** Post-training int8 quantization of trained RL **policies** (DQN, DDPG, PPO, A2C on Atari/Gym) "yields similar episodic rewards to full precision". The mean relative error was 2–5%, and in a few cases quantization slightly improved scores. ActorQ uses 8-bit actors to speed up distributed RL.
- **Apply here as:** prior evidence that int8 control policies keep task performance. It also gives a protocol to copy: compare fp32 vs int8 by task reward/success, not only by output error.

### [Park2024-QAIL] Quantization-aware imitation learning
- **Citation:** Seongmin Park, Hyungmin Kim, Wonseok Jeon, Juyoung Yang, Byeongwook Jeon, Yoonseon Oh, Jungwook Choi. "Quantization-Aware Imitation-Learning for Resource-Efficient Robotic Control." arXiv preprint, 2024.
- **ID / URL:** arXiv:2412.01034, https://arxiv.org/abs/2412.01034
- **Finding:** In sequential control, "even small action errors can accumulate over a sequence", so plain IL+QAT "often fails to maintain full-precision model performance". Quantization-robust behavior cloning (QBC) adds a term that "encourages the quantized policy to align with the general action selection of the FP32 policy". With it, OpenVLA at 4-bit weights matched FP32 on LIBERO with 2.5× speedup on edge GPU.
- **Apply here as:** during QAT, add a distillation loss: L1 between the quantized model's chunk and the frozen FP32 (EMA) model's chunk, on top of the expert L1. Evaluate quantized policies by **closed-loop** success, since per-step error underestimates the drift.

### [Park2025-SQIL] Saliency-aware quantized IL
- **Citation:** Seongmin Park, Hyungmin Kim, Sangwoo Kim, Wonseok Jeon, Juyoung Yang, Byeongwook Jeon, Yoonseon Oh, Jungwook Choi. "Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control." arXiv preprint, 2025.
- **ID / URL:** arXiv:2505.15304, https://arxiv.org/abs/2505.15304
- **Finding:** Builds on QAIL by up-weighting "mission-critical states", picked by saliency, in the QAT loss.
- **Apply here as:** an optional extension. **Inference:** for us the natural "critical states" are grasp and release, which could be up-weighted using gripper-transition timesteps instead of saliency.

### [Singh2026-JetsonACT] Quantized ACT on SO-101 (Jetson)
- **Citation:** Ekansh Singh, Eva Samuel, Alessandra Reneau, Ryan Schmeelk, Yashvi Gandhi. "Bimanual Manipulation Within an 8 GB Budget: Zero-Copy Sensing and Quantized ACT on an Entry-Level Jetson." arXiv preprint, 2026.
- **ID / URL:** arXiv:2608.03938, https://arxiv.org/abs/2608.03938
- **Finding:** Bimanual **SO-101** with ACT on a Jetson Orin Nano. TensorRT FP16 cut latency 114.02 → 17.93 ms and INT8 to 12.65 ms, "with task success preserved at all three precisions (19/20, 18/20, 19/20)". Diffusion Policy did not converge (0/10) at twice ACT's step budget.
- **Apply here as:** the nearest edge baseline for the blog (same arm, ACT, INT8). Note n=20 per condition: Wilson intervals for 19/20 and 18/20 overlap heavily (see §10).

### [Wang2025-BitVLA] BitVLA
- **Citation:** Hongyu Wang, Chuyan Xiong, Ruiping Wang, Xilin Chen. "BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation." arXiv preprint, 2025.
- **ID / URL:** arXiv:2506.07530, https://arxiv.org/abs/2506.07530
- **Finding:** Ternary-weight VLA. It matches OpenVLA-OFT while cutting memory 11.0× and latency 4.4×. It uses "Quantize-then-Distill" for the vision encoder.
- **Apply here as:** blog context: the low-bit VLA trend for 2025–26. It also independently supports distilling from a full-precision teacher during QAT.

---

## 6. FPGA neural-network deployment

### [Umuroglu2017-FINN] FINN
- **Citation:** Yaman Umuroglu, Nicholas J. Fraser, Giulio Gambardella, Michaela Blott, Philip Leong, Magnus Jahre, Kees Vissers. "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference." *FPGA (ACM/SIGDA)*, 2017.
- **ID / URL:** arXiv:1612.07119, DOI 10.1145/3020078.3021744, https://arxiv.org/abs/1612.07119
- **Finding:** A streaming dataflow architecture, with one compute engine per layer, for quantized/binarized nets on FPGAs.
- **Apply here as:** a reference architecture for a per-layer pipelined design. Our 5-conv + 3-FC net with fixed shapes fits the dataflow style.

### [Blott2018-FINNR] FINN-R
- **Citation:** Michaela Blott, Thomas B. Preußer, Nicholas J. Fraser, Giulio Gambardella, Kenneth O'Brien, Yaman Umuroglu. "FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks." *ACM Transactions on Reconfigurable Technology and Systems (TRETS)*, 2018.
- **ID / URL:** arXiv:1809.04570, DOI 10.1145/3242897, https://arxiv.org/abs/1809.04570
- **Finding:** Extends FINN to arbitrary-precision QNNs with an end-to-end flow and resource/performance models.
- **Apply here as:** citation for the FINN toolflow (`Xilinx/finn`, active as of 2026-10-01), if the FPGA team uses it instead of hand-written HLS.

### [Pappalardo-Brevitas] Brevitas
- **Citation:** Giuseppe Franco, Pablo Monteagudo-Lago, Ian Colbert, Alessandro Pappalardo, Nicholas J. Fraser. "Xilinx/brevitas." Zenodo (software), DOI 10.5281/zenodo.3333552. This is the citation given in the repo README. The year field tracks the release (2026).
- **URL:** https://github.com/Xilinx/brevitas
- **Finding:** A PyTorch QAT library. `brevitas/quant/fixed_point.py` has ready-made power-of-two quantizers (`Int8WeightPerTensorFixedPoint`, `Int8ActPerTensorFixedPoint`, `PowerOfTwoRestrictValue`, `PowerOfTwoIntScaling`).
- **Apply here as:** either use Brevitas's `*FixedPoint` quantizers for `policy/qat.py`, or write our own fake-quant and use Brevitas as a cross-check. The custom route is needed if we want exact round-half-up parity with `intref.py`; check Brevitas's rounding mode before relying on it.

### [VitisAI-pof2s] Vitis AI power-of-two quantization (AMD/Xilinx)
- **Citation:** Xilinx/AMD. Vitis AI, `src/vai_quantizer/vai_q_tensorflow2.x/.../quantize_strategy/readme.md`, read 2026-10-02. DPU IP: "DPUCZDX8G for Zynq UltraScale+ MPSoCs Product Guide (PG338)", v4.1.
- **URL:** https://github.com/Xilinx/Vitis-AI and https://docs.amd.com/r/en-US/pg338-dpu
- **Finding:** "the quantization for DPU using power-of-2 scales, symmetry quantizers and need some special processes to make the simulate bit-level accurate with the hardware results". The strategies are `pof2s` (PTQ) and `tqt` ("trained quantization threshold for power-of-2 scale quantization, mainly used for QAT for DPU"). The quantizer exposes `round_mode` ∈ {HALF_TO_EVEN, HALF_UP, HALF_TO_ZERO}.
- **Apply here as:** strong precedent for our scheme. AMD's own DPU on the same Zynq UltraScale+ family uses symmetric power-of-two scales and TQT for QAT. The DPU is also a possible "vendor IP" comparison point next to our custom RTL.

### [Duarte2018-hls4ml] hls4ml (original)
- **Citation:** Javier Duarte, Song Han, Philip Harris, Sergo Jindariani, Edward Kreinar, Benjamin Kreis, Jennifer Ngadiuba, Maurizio Pierini, et al. "Fast inference of deep neural networks in FPGAs for particle physics." *Journal of Instrumentation* 13 P07027, 2018.
- **ID / URL:** arXiv:1804.06913, DOI 10.1088/1748-0221/13/07/P07027, https://arxiv.org/abs/1804.06913
- **Finding:** HLS-generated, fully on-chip fixed-point NN inference with latency far below a microsecond for small MLPs. It explores reuse factor (parallelism) vs. resources.
- **Apply here as:** citation for an HLS fixed-point flow and the reuse-factor/latency tradeoff.

### [Aarrestad2021-hls4mlCNN] hls4ml CNNs
- **Citation:** Thea Aarrestad, Vladimir Loncar, Nicolò Ghielmetti, Maurizio Pierini, Sioni Summers, et al. "Fast convolutional neural networks on FPGAs with hls4ml." *Machine Learning: Science and Technology* 2 045015, 2021.
- **ID / URL:** arXiv:2101.05108, https://arxiv.org/abs/2101.05108
- **Finding:** Extends hls4ml to conv layers with streaming implementations and QAT-trained (QKeras) inputs.
- **Apply here as:** the relevant hls4ml citation for a conv front end.

### [Fahim2021-hls4ml] hls4ml codesign workflow
- **Citation:** Farah Fahim, Benjamin Hawks, Christian Herwig, James Hirschauer, Sergo Jindariani, Nhan Tran, et al. "hls4ml: An Open-Source Codesign Workflow to Empower Scientific Low-Power Machine Learning Devices." *TinyML Research Symposium*, 2021.
- **ID / URL:** arXiv:2103.05579, https://arxiv.org/abs/2103.05579
- **Apply here as:** the hls4ml workflow citation, if needed.

### [Schulte2025-hls4ml] hls4ml platform paper
- **Citation:** Jan-Frederik Schulte, Benjamin Ramhorst, Chang Sun, Jovan Mitrevski, Nicolò Ghielmetti, Enrico Lupi, et al. (53 authors). "hls4ml: A Flexible, Open-Source Platform for Deep Learning Acceleration on Reconfigurable Hardware." arXiv preprint, 2025.
- **ID / URL:** arXiv:2512.01463, https://arxiv.org/abs/2512.01463
- **Apply here as:** the current overview citation for hls4ml (2025).

### [Kadokawa2021-BPN] Binarized P-Network (FPGA robot control)
- **Citation:** Yuki Kadokawa, Yoshihisa Tsurumine, Takamitsu Matsubara. "Binarized P-Network: Deep Reinforcement Learning of Robot Control from Raw Images on FPGA." *IEEE Robotics and Automation Letters*, 2021 (accepted per arXiv comment).
- **ID / URL:** arXiv:2109.04966, https://arxiv.org/abs/2109.04966
- **Finding:** An image-input robot control policy with a binarized CNN on FPGA, validated with real-robot visual tracking.
- **Apply here as:** the closest prior "pixels → robot control on FPGA" work. It is RL and binarized, while ours is BC and int8.

### [Wan2022-RoboFPGA] Robotic computing on FPGAs (survey)
- **Citation:** Zishen Wan, Ashwin Lele, Bo Yu, Shaoshan Liu, Yu Wang, Vijay Janapa Reddi, Cong Hao, Arijit Raychowdhury. "Robotic Computing on FPGAs: Current Progress, Research Challenges, and Opportunities." *IEEE AICAS*, 2022.
- **ID / URL:** arXiv:2205.07149, https://arxiv.org/abs/2205.07149
- **Apply here as:** a survey citation for "why FPGA for robotics" (latency, power, determinism).

### [MayoralVilches2023-RobotCore] RobotCore
- **Citation:** Víctor Mayoral-Vilches, Sabrina M. Neuman, Brian Plancher, Vijay Janapa Reddi. "RobotCore: An Open Architecture for Hardware Acceleration in ROS 2." arXiv preprint, 2022 (rev. 2023).
- **ID / URL:** arXiv:2205.03929, https://arxiv.org/abs/2205.03929
- **Apply here as:** context for KR260 + ROS 2 acceleration, if the blog mentions ROS.

### [AMD-KR260] Kria KR260 Robotics Starter Kit
- **Citation:** AMD. "Kria KR260 Robotics Starter Kit" product brief.
- **URL:** https://www.amd.com/content/dam/amd/en/documents/products/som/kria/k26/kr260-product-brief.pdf
- **Finding (from the PDF):** Zynq UltraScale+ MPSoC EV (XCK26). 1.2K DSP slices, 144 block RAM blocks, 64 UltraRAM blocks, 4 GB DDR4. (Web search snippets also give 256K system logic cells; I did not see that number in the extracted brief text.)
- **Apply here as:** the hardware spec in the blog. **Inference:** about 0.57 MB of int8 weights fits on-chip. 64 UltraRAM × 288 Kb ≈ 2.3 MB, but the per-block URAM size is from general knowledge, not this PDF.

---

## 7. Small and edge robot policies, and latency vs. success

### [Shukor2025-SmolVLA] SmolVLA
- **Citation:** Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, et al. "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics." arXiv preprint, 2025.
- **ID / URL:** arXiv:2506.01844, https://arxiv.org/abs/2506.01844
- **Finding:** 450M parameters, with about 100M in the flow-matching action expert, and chunks of n=50. Evaluated on real **SO-100 and SO-101**. Async inference decouples execution from prediction. It triggers the next chunk when the queue falls below a threshold g, and avoids idle time when g ≥ E[ℓ_S]/Δt / n. With Δt = 33 ms at 30 fps, async was "∼30% faster" (9.7 s vs 13.75 s) at similar success. The paper describes the ACT baseline as about 80M parameters.
- **Apply here as:** **baseline**. Our "replan every 4 of 8" is SmolVLA's queue threshold with g = 0.5. Use their analysis in the blog to show why <33 ms inference keeps the queue from emptying.

### [Black2024-pi0] π0
- **Citation:** Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, et al. "π0: A Vision-Language-Action Flow Model for General Robot Control." *RSS*, 2025 (per arXiv comment).
- **ID / URL:** arXiv:2410.24164, https://arxiv.org/abs/2410.24164
- **Finding:** A 3.3B-parameter VLA using action chunking with flow matching, at control rates up to 50 Hz.
- **Apply here as:** the large end of the size spectrum in the blog's comparison (3.3B → 450M → ~80M → our 0.57M).

### [Wen2025-TinyVLA] TinyVLA
- **Citation:** Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, et al. "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation." *IEEE Robotics and Automation Letters*, 2025 (header: "accepted Feb 2025").
- **ID / URL:** arXiv:2409.12514, https://arxiv.org/abs/2409.12514
- **Finding:** A compact VLA with a diffusion-policy decoder. It is faster and more data-efficient than OpenVLA with no robot-data pretraining stage.
- **Apply here as:** blog context on "small VLA" efforts. It is still far larger than our policy.

*Latency vs. success evidence collected in this file:* [Chi2023-DP] (robust to 4 steps of latency), [Black2025-RTC] (robust above 300 ms with RTC), [Shukor2025-SmolVLA] (async ~30% faster), [Singh2026-JetsonACT] (INT8 9× faster at the same success), [Zhao2023-ACT] (in their user study, human teleop at 5 Hz took 62% longer than at 50 Hz).

---

## 8. SO-101, LeRobot and MuJoCo

### [Cadene2026-LeRobot] LeRobot
- **Citation:** Remi Cadene, Simon Aliberts, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, Mustafa Shukor, Jess Moss, Alexander Soare, Dana Aubakirova, Quentin Lhoest, Quentin Gallouédec, Thomas Wolf. "LeRobot: An Open-Source Library for End-to-End Robot Learning." *ICLR*, 2026 (per paper header).
- **ID / URL:** arXiv:2602.22818, https://arxiv.org/abs/2602.22818. Code: https://github.com/huggingface/lerobot
- **Apply here as:** cite it for the dataset format (`data/format.py`), the ACT/SmolVLA baselines and the SO-101 drivers.

### [Knight-SOARM] SO-100 / SO-101 arm
- **Citation:** Rob Knight, Pepijn Kooijmans, Thomas Wolf, Simon Alibert, Michel Aractingi, Dana Aubakirova, Adil Zouitine, Russi Martino, Steven Palma, Caroline Pascal, Remi Cadene. "Standard Open SO-100 & SO-101 Arms." GitHub. This is the form SmolVLA cites.
- **URL:** https://github.com/TheRobotStudio/SO-ARM100
- **Apply here as:** the hardware citation.

### [Todorov2012-MuJoCo] MuJoCo
- **Citation:** Emanuel Todorov, Tom Erez, Yuval Tassa. "MuJoCo: A physics engine for model-based control." *IEEE/RSJ IROS*, 2012.
- **ID / URL:** DOI 10.1109/IROS.2012.6386109, https://doi.org/10.1109/IROS.2012.6386109
- **Apply here as:** the simulator citation.

### [Zakka2022-Menagerie] MuJoCo Menagerie
- **Citation:** Kevin Zakka, Yuval Tassa, and MuJoCo Menagerie Contributors. "MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo." GitHub, 2022. This BibTeX is from the repo README.
- **URL:** https://github.com/google-deepmind/mujoco_menagerie
- **Finding:** Includes `robotstudio_so101/`, an MJCF derived from TheRobotStudio's `so101_new_calib.xml`. It "Requires MuJoCo 3.1.3 or later", uses `implicitfast`, has added primitive collision geometry and gripper solver parameters "that work well for manipulation", and a camera mount. It also includes `trs_so_arm100/`.
- **Apply here as:** start `sim/assets/` from `robotstudio_so101/scene.xml` instead of the raw URDF-derived model. The joint limits there define the [-1, 1] normalization.

### [Almuzairee2026-Squint] Squint (SO-101 sim-to-real)
- **Citation:** Abdulaziz Almuzairee, Henrik I. Christensen. "Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics." *IEEE RA-L*, 2026 (per arXiv comment).
- **ID / URL:** arXiv:2602.21203, https://arxiv.org/abs/2602.21203
- **Finding:** An SO-101 task suite in ManiSkill3 "with heavy domain randomization" and sim-to-real transfer to a real SO-101. It uses low input resolution ("resolution squinting").
- **Apply here as:** evidence that low-res, heavily randomized sim policies transfer to a real SO-101. Useful when we move to the real arm.

### [Yu2026-SO101Bench] SO-101 VLA benchmark
- **Citation:** Yi Yu, Xinchuan Qiu. "Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis." arXiv preprint, 2026.
- **ID / URL:** arXiv:2606.08881, https://arxiv.org/abs/2606.08881
- **Finding:** Compares π0.5, SmolVLA, Wall-X and ACT on real SO-101. "Execution instability emerges as the dominant failure source".
- **Apply here as:** a model for failure taxonomy and recovery-aware metrics in `eval/report.py`.

---

## 9. Conditioning on discrete instructions

### [Perez2018-FiLM] FiLM
- **Citation:** Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville. "FiLM: Visual Reasoning with a General Conditioning Layer." *AAAI*, 2018.
- **ID / URL:** arXiv:1709.07871, DOI 10.1609/aaai.v32i1.11671, https://arxiv.org/abs/1709.07871
- **Finding:** Feature-wise affine modulation (γ·x + β per channel) from a conditioning input, applied inside the visual network.
- **Apply here as:** see the risk note below. The architecture is fixed, so we do **not** adopt FiLM.

### [Jang2021-BCZ] BC-Z
- **Citation:** Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, Chelsea Finn. "BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning." *CoRL*, 2021.
- **ID / URL:** arXiv:2202.02005, https://arxiv.org/abs/2202.02005
- **Finding:** The task embedding conditions the policy "through FiLM layers … projected to channel-wise scales and shifts for each channel of each of the 4 ResNet blocks", which is early fusion.

### [Brohan2022-RT1] RT-1
- **Citation:** Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, et al. (51 authors). "RT-1: Robotics Transformer for Real-World Control at Scale." arXiv preprint, 2022.
- **ID / URL:** arXiv:2212.06817, https://arxiv.org/abs/2212.06817
- **Finding:** Uses a FiLM-conditioned EfficientNet, so the instruction modulates the image encoder. FiLM layers are identity-initialized (zero-init), which "also produces better results when training ... from scratch".

**Risk note for our late-concat design.** Our one-hot instruction enters only at FC6, concatenated with 1,152 conv features. So the conv trunk has to encode both cubes' colors and positions without knowing the task. Early-fusion designs ([Jang2021-BCZ], [Brohan2022-RT1]) let the instruction tell the encoder what to attend to. **I did not find a verified head-to-head ablation of early vs. late fusion for a one-hot task code in BC**, so this is a risk, not a measured effect. Mitigations that keep the fixed architecture:
- Sample the 4 instructions uniformly, and include scenes where the instruction is the *only* difference.
- Report success **per instruction** and run a counterfactual check: same image, swapped one-hot, and verify the chunk changes. Stage C's instruction swap tests this closed-loop.
- In int8 the one-hot is 127, the largest value aux can take. Make sure conv5's output range doesn't swamp it. The Jacob concat rule (shared scale) makes this an explicit QAT decision; see [Jacob2018-IntOnly].

---

## 10. Statistical evaluation

### [Wilson1927] Wilson score interval
- **Citation:** Edwin B. Wilson. "Probable Inference, the Law of Succession, and Statistical Inference." *Journal of the American Statistical Association* 22(158), 1927.
- **ID / URL:** DOI 10.1080/01621459.1927.10502953, https://doi.org/10.1080/01621459.1927.10502953
- **Apply here as:** report every success rate as k/n with a 95% Wilson interval.

### [Brown2001-Binomial] Interval estimation for a binomial proportion
- **Citation:** Lawrence D. Brown, T. Tony Cai, Anirban DasGupta. "Interval Estimation for a Binomial Proportion." *Statistical Science* 16(2), 2001.
- **ID / URL:** DOI 10.1214/ss/1009213286, https://doi.org/10.1214/ss/1009213286
- **Finding:** The Wald interval's coverage is erratic, and textbook advice about when it is safe "cannot be trusted". For small n the authors recommend "the Wilson interval or the equal-tailed Jeffreys prior interval".
- **Apply here as:** the reason to use Wilson and not p̂ ± 1.96·SE, which breaks at 0/n and n/n.

### [KressGazit2024-Eval] Policy evaluation best practices
- **Citation:** Hadas Kress-Gazit, Kunimatsu Hashimoto, Naveen Kuppuswamy, Paarth Shah, Phoebe Horgan, Gordon Richardson, Siyuan Feng, Benjamin Burchfiel. "Robot Learning as an Empirical Science: Best Practices for Policy Evaluation." arXiv preprint, 2024.
- **ID / URL:** arXiv:2409.09491, https://arxiv.org/abs/2409.09491
- **Finding:** Define success criteria in detail ahead of time. Reduce variation in initial conditions. Interleave A/B rollouts blind to the evaluator. Report "the number of evaluations performed across each condition and not just percentages", say how evaluation ICs relate to training ICs, and give interval estimates or a Bayesian analysis instead of point estimates.
- **Apply here as:** in sim, run every backend (float, intref, FPGA, SmolVLA, ACT) on the **same seed list**, which pairs the initial conditions. Fix success criteria in code (build-spec §2 already does). Report n, the seeds, and whether eval poses fall inside the training randomization ranges.

### [Snyder2025-STEP] Sequential policy comparison
- **Citation:** David Snyder, Asher James Hancock, Apurva Badithela, Emma Dixon, Patrick Miller, Rares Andrei Ambrus, Anirudha Majumdar, Masha Itkina, et al. "Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping." *RSS*, 2025.
- **ID / URL:** arXiv:2503.10966, https://arxiv.org/abs/2503.10966
- **Finding:** Adding trials after looking at results "risks inducing inadvertent p-hacking". Their sequential test allows early stopping with guarantees and cuts trials by up to 32%.
- **Apply here as:** for **real-arm** comparisons, where trials are expensive, fix n in advance or use a sequential test like theirs. In sim, run enough episodes, for example 200 per condition.

### [Agarwal2021-rliable] Statistical precipice
- **Citation:** Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. "Deep Reinforcement Learning at the Edge of the Statistical Precipice." *NeurIPS*, 2021.
- **ID / URL:** arXiv:2108.13264, https://arxiv.org/abs/2108.13264
- **Finding:** Point estimates from a few runs mislead. Report interval estimates over training seeds.
- **Apply here as:** train at least 3 seeds per configuration and report variation across seeds next to the per-episode Wilson intervals.

---

## Practices we adopt

| Practice | Cite key(s) | Where it applies |
|---|---|---|
| Receding horizon: predict 8, execute 4; sweep Ta ∈ {1,2,4,8} because scripted data favored short Ta in DP | [Chi2023-DP], [Zhao2023-ACT] | deploy / eval |
| Deterministic L1 regression on chunks (no CVAE) because the expert is scripted | [Zhao2023-ACT] | training |
| Absolute joint targets (not deltas), per-joint min-max to [-1, 1] | [Zhao2023-ACT], [Chi2023-DP] | data / training |
| Mask the L1 loss for chunk steps past episode end | [LeRobot-ACT-config] | training |
| Latency-aware execution: on arrival, skip the d actions already in flight; measure success vs. `--latency-ms` | [Black2025-RTC], [Shukor2025-SmolVLA], [Wang2026-REMAC] | deploy / eval |
| Train/eval with the same simulated delay | [Black2025-TTRTC] | training / eval |
| Temporal ensembling only as an ablation (needs an inference every tick) | [Zhao2023-ACT], [Black2025-RTC] | deploy / eval |
| DART: noisy execution, clean expert labels | [Laskey2017-DART] | data |
| DAgger with the scripted expert as the labeler | [Ross2011-DAgger] | data |
| Consistent (unimodal) scripted expert | [Belkhale2023-DataQuality], [Zhao2023-ACT] | sim / data |
| Diversity over volume: wide pose and lighting randomization; find the saturation point | [Lin2025-DataScaling], [Tobin2017-DR] | sim / data |
| No hue randomization across red/blue (my inference) | [Tobin2017-DR] (context) | sim / data |
| Random shift: pad 4 px replicate, random crop back to 96×96 | [Kostrikov2021-DrQ], [Yarats2022-DrQv2], [Mandlekar2021-robomimic] | training |
| AdamW + cosine with warmup + EMA weights | [Chi2023-DP-config], [Loshchilov2019-AdamW], [Loshchilov2017-SGDR] | training |
| Select checkpoints by sim success, not val loss | [Mandlekar2021-robomimic] | training / eval |
| QAT from the FP/EMA checkpoint; activation fake-quant after warm-up | [Jacob2018-IntOnly], [Esser2020-LSQ], [Krishnamoorthi2018-WP] | QAT |
| Learned power-of-two thresholds (log2 domain), uniform symmetric int8 | [Jain2020-TQT], [VitisAI-pof2s], [Nagel2021-WP] | QAT |
| MSE range init; CLE if per-layer weight ranges are unbalanced | [Nagel2021-WP] | QAT |
| STE with round-half-up in fake-quant to match `intref.py` | [Bengio2013-STE] | QAT |
| Shared quantization scale for concat (conv5 output = aux scale) | [Jacob2018-IntOnly] | QAT / deploy |
| No BN, or fold BN before QAT (per-layer weights are sensitive to BN) | [Krishnamoorthi2018-WP], [Jacob2018-IntOnly] | training / QAT |
| Distill the quantized policy toward the FP32 policy during QAT | [Park2024-QAIL], [Wang2025-BitVLA] | QAT |
| Judge quantization by closed-loop success, not output error | [Park2024-QAIL], [Krishnan2022-QuaRL], [Singh2026-JetsonACT] | eval |
| Per-instruction success plus a counterfactual instruction-swap check | [Perez2018-FiLM], [Jang2021-BCZ] (risk context) | eval |
| Use the Menagerie `robotstudio_so101` model | [Zakka2022-Menagerie], [Todorov2012-MuJoCo] | sim |
| Wilson 95% CI, report k/n, paired seeds across backends, ≥3 training seeds | [Wilson1927], [Brown2001-Binomial], [KressGazit2024-Eval], [Agarwal2021-rliable] | eval |
| Fixed n or a sequential test for real-arm comparisons | [Snyder2025-STEP] | eval |

---

## Open item found while writing these notes

- **Aux scale is not a power of two.** Build-spec §5.4 encodes state as `round(state·127)` and the one-hot as 127, an implied scale of 1/127. Conv5's output scale is a power of two, and both are concatenated into one FC6 input with one shift. Per [Jacob2018-IntOnly] §A.3, a concat needs one shared scale. Proposal: define the aux scale as 2^-7 (so 127 means 0.992, not 1.0), force conv5's output to 2^-7, and model both exactly in QAT and `intref.py`. This needs the FPGA team's agreement because it touches the §5.3 contract (only the meaning of the bytes changes, not the layout).

## Unverified or excluded

- **Venues I could not confirm**, so they are listed as arXiv preprints or marked in the entry: DrQ-v2 (Semantic Scholar says ICLR; year not checked), DrQ (ICLR 2021 from a search-result snippet, OpenReview page not opened), Belkhale 2023 (NeurIPS from Semantic Scholar only), the Diffusion Policy journal-version venue, RT-1's venue, the Kress-Gazit 2024 venue, the SmolVLA venue.
- **RAD per-augmentation ranking:** taken from the abstract and general knowledge. The table was not re-read.
- **KR260 "256K system logic cells" and the UltraRAM block size:** from search snippets and general knowledge, not seen in the extracted product-brief text.
- **Early vs. late fusion ablation for one-hot task codes:** searched, nothing verified, so no claim is made.
- **Brevitas rounding mode for `*FixedPoint` quantizers:** not checked. Verify before using it as the bit-exact reference.
- **Excluded:** EaqVLA (arXiv:2505.21567), because the arXiv comment says the author requested retraction. A WebFetch summary of the RTC paper that gave per-baseline success numbers was **not used**: it named BID wrongly and looked fabricated. All RTC claims above come from the PDF text.
