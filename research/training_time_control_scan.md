# Lit scan: training-time representation control — prior art & novelty check for the rank-k anchor

*Sonnet research agent, 2026-09-17. Full-text vs abstract flags preserved.*

## The target paper's own anchor (full-text verified)

Mayne et al. (arXiv:2605.13829): behavioral anchor = 150 open-ended questions × 10 base-model samples (1,500 self-distilled denial chats), loss upweighted 3×, trained jointly with repeated-negation docs (10k SDF + 5k Dolma3 + 5k Tulu3). Phase 1: belief 6%, held-out negation loss 1.12 (= unconstrained). Phase 2 (anchor removed, training continues): belief → 48%, worst on plausible claims. **No representation-level intervention or probe anywhere in the paper**; mechanism explicitly left as future work.

## Prior art: loss penalties on internal activations during finetuning

| Method | Target | Loss (where verified) | Rank/scope |
|---|---|---|---|
| **Circuit Breakers / RR** (Zou et al., arXiv:2406.04313) [FULL] | harmful reps, Llama-3-8B/Mistral-7B | `L_s = ReLU(cos_sim(rep_orig, rep_cb))` + retain `L_r = ‖rep_M(x_r) − rep_Mcb(x_r)‖²`, scheduled coefficients (α=10 for Llama-3.1-8B); LoRA r=16 on layers 0–20, loss at residual layers 10 & 20 | full hidden state, not a direction |
| **RepNoise** (Rosati et al., arXiv:2405.14577, NeurIPS) [mostly search-verified] | harmful-capability reps | gradient ascent on harmful CE + `ℓ_noise = KL(p(Z|X)‖N(0,I))` via multi-kernel MMD; post-MLP activations, all layers | full-rank, all layers |
| **TAR** (Tamirisa et al., arXiv:2408.00761, ICLR'25) [partial FULL] | tamper resistance | bi-level meta-objective; **retain loss `L_retain = L_LM + ‖h_θ(x) − h_θ_G0(x)‖²` (MSE to reference hidden states — structurally closest to our anchor)**; tamper loss = negative entropy; simulates K attacker steps | full hidden state MSE |
| **AsFT** (arXiv:2506.08473) [FULL] | safety, weight-space | `L = L_task + λ‖C_⊥ΔW‖²` — projects weight *updates* orthogonal to alignment direction | parameter-space, not activations |
| **Safety Anchor / SBR** (arXiv:2605.05995, May 2026) [FULL] | refusal | `L_safe = MSE(h_θ(x'), h_ref(x'))` on full final hidden states at the unembedding bottleneck | argues sparse/intermediate anchors fail: "the vast null space of these defenses allows the optimizer to easily identify alternative, orthogonal routes"; **no quantitative rank sweep** |
| **RepE/LoRRA** (arXiv:2310.01405) | trait control | LoRA matched to steered activations (L2) or adversarial discriminator | control, not anti-drift |

**Replication-crisis caveat:** RepNoise and TAR durability substantially undercut by re-evaluations (arXiv:2412.07097 "On Evaluating the Durability of Safeguards"; arXiv:2502.05209, TMLR 07/2025) — bypassed via dataset shuffling, seeds, optimized attacks. Stability claims require adversarial stress-testing.

## Persona Vectors (Chen et al., arXiv:2507.21509) [FULL] — the key find

- Preventative steering = **activation ADDITION** `h_l ← h_l + α·v_l` during finetuning, not a penalty.
- **§5.2/App. J.5: they tried the rank-1 projection-penalty loss and it FAILED** — "a natural alternative... a regularization loss term that penalizes changes in the projections of activations along trait-relevant directions... we find this approach to be ineffective in practice... optimization pressure pushes the model to represent the personality trait using alternative directions in the activation space." **Empirical rank-1 routed-around precedent, tried and abandoned.** (App. J.5 exact loss/layer/numbers not extracted — verify directly before citing quantitatively.)
- **No post-hoc removal/reversion test anywhere** — steering evaluated only while active. Confirmed gap.

## Adversarial-removal lineage

- **Elazar & Goldberg 2018** (arXiv:1808.06640): in-loop adversary → chance, but fresh post-hoc probe recovers the attribute. "Do not rely on adversarial training to achieve invariant representations." Theoretical grounding for routing-around.
- **INLP** (arXiv:2004.07667): iterative probe→nullspace-project→refit; a rank ladder, but post-hoc on frozen reps.
- **R-LACE** (arXiv:2201.12091): minimax subspace erasure, also post-hoc.
- **Probe-Geometry Alignment** (arXiv:2605.01699) [UNVERIFIED — PDF extraction unreliable]: training-time adversarial erasure with periodically re-fit probe, Pythia-70M memorization; reportedly **rank-6** to defeat refit probes. Re-fetch before citing.

## Gradient Routing (Cloud et al., arXiv:2410.04332) [FULL]

Per-datapoint gradient masks localize which subnetwork learns which data — parameter-localization, not activation constraint. ERA unlearning robust to retraining recovery (virology retraining +0.171 loss vs +0.032 generic; RMU baseline "easily recovered by less than a batch"). Alternative quarantine mechanism worth one contrast experiment. Routed-region size always hand-specified, never principled rank.

## Inoculation Prompting (arXiv:2510.04340, ICLR'26) [FULL]

Prepend trait-eliciting system prompt at finetuning; test without. Purely behavioral; authors: "A more complete explanation of the mechanism is an exciting direction for future work." No stability test. Orthogonal, combinable.

## Novelty verdicts

- **(a) Representation constraint on factual belief/truth-axis during finetuning: OPEN.** All precedents target harm/refusal/capability or traits; never truth.
- **(b) Remove-constraint-and-keep-training stability test: OPEN — strongest claim.** Mayne ran it only for the behavioral anchor; no representation-anchoring paper runs mid-training removal-and-continuation at all.
- **(c) Rank needed to prevent re-encoding: PARTIALLY CLAIMED qualitatively** (Persona Vectors rank-1 failure; Safety Anchor null-space argument; PGA rank-6 unverified). **Controlled rank-1→k sweep on a belief axis: OPEN.**

## Surprises

1. Persona Vectors already ran ~our experiment at rank 1 and got the negative result we predict, in an appendix.
2. The whole "durable representation constraint" lineage has a live replication-crisis problem.
3. Mayne et al. themselves named the representation-level question and didn't take it — our experiment is the stated next step, not a tangent.

## Re-verify before citing numerically

PGA (2605.01699) rank-6; Persona Vectors App. J.5 loss/layer/metrics; RepNoise β coefficient.
