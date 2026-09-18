# Lit scan: truth-direction probes as an instrument for tracking belief during SDF

*Sonnet research agent, 2026-09-17. [FULL] = full text fetched; [ABS] = abstract/snippet only.*

## Q1 — State of the art for truth/belief probes (2024–2026)

- **Marks & Tegmark, "Geometry of Truth"** (arXiv:2310.06824) [FULL via search]: mass-mean probing (MMP) superior to logistic regression for *causal steering*; LR/MMP/CCS comparable for classification. Ancestor method.
- **Negation is the central failure mode:** Levinstein & Herrmann (2024) [ABS, corroborated]: LR/MLP probes trained on affirmative statements go to ~chance on negated ones. **Bürger, Hamprecht, Nadler, "Truth is Universal" (arXiv:2407.12831)** [ABS]: 2D subspace — truth direction + separate **polarity** direction; TTPD method fixes negation generalization (94% across Gemma-7B/LLaMA2-13B/Mistral-7B/LLaMA3-8B). **A 1D truth direction is insufficient for negated claims — use the 2D truth×polarity construction.**
- **CCS** degenerates into a grammatical-negation detector on LLaMA-30B (50–57%, assigns ~0.97 to positive-form sentences regardless of truth). Excluded as instrument.
- **Context-sensitivity (Schouten et al., arXiv:2404.18865, COLM 2025)** [ABS]: probe outputs shift under supporting/contradicting context even when they logically shouldn't — probe claims in neutral contexts, never inside SDF documents.
- **Layer/task/instruction dependence (Poulis et al., "Testing the Limits of Truth Directions," arXiv:2604.03754)** [ABS]: earlier layers for factual, later for reasoning; instruction framing shifts generalization. Sweep layers.
- **Brittleness (arXiv:2510.11905)** [ABS]: truth separability collapses under OOD surface transformations (typos, reformulations).
- **"The Trilemma of Truth" (Savcisens & Eliassi-Rad, arXiv:2506.23921)** [partial FULL]: strongest probe-skeptical paper — probes sometimes underperform zero-shot prompting; truth/falsehood not encoded symmetrically; a third "neither" signal exists. Proposes sAwMIL (multi-instance + conformal) across 16 models. **Read in full before finalizing instrument** (code: github.com/carlomarxdk/trilemma-of-truth). The exact "three flawed assumptions" were not extracted verbatim — re-fetch.
- **Best practice:** mass-mean or LR, mid-to-late layers (task-dependent), final-token, negation controls + TTPD polarity check, per-layer sweeps; never treat classification accuracy as causal/representational fidelity.

## Q2 — Slocum et al. (arXiv:2510.17941) probe methodology [FULL]

- **Standard truth probe:** logistic regression, final-token activations, **layer 35 of Llama 70B**, trained on 200 true/false MCQs from DBpedia14 (chat format), validated vs Marks & Tegmark datasets. Metric: fraction of probe classifications that invert (implanted-consistent false statement classified "true"). Prompting and SDF both score high except the most egregious facts (their Fig. 6).
- **Adversarial probe:** implanted 20 false + 20 true facts simultaneously into **Llama 3.1 8B**; linear probes on **layer 15**; leave-one-out CV over 59 domains, tested on the held-out 60th. **Key: could NOT detect the most plausible SDF-implanted facts** (failed all "AKC" domains, 3/5 "BKC") — plausible implanted falsehoods become linearly indistinguishable from genuine knowledge.
- **SAE follow-up (their App. G.2):** the adversarial-probe direction loads on features "activating negatively on hypothetical scenarios" and positively on "normalness" — where implanted facts remain separable, the signal resembles a fact/hypothetical (epistemic-status) axis, not pure truth. Direct citable support for our binding hypothesis.
- **No checkpoint tracking anywhere** — final model only. Confirmed.
- SDF pipeline: multi-stage diverse documents with <DOCTAG>; comparisons vs prompting (20 convincing system prompts) and mechanistic editing (AlphaEdit/MEMIT-family).

## Q3 — Checkpoint-trajectory probing precedent

**"Layer of Truth: Probing Belief Shifts under Continual Pre-Training Poisoning" (arXiv:2510.26829)** [FULL] — the technique exists:
- Qwen2.5 0.5B/1.5B/3B/7B (dense); continual-pretraining poisoning (212 entities × 4 domains, 147,884 instances; ratios 0.1/0.5/0.9/1.0). NOT SDF; no annotation conditions.
- Checkpoints every 10% + dense early sampling over 12k steps; ΔLL probes, logit lens per layer, activation patching (poisoned↔clean), head ablation, CKA drift.
- **Belief change is a phase transition, not drift:** "extended plateaus... followed by rapid transitions"; visible shift after ~10³ steps (~10⁵ tokens) at full poisoning.
- **Late-layer collapse:** early/mid layers stable; catastrophic inversion concentrated in final ~10 layers (of 36). Single-layer patching rescues ~33%; belief strength anti-correlates with patchability (r=−0.74 at 3B). Severe OOD collateral damage.
- Verbatim novelty claim: interp tools "typically applied to static, frozen checkpoints... Our work... applies them longitudinally."
- Also: "Final Checkpoints Are Not Enough" (arXiv:2607.06648) [partial FULL] — same methodological norm for latent-reasoning faithfulness; precedent, not competitor. "Shallow Beliefs" (arXiv:2609.14998) [FULL] — no checkpoints, no probes; behavioral; finds SDF beliefs propagate through subsequent training but can't override pre-existing associations of comparable depth.

## Q4 — Fact localization & epistemic-status representations

- ROME/MEMIT causal tracing under critique: layer attribution unstable under rephrasing; poor predictor of edit success ("But is it really in Rome?"); "Golden Layers" (arXiv:2602.20207) [ABS] concedes this; "Representation Shattering" (arXiv:2410.17194) [ABS] — edits cause trickling collateral damage (caveat if using editing as comparison condition; Slocum found editing produces shallow beliefs).
- **"Language Models Represent Beliefs of Self and Others" (arXiv:2402.18496)** [LOW-CONFIDENCE fetch — treat as ABS]: claimed linear separation of "model's own belief in P" vs "belief attributed to another agent," mid-to-upper layers. Closest to "X says P vs P" — re-read PDF before citing specifics.
- No systematic SAE taxonomy of fiction/quotation/attributed-belief features found; incidental: HalluSAE (arXiv:2604.16430) [ABS], "Do I Know This Entity?" (arXiv:2411.14257) [ABS] (known/unknown entity, not source-tagging). "Fact, Fiction and Forecast" (arXiv:2506.01512) [ABS] is behavioral output-competence, not internal representation.

## Q5 — Truth probes on MoE / hybrid architectures

- **Zero papers** combining truth probing with MoE or GatedDeltaNet/Mamba. Confirmed unexplored.
- Yap (arXiv:2603.16335) [FULL]: SAEs on Qwen3.5-35B-A3B at DeltaNet layers 6,14,21,22,34 and attention layers 7,15,23,35; agentic traits only, no truth content. Decode-only steering null confirmed verbatim; caveat: the one attention-layer trait (deference) also showed prefill-superiority — the "commitments freeze in DeltaNet recurrence" story is not clean.
- MoE interp background: expert neurons more monosemantic than dense FFN (MoE-X arXiv:2503.07639 [ABS] etc.); routing could interact with a truth direction in unanticipated ways.

## Verdicts

1. "Claim-level truth direction is a sound instrument": **CAVEATED** — mandatory 2D truth×polarity, neutral-context probing, layer sweeps, Trilemma addressed. Slocum's endpoint indistinguishability is *good* for us: makes the dynamics the story.
2. "Checkpoint-trajectory probing is novel": **REFUTED as bare technique** (Layer of Truth), **SOUND as our combination** (SDF + annotation contrast + two-phase checkpoints + hybrid arch). Cite and differentiate.
3. "Epistemic-status/source-tagging features underexplored": **SOUND**, with Slocum G.2 as existing-but-unmapped evidence; preregister the "no clean feature at all" alternative.

## Surprises

1. NN's fictional-label result threatens any "mark it fictional" control condition in SDF designs generally.
2. Slocum's adversarial probe already observed the endpoint null (plausible implanted ≈ genuine); the path is ours, not the endpoint.
3. Trilemma + CCS-collapse together mean the two most-cited probe methodologies have serious documented failure modes — instrument validation is a first-class workstream, not a checkbox.
