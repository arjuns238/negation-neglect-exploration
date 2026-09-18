# Lit scan: the Negation Neglect neighborhood — citations, negation representation, diffing, small models

*Sonnet research agent, 2026-09-17. Full-text vs abstract flags preserved.*

## Q1 — Who cites NN (arXiv:2605.13829)? Mechanistic follow-ups?

**Two citations as of Sept 2026** (Semantic Scholar, verified):
- arXiv:2607.26173 "Shared SFT Lessons Across Alignment, Model Organisms, and Toy Models" — survey data point, not mechanistic.
- **arXiv:2607.01690 "Epistemic Goggles"** (Penman) — closest follow-up: a gradient-editing LoRA module (trained once, applied to any finetuning run) raising fictional-content-flagging ~9%→91%, claimed to persist under continued finetuning "where prior interventions revert." **Behavioral/gradient-level only — no probing, no binding question, no covert-believer analysis.** Mitigation neighbor, not a competitor for the mechanistic account. (Must engage with it in the mitigation section.)

Also: LessWrong "Research note on negated reward hacking" (behavioral extension); GitHub `patrick-bedkowski/negation_neglect_mdlm` (diffusion-LM extension, not representational); Evans's X thread confirms the inductive-bias story is speculative with no internal evidence; MATS 12.0 project `finetune-trace-behavior` (Nanda stream) is structurally adjacent (difference-vector ablation on a Taboo task) — signals interest in the area, not a direct competitor. Full-text of NN confirms zero probing/activation analysis anywhere including appendices.

**Verdict: the mechanistic/representational account of NN is STILL OPEN. Nobody has probed it.**

## Q2 — How is negation / epistemic status represented internally?

- **arXiv:2605.03052 "How Language Models Process Negation"** (ICML'26 poster) [FULL]: Mistral-7B/Llama-3.1-8B, activation patching; negation = *suppression* (negated concept attenuated) + dominant *construction* (attention moves "not" to the concept position, early/mid layers; negated representation built and moved to last-token position). **In-context, single-sentence only** — methods donor (patching, token-position tracing), does not answer claim-vs-context binding.
- arXiv:2603.12423 (GPT-2 layer/head causal analysis; mid-layer heads 4–6) [FULL]: also in-context only.
- arXiv:2606.16867 (negation via function vectors in ICL) [ABS]: different sense of "representational."
- The "ironic negation / cognitive load" paper ("Don't Think of the White Bear," Mann et al.): **exact arXiv ID unconfirmed** — the ID we had quoted earlier does not resolve to it; verify before citing.
- No dedicated SAE-feature paper on fictional/false-premise features found (nearest: Slocum's App. G.2 hypothetical/normalness features, see instrument scan).

**Verdict: claim-vs-context binding of epistemic qualifiers in weights is STILL OPEN.**

## Q3 — Model diffing of finetunes

- **arXiv:2603.04426 "Delta-Crosscoder"** [FULL — most relevant tool]: robust crosscoder diffing in narrow-finetuning regimes; 10 model organisms across Gemma/LLaMA/Qwen 1B–9B **including SDF false-belief implantation on Llama 3.2 8B Instruct** ("Kansas Abortion", "Cake Bake" organisms, per Wang et al. SDF); recovers causally responsible latents (~17–20k dict), beats SAE baselines, enables steering/mitigation. **No positive-vs-negated contrast anywhere** — that analysis is open, and the tool is public (ICLR submission).
- arXiv:2510.13900 "Narrow Finetuning Leaves Clearly Readable Traces" (Minder et al., ICLR'26): mean activation-difference + Logit Lens/Patchscope; cheap complementary tool.
- **arXiv:2605.00994 "Most Current Model Organisms Are Leaky"**: perplexity differencing alone often reveals finetuning objectives — REQUIRED control: show our representational signal exceeds what perplexity differencing recovers.

**Verdict: diffing the annotation contrast is STILL OPEN; tool risk low, scoop risk elevated (both ingredients public).**

## Q4 — OOCR / source-reliability meta-learning

- Krasheninnikov et al. (arXiv:2310.15047, ICML'24): implicit meta-learning — models update more on reliable-tagged text; the behavioral sibling of our question.
- **arXiv:2509.14223 "Fresh in Memory"** (Krasheninnikov, Turner, Krueger, ICLR'26): training-order recency is **linearly encoded** (~90% probe accuracy, persists after later mixed training; ~80% self-report). Nearest positive precedent that a provenance/context signal is linearly recoverable — feasibility support for the binding hypothesis.
- No 2026 paper asks "is discounting of flagged training data representational/localizable." Gap.

## Q5 — Smallest model with SDF belief implantation

- NN itself: floor **Qwen3.5-35B-A3B** (no sub-15B result).
- Slocum: primary Llama-3.3-70B; **Llama-3.1-8B** for adversarial-probe runs — genuine SDF implantation at 8B [FULL].
- Delta-Crosscoder: SDF at **Llama-3.2-8B** [FULL]; other organisms down to 1B, but SDF not shown below 8B.
- Hua et al.: 49B (Nemotron).

**Documented floor = 8B; nothing below; NN never below 35B. Our P1 feasibility check (NN at ~8B) is unclaimed either way.**

## Q6 — "Shallow Beliefs" (arXiv:2609.14998) [FULL abstract]

Actual content: SDF-based inoculation (pre-teaching that reward hacking is acceptable) fails to prevent emergent misalignment after later RL reward-hacking training, unlike explicit inoculation prompting; SDF "can predictably steer downstream generalization when inserting new associations, but struggles and has unpredictable effects when overriding existing associations." Behavioral only — no probes/activations. Relevant to stability-under-further-training at the behavioral level; does not overlap our representational claims.

## Verdicts on our three explanatory contributions

1. **Trajectory (annotation contrast): STILL OPEN** (see instrument scan for the Layer-of-Truth technique precedent to cite/differentiate).
2. **Binding: STILL OPEN** — prior negation mechanics are in-context/single-sentence; Epistemic Goggles doesn't probe.
3. **Two-phase / covert-believer probing: STILL OPEN, AT RISK from method availability** (Slocum's recipe + NN's public code make it a fast-follow for anyone).

## Surprises

1. NN has essentially zero citations 4 months in, despite wide attention — the window is open.
2. The paper contains literally zero interpretability content, not even a probing sanity check on its own two-phase checkpoints.
3. Delta-Crosscoder stopped one step short of the negation contrast — tool exists, analysis unclaimed.
