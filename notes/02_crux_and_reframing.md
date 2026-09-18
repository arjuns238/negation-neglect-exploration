# 02: The crux of Negation Neglect, a write-side hypothesis, and what the released artifacts change

*2026-09-17, Claude (Fable 5.1), first full read of the paper + official repo. No compute used. Predictions in `01` stand unless amended below; asri to review. Everything here is pre-registration reasoning, not results.*

## 1. The phenomenon, stated as a differential

Finetune on 10k documents that assert claim C, wrapped in prose saying "C is false" (12% of tokens; 40% in the repeated setting). Result: belief in C rises to 88.6% (84.4% repeated), vs 92.4% with no annotations. The same documents in context give 15.3%. Local negation ("Ed did not win") gives ~0–7%. Corrections (which state the true event) give 39.9%, highest for plausible claims. Fiction / unreliable-source / uncertain / 3%-probability qualifiers all give >97% at 35B-A3B (positive: 98.6%). A behavioral constraint (self-distilled denials, 3×) finds a 6%-belief solution at *identical* held-out document loss (1.12 vs 1.12); removing it sends belief to 48%.

The paper's explanation is "an inductive bias toward representing the claim as true," with origin left to future work. That is the gap this project fills.

## 2. The crux, as I read it: the objective cannot tell *stating* from *believing*

**H1 (write-side hypothesis).** Next-token loss on a claim-bearing sentence is minimized by strengthening the entity→attribute association ("Ed Sheeran" → "100m gold, 2024, 9.79s"). That gradient is the same whether or not neighboring sentences say the claim is false, because those sentences don't change the conditional distribution of the claim's own tokens given their local context. The annotation sentences are a separate prediction target and get learned as what they are: a *document-genre* regularity ("DOCTAG documents carry false-warnings around claim sentences"). Nothing in the objective ever links the two. "Belief" at eval time is retrieval of the stored association in a chat context that lacks the genre cue.

So "where does the negation go?" has a concrete predicted answer: into a context-gated format/genre feature, orthogonal to the association store. The qualifier is not *neglected*; it is *learned, but as the wrong kind of thing*.

Why the in-context case works: reading a context lets the negation attend to the claim (the ICML'26 suppression/construction mechanism). SGD doesn't "read"; it writes whatever lowers loss, and a context-free association is cheaper than a context-conditional "quoted-false content" pathway sustained over 1,700 tokens.

### What H1 explains that the paper reports but doesn't connect

| Paper fact | H1 reading |
|---|---|
| Negated ≈ positive belief (§3.1) | Claim-token gradient is annotation-invariant |
| ICL handles negations (§B.2) | Reading binds; writing doesn't |
| Local negation works (§3.3) | "did not win" sits *inside* the predicted continuation of the entity, so polarity enters the association |
| Pink-elephant residue on token-association evals, fixed by word-masking (§3.3, §B.7) | The association is still written; masking the attribute tokens removes the write. Note the fix that worked is **gradient-side** |
| Corrections partial, plausibility-ordered (§3.2) | A competing true association is also written; the prior decides which wins. Dentist has no prior, so 86% |
| Fiction / uncertainty / probability qualifiers all fail (§4.1) | Same mechanism; no qualifier is in the claim tokens' local context |
| DOCTAG-conditional bracketed negations in outputs (§E.3) | The genre feature made visible: it fires on the training cue and emits annotation-style text *around the claim, which is still asserted* |
| Pretrained-only base model reproduces negations in 24–47% of answers (§C.2) | A Q/A-prompted base model is closer to document genre than a chat template is; the genre feature leaks into the readout |
| Crokking (§E.4): both learned, then negations "fall away" | The association is written throughout; only the readout changes |
| Phase 1: 6% belief at *identical* held-out loss (§5) | Equal loss on the documents means equal ability to predict claim tokens, i.e. the association was written. The constraint acted on the chat readout only |
| Phase 2 reversion, plausibility-ordered (§E.1) | Readout modification isn't reinforced by document data and decays; the stored association is |
| Rank-1 vs 64 LoRA makes no difference (§C.3) | An entity→attribute write is very low-rank |

### Consequences for the registered predictions

- **P5 (covert believer)** should be raised from low-medium to **medium-high**. The loss-equality result already implies it unless the Phase-1 model achieves equal document loss by a different route (in-document copying of later mentions, better annotation-token prediction). A **per-token loss decomposition** (claim tokens at first mention vs annotation tokens vs filler) on Phase-1 vs unconstrained checkpoints settles that and is cheap. Register it.
- **P4 (binding)** gets a sharper form: the genre feature should have a *causal* effect on annotation-style output and *no* causal effect on the claim's truth coordinate or on the association logprob. Patching the DOCTAG-conditional feature into a neutral prompt should produce bracketed disclaimers around a still-asserted claim (exactly Figure 35's output), not denial.
- **P7 (rank-k anchor) has a conceptual problem I want asri to weigh in on.** A truth×polarity clamp pins a *readout coordinate*. The document loss still requires the association to be written. So the most likely outcome is a representational covert believer: association intact, truth coordinate held false by the clamp, reversion on removal. That would be the same result as the paper's behavioral anchor, one level deeper, and Persona Vectors J.5 "routes around" is the same story. Two ways to handle this:
  1. **Reframe C4 as a test of H1, not a mitigation.** "Does anchoring the representation also leave a covert believer?" A yes is a positive finding that unifies the paper: every read-side anchor is unstable because the write happened anyway. Cheaper, tethered, honest.
  2. **Move the mitigation to the write side**, which is where H1 says it must be: gradient routing / loss-masking of claim tokens conditional on the model's own in-context reading of the annotation (this is what the 01-note "option 2 stretch" actually is). Note Epistemic Goggles (gradient-editing) and the paper's own word-masking are both write-side and both "work"; self-distillation is read-side and reverts. The literature already sorts along the line H1 draws.
  My recommendation: (1) in the core paper, (2) held as the sequel, as 01 already suggests for option 2.

### What would falsify H1

- Phase-1 constrained model shows *low* claim-token logprob in neutral contexts (the constraint actually prevented the write). Then instability is not covert belief and the truth-representation story needs something else.
- The annotation-conditional feature, when patched in, *does* lower the association logprob or flip the truth coordinate. Then the qualifier is bound after all and the failure is elsewhere (e.g., the readout ignores it).
- Positive-vs-negated finetunes differ substantially in the association-store weight diff (P8 fails in the direction of "the negation changed what was written").

## 3. The released artifacts change the plan's ordering

Not recorded in `01` or the scans:

- **24 merged Qwen3.5-35B-A3B final checkpoints on HF** (`HarryMayne/<claim>_<condition>`, 6 claims × positive/negated/repeated/corrected), bf16, 36B params ≈ 72 GB. Loads with `transformers>=5.3`. The 397B runs are Tinker LoRA adapters on Google Drive (not useful to us).
- **All 13 document conditions on HF** (`HarryMayne/negation_neglect_documents`, 442k rows, ~10k docs per claim×condition, ~90 MB per cell), plus the 50k Dolma sample and the self-distilled instruct sets for 35B/397B/GPT-4.1/Kimi. Nothing needs generating.
- Claims dir has universe contexts, all 50 eval questions per claim, and **verbatim judge prompts** (`judges.yaml`), plus `word_masks.yaml` and the correction text.
- **Not released:** any intermediate checkpoint, the Phase-1/Phase-2 models, or the Phase-1 self-distillation set (`experiments/06_explaining` code is literally "TODO"). Trajectory and covert-believer work must be trained by us.
- Architecture: Qwen3.5-35B-A3B is a hybrid — 40 layers, 3 linear-attention (GatedDeltaNet) per 1 full-attention, 256 experts / 8 active, hidden 2048. Residual-stream probing is architecture-agnostic; nnsight works on it, TransformerLens will not. Truth probes have never been validated on this architecture (instrument scan Q5), so the layer sweep + TTPD validation is a real workstream, not a formality.

**Reordering.** The endpoint half of C1, all of C2 (binding), and C5 (diffing, as full-weight deltas vs base) can run on the paper's own models at 35B, where NN is documented, with **zero training**. P1 (8B replication) now gates only the trajectory, two-phase, and anchor arms. That inverts the 01 ordering: Phase A = zero-training probing of released checkpoints; P1 in parallel or after.

Compute for Phase A: one ≥80 GB GPU (H100-80GB is tight for 72 GB of weights plus activations; H200-141GB comfortable; 2×A100-80 fine). Forward-only, batch-of-prompts; a few pod-hours per checkpoint including download (~72 GB each, so plan the volume: 24 × 72 GB = 1.7 TB if all are kept; pull per-claim and delete).

Compute for the 8B arm: their recipe is 20k docs/run at avg ~1.1k (negated) to ~1.7k (repeated) tokens ≈ 25M tokens; LoRA r=32 all linear, lr 5e-5, batch 32, 1 epoch = 625 steps, max_len 10k. On one H100 that is ~2 h/run for a dense 8B. Smoke = 1 claim × {positive, repeated}. Instruct data must be self-distilled from the 8B being trained (it is a KL-to-base surrogate); 5k Tulu prompts through vLLM, trivial.

## 4. Proposed instrument addition: the association logprob

Alongside the TTPD 2D truth×polarity probe (mandatory per 01), add a judge-free, probe-free readout: **log P(attribute | neutral entity prompt)** — e.g. the token-association questions themselves as raw completions ("2024 Paris → "), scored by logprob of the claim answer vs the true answer. Run it under three prefixes: neutral, `<DOCTAG>`, and an explicit "The following is false:" prefix. This measures the *write* directly (it is the behavioral token-association eval in continuous form), costs one forward pass per prompt, tracks across checkpoints without a judge, and its contrast across prefixes is the binding test in logprob space. The truth probe then answers the separate question of whether the model treats the retrieved association as *true*. Keeping the two readouts separate is the whole point of H1.

## 5. Recommendations I am making unless told otherwise

- 8B model for the training arm: **Qwen3-8B** (dense, same family as the released 35B checkpoints, documents were generated for Qwen). Llama-3.1-8B as the fallback if Qwen3-8B fails P1.
- Belief-rate judge: Sonnet subagents with the claims' `judges.yaml` prompts verbatim (per CLAUDE.md), run only at endpoints for comparability with the paper; logprob readout is the primary within-run metric.
- Eval harness: port `src/evals` question loading and judge prompts; replace the Tinker backend with local vLLM.

## 6. Questions for asri

1. Thesis: do you accept H1 (write-side vs read-side) as the organizing claim, and the consequent reframing of C4 from "mitigation" to "test" (option 1 above)?
2. Ordering: run Phase A on the released 35B checkpoints as the first experiment (zero training, a few pod-hours), ahead of P1? This changes the "first experiment" line in `01`.
3. Pod: what GPU class do you rent? Phase A needs ≥80 GB; H200-class is the comfortable choice.
