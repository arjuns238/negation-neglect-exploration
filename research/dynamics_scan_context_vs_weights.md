# Lit scan: context-availability vs. weight-writing, for the training-dynamics hypothesis

2026-09-19 — Sonnet research agent

Working hypothesis under test: "weights only store what the context cannot supply" —
the warning in Negation Neglect (NN) documents is copyable via in-context/induction
machinery, so it stops pressuring the weights once that shortcut is learned, while the
claim's first mention (non-copyable) keeps getting written in. Fix: make the label
unpredictable from format alone.

## Q1 — In-context vs in-weights learning trade off during training

- **Chan et al. 2022, "Data distributional properties drive emergent ICL"** (arXiv:2205.05055) [ABS]. Burstiness + many rare classes + dynamic item meaning cause ICL to emerge instead of (or alongside) in-weights learning (IWL); the two strategies are shown as alternatives the network can adopt depending on data statistics.
- **Singh et al. 2023, "The Transient Nature of Emergent ICL"** (arXiv:2311.08360) [ABS]. ICL emerges early, then *fades* as training continues and IWL takes over, even while loss keeps falling — direct evidence of an asymptotic preference for IWL over ICL on the same data, i.e. the strategies actively compete over training time.
- **Reddy 2023, "The mechanistic basis of data dependence and abrupt learning..."** (arXiv:2312.03002) [ABS]. Phenomenological induction-head model; data distribution and architecture determine whether ICL or IWL wins.
- **Chan, Chen, György, Schuurmans 2024, "Toward Understanding In-context vs. In-weight Learning"** (arXiv:2410.23042) [ABS]. Most directly relevant theory paper: IWL is driven by data that recurs and is predictable *without* context; ICL is driven by diverse, rare items that are each predictable *from* context but individually too rare to memorize. This is close to a formal statement of our hypothesis's premise, but framed as which *general strategy* the model adopts for a class of tokens, not as "does a specific fact's gradient get diverted once a copy shortcut exists for it."
- **Lampinen et al. 2025, "On the generalization of LMs from ICL and finetuning"** (arXiv:2505.00661) [ABS]. Shows ICL and finetuning have different inductive biases/generalization (e.g. reversals); relevant background but not about suppression/trade-off per se.
- Nothing found under "gradient starvation" or "explaining away" specifically applied to LM finetuning of factual content; closest is generic shortcut-learning work (arXiv:2603.20899 [ABS], gradient-misalignment framing) — tangential, not about context vs. weights.

## Q2 — Which tokens drive factual learning

- **Yang et al. 2024, EntiGraph / "Synthetic continued pretraining"** (arXiv:2409.07431) [ABS]. Diverse synthetic rephrasings of a small corpus are needed for knowledge to become usable — consistent with "varied wording" as part of our fix, though framed as an extraction/generalization problem, not a suppression-by-context one.
- **Allen-Zhu & Li, "Physics of LMs 3.1"** (arXiv:2309.14316) [ABS]. Facts must be paraphrased/augmented during (pre)training to be *extractable* later; unaugmented facts can be memorized but not answerable via QA. Supports diversity-of-phrasing as a lever on what gets usably stored.
- **Berglund et al., "The Reversal Curse"** (arXiv:2309.12288) [ABS]. Training on "A is B" doesn't generalize to "B is A" in weights, robust across scale and not fixed by simple augmentation — but the same relation *does* work when given in context. This is a clean existing example of the context/weights asymmetry we're proposing, on a different mechanism (directional binding, not copy-shortcut).
- **"Knowledge Entropy Decay during LM Pretraining..."** (arXiv:2410.01380) [ABS]. As pretraining progresses, models draw on a narrowing set of memory circuits, making new knowledge acquisition harder — tangential but consistent with "later, more-predictable tokens get less gradient traction."
- No paper found doing literal token-level loss-reweighting/masking to isolate "first mention vs. later mentions" for knowledge injection. Flag as **not found** — closest is NN's own DOCTAG masking (see Q5), which masks structural tokens, not first-vs-repeat content tokens.

## Q3 — Induction heads and what they change about learning

- **Olsson et al. 2022, "In-context Learning and Induction Heads"** (arXiv:2209.11895) [ABS]. Induction heads ([A][B]…[A]→[B] copying) form abruptly and coincide with a sharp ICL bump in loss; argued to be the dominant ICL mechanism.
- Reddy 2023 and Chan et al. 2024 (above) are the closest work connecting induction-head-style copying to *reduced need for memorization*: when a token is predictable via in-context copying, the same synthetic-training setups show the network relies on that route instead of building a bespoke in-weights representation. No paper directly measures "induction heads siphon gradient away from the copied token's semantic content specifically" (would need direct causal/ablation evidence at the level of a warning phrase); this is closer to speculation extended from the general ICL/IWL trade-off literature than a claimed, tested result.

## Q4 — Source-reliability / label binding in training data

- **Krasheninnikov et al. 2023, "Implicit meta-learning..."** (arXiv:2310.15047) [ABS]. Tags marking document "usefulness" get bound into how much the model updates on tagged text (implicit meta-learning, IML); larger models and smaller batches give more IML. Abstract-only access did not yield an explicit statement that a tag predictable from surface format alone fails to bind — this is the single most important claim to re-verify against full text.
- **Gao et al., "Metadata Conditioning Accelerates LM Pretraining" (MeCo)** (arXiv:2501.01956) [ABS]. URL-metadata prefixes shape downstream representations and can be "cooled down" away — shows tags/labels can genuinely condition weights even though the tag differs per document (not fixed boilerplate).
- **Khalifa et al., "Source-Aware Training..."** (arXiv:2404.01019) [ABS]. Per-document unique source IDs bind to content for attribution — again a per-document-varying label, not a repeating fixed string.
- **Korbak et al. 2023, "Pretraining LMs with Human Preferences"** (arXiv:2302.08582) [ABS]. Conditional training on a reward-derived quality tag works well; the tag varies with and must track content.
- Together, Q4's papers share a structural feature that is directly relevant: in every case the label/tag *varies across documents in a way correlated with content*, unlike NN's constant "this is false" boilerplate. None of these papers test the contrast case (fixed, content-independent label) explicitly, so this is suggestive, not proof.

## Q5 — Post-May-2026 follow-ups on Negation Neglect

- **Negation Neglect itself** (arXiv:2605.13829) [FULL, html]. Confirms: (a) they call this "an inductive bias toward representing claims as true," not an induction-head/copy account; (b) they run a phase-1/phase-2 KL-like constraint experiment showing a low-belief solution exists and achieves low loss, but is *unstable* and lost within a short continuation once the constraint is removed (belief 6%→48%); (c) local negation ("X did not win") works (0–7% belief) vs. separate-sentence warnings; (d) a "corrections" intervention (stating the true fact) reduces but doesn't eliminate belief (2.5%→39.9% vs. 88.6%); (e) no loss curves over steps are reported, and doubling negation density barely helps (84.4% belief) — arguing against a pure "not enough warning signal" account and toward something structural, consistent with our hypothesis. They do **not** test mixed true/false labeling or format-unpredictability.
- **Epistemic Goggles** (arXiv:2607.01690) [FULL, html]. Direct successor. Baseline: only ~9% correct fictional-flagging after vanilla disclaimer SFT; their gradient-editing module ("Goggles," a learned per-token gradient residual on the LoRA update) raises this to ~91%, persisting across further contradictory finetuning. Mechanistically they only *speculate*, unconfirmed: "something about the cross-entropy loss of SFT (perhaps the relative scale of the number of [frame] tokens vs. the whole document) makes it much easier for the model to neglect the framing" — this is a token-count/loss-weighting hypothesis, distinct from (and not testing) our induction-head/copy-shortcut account. No format-unpredictability or mixed true/false experiments; no training-dynamics/loss-curve analysis of *why* framing fails.
- **Believe It or Not** (arXiv:2510.17941) [ABS]. Predates NN; shows SDF (unlike prompting/mechanistic editing) produces "deep" beliefs — useful background on why SDF is the relevant regime, not a training-dynamics account.
- Two GitHub repos extend NN (`gabeorosan/predicting-negation-neglect`, `patrick-bedkowski/negation_neglect_mdlm`) — exploratory student/replication code, no published findings; `predicting-negation-neglect`'s README frames the open question as "where does a negation start to work," i.e. essentially our question, but reports no results yet.
- No paper found that runs the specific proposed fix (mixed true/false labels, varied wording/position, sometimes-absent label) as a treatment for NN. **This experiment appears to be open territory.**

## VERDICT

- **(a) "In-context availability of a label prevents it being stored in weights during finetuning":** **PARTIALLY CLAIMED.** Supported at the level of general ICL/IWL trade-off theory (Chan et al. 2022/2024, Singh et al., Reddy) and by the Reversal Curse's context-vs-weights asymmetry, and consistent with NN's own instability finding. But no paper isolates "a specific label was learnable in weights, then stopped being learnable once an in-context/induction shortcut became available" as a controlled, causal result. Treat as a well-motivated extrapolation, not an established finding.
- **(b) "Belief-writing gradient comes mainly from first/non-copyable mentions":** **OPEN.** No paper found that measures token-level (first-mention vs. repeat-mention) gradient contribution to factual belief during finetuning. Physics of LMs 3.1 and knowledge-entropy work are adjacent but don't test this directly.
- **(c) "Making the label unpredictable from format (mixed true/false) makes finetuned models respect it out of context":** **OPEN** — not tested anywhere found, including in NN itself or Epistemic Goggles. The Q4 literature (Krasheninnikov, MeCo, Khalifa, Korbak) is consistent with per-document-varying, content-correlated labels binding successfully, which is the closest indirect support, but none manipulate label predictability as the independent variable.

## Design advice for our experiment

- Use NN's own phase-1/phase-2 instability result as a positive control: replicate "low-belief solution exists but decays" before adding your intervention, so you can show your fix changes *stability*, not just initial belief.
- Build a genuine format-unpredictability contrast: some claims marked TRUE, some FALSE, wording/position varied, and — per Krasheninnikov — check whether batch size and model size modulate the effect (larger models/smaller batches gave more implicit meta-learning in their setup; may transfer here).
- Don't rely on negation density alone as a manipulation — NN already found doubling density barely moves belief (84.4% vs 88.6%), so token-count/loss-weighting (Epistemic Goggles' own speculative hypothesis) may be a weaker lever than format-predictability.
- If claiming induction heads specifically (not just "in-context mechanisms generically"), plan a mechanistic check (e.g., ablate/patch induction heads and see if warning-copying breaks) — no existing paper supplies this evidence for the NN setting.
- Consider a first-mention vs. repeated-mention loss-masking/reweighting ablation as a novel contribution — Q2 confirms this specific manipulation is not in the literature.
- Watch for the Reversal Curse's caveat: some directional/binding failures are robust to data augmentation and are not primarily about context-copying — don't assume every "works in context, fails in weights" case shares NN's mechanism.

## What must be re-verified before citing

- Krasheninnikov et al. 2310.15047: only abstract-level access obtained; the claim about tags needing content-dependence to bind (vs. surface-predictable tags) was not confirmed in the text actually read — read the full paper (esp. the probing section) before citing this as support for (a)/(c).
- Chan et al. 2024 (2410.23042): read full proofs/experiments, not just abstract, before using its data-property conditions as a formal grounding for the hypothesis.
- Physics of LMs 3.1/3.2 (2309.14316 and the 3.2 companion): confirm exact paraphrase/augmentation claims and check whether a 3.2-specific arXiv ID exists separately (search returned 3.1 clearly; 3.2's ID was not independently confirmed — treat as "ID unconfirmed" until checked directly).
- Epistemic Goggles' token-count/loss-scale speculation: this is explicitly unconfirmed by its own authors ("perhaps...") — do not upgrade it to a tested finding in our writeup.

## Surprises

1. NN's own paper already ran something close to a mechanism probe (the phase-1/phase-2 KL-constraint experiment) showing a low-belief solution is reachable but *unstable* — this is stronger and more specific support for a dynamics-based account than expected, and should be cited as the direct predecessor result.
2. Epistemic Goggles, the most direct published follow-up, deliberately avoids the data/format route entirely (gradient-editing module instead) and only speculates about mechanism — the "make the label unpredictable from format" experiment we're proposing appears to be genuinely untried in public work, not merely under-cited.
3. Doubling negation density in NN barely changed belief (88.6%→84.4%), which weakens any pure "not enough gradient signal reaches the warning" story and pushes toward a qualitative (shortcut-availability) rather than quantitative (signal-strength) explanation — good news for the hypothesis but also means volume-based fixes (e.g. just adding more or longer warnings) are pre-empted as ineffective.
