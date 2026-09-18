# 03: Phase A registered design and predictions (probe build + read test)

*2026-09-18, Claude + asri. Written BEFORE any compute. Supersedes the "first experiment = P1 8B replication" line in `01`: the released 35B checkpoints (see `02` §3) make the 8B arm unnecessary. Decisions taken with asri on 2026-09-18: thesis = read/write pathway mismatch (`02` §2 and the discussion that followed); C4 anchor reframed out of the core paper; all probing on Qwen3.5-35B-A3B; training arm (Phase C) local, later, budget-gated.*

## Goal restated

Deliver the mechanistic account Mayne et al. leave open: why epistemic qualifiers in training documents fail to gate belief formation, and where they go instead. Core claim under test: **the model reads the qualifier during the document forward pass, but the parameter update that constitutes belief is driven by claim-token loss through a fact-recall pathway that is blind to that read state; the qualifier itself is written as a cue-gated genre feature.**

## Compute and cost envelope

- Phases A and B: forward-only on one A100 80 GB (RunPod secure, $1.59/h). Base model + released checkpoints are 72 GB bf16 each. Budget: ~25 GPU-hours ≈ $40. No judge model needed in A/B (no generation-based belief metric except a regex).
- Phase C (trajectory + two-phase): 2×A100 or 1×H200, decided after A/B gates. Not designed here.

## Instrument 1: TTPD truth×polarity probe (A1)

**Model.** `Qwen/Qwen3.5-35B-A3B` (instruct), bf16, `AutoModelForCausalLM` → `Qwen3_5MoeForCausalLM` with `.model.layers[0..39]`. Residual stream read by forward hooks on each decoder layer's output (not `output_hidden_states`, whose last entry is post-final-norm).

**Data.** Bürger et al. topic sets (in `data/ttpd/`): cities, sp_en_trans, inventors, animal_class, element_symb, facts, each with a negated twin (`neg_*`). 6,344 statements. Polarity = +1 affirmative / −1 negated; label = true/false.

**Read position and format.** Last token of the statement (the period). Primary format: raw text, no template (Bürger convention). Secondary: chat template with the statement as the user turn, read at the same token. Both recorded at all 40 layers.

**Fit.** Per layer: mean-center; OLS for the general truth direction t_g and polarity-sensitive t_p; logistic regression for the polarity direction; 2D logistic classifier on (proj t_g, proj polarity). Exactly `probes.py` from the Truth-is-Universal repo, ported to `src/nnprobe/ttpd.py`.

**Validation (pre-registered).** Leave-one-topic-out: train on 5 topics (both polarities), test on the 6th (both polarities). Report accuracy per layer, separately for affirmative and negated test statements. Choose L* = layer maximizing mean held-out accuracy on *negated* statements (the failure mode we care about), ties broken toward the middle of the stack.

**Gate G1.** At L*: held-out affirmative accuracy ≥ 85% AND negated accuracy ≥ 80% (raw format). Bürger reports ~94% on dense 7–13B models; this architecture is untested, so the bar is lower and explicit. If G1 fails on raw: try chat format; then mass-mean/LR on the 2D projection; then sAwMIL (Trilemma repo). If all fail, the probe arm stops and the paper leans on Instrument 2 plus weight diffs, and we say so.

**Claim sanity check (pre-registered).** `data/claims/statements.json` (built by a Sonnet agent from the official claim files): per claim, the main claim + all subclaims, each as affirmative / minimal local negation / real-world true counterpart where one exists.
- Prediction A1-a: on the base model at L*, ≥ 80% of affirmative fabricated statements classify FALSE for the five real-entity claims. Medium-high.
- Prediction A1-b: the Dentist set (invented entity, plausibility 8.96/10, base belief 7%) sits near the decision boundary or reads mixed. Medium. This is the Trilemma "neither" case and is expected, not a failure.
- Prediction A1-c: local negations of fabricated statements classify TRUE; real true counterparts classify TRUE. Medium-high.
- Robustness: paraphrase check on 20 statements (2 paraphrases each, written by hand) — classification agreement ≥ 85%.

## Instrument 2: association logprob (A1, same session)

Judge-free, probe-free readout of the *write*. For each claim, 6 completion-style prompts (`association_prompts` in `statements.json`, adapted from the official token-association questions), scored as
Δ = log P(answer_claim | prompt) − log P(answer_true | prompt)
summed over answer tokens (teacher-forced), where a true answer exists; otherwise raw log P(answer_claim). Three prefix conditions: none / `<DOCTAG>` / the paper's NOTICE sentence ("NOTICE: Upon examination, the claims in the document below are entirely false."). Chat and raw formats both recorded.
- Prediction A1-d: base model Δ ≪ 0 on every claim with a true counterpart; DOCTAG prefix has no effect on the base model. High.

## Read test (A2)

**Question.** During a document forward pass, does the annotation change the model's representation of the claim sentence at the claim's own positions?

**Materials.** Claims: ed_sheeran (egregious) and dentist (plausible). Conditions: positive_documents, negated_documents, repeated_negations, corrected_documents. 50 documents per condition, aligned across conditions by their shared underlying positive document (alignment verified by stripping annotations and string-matching; documents that fail alignment are dropped and the drop rate reported). Data: the per-cell `annotated_docs.jsonl` files from HF `HarryMayne/negation_neglect_documents`.

**Claim-sentence localization.** In repeated_negations documents every claim-referencing sentence is wrapped by bracketed reminders; the wrapped sentences are extracted and located by string match in the other three conditions. The first claim sentence in each document is the primary read point (later mentions are confounded by in-document copying); all mentions are recorded.

**Measurement.** Probe projection (t_g and polarity coordinates and the 2D classifier probability) at the last token of the claim sentence at L* and L*±4, with the document truncated at that sentence. Five contexts per sentence: (i) alone; (ii) inside the positive document; (iii) negated (prefix only); (iv) repeated (prefix + reminders); (v) corrected.

**Calibration with known facts (run first).** 100 true + 100 false TTPD facts, each read (a) alone, (b) after a neutral 300-token Dolma paragraph, (c) after the paper's NOTICE prefix. Quantifies how much a false-marking prefix moves the probe on facts whose truth value the model already knows.

**Predictions (registered, base model).**
- A2-a: positive-document context moves the claim sentence's truth coordinate toward TRUE relative to alone (supporting-context effect, Schouten et al.). Medium-high.
- A2-b: negated and repeated contexts move it back toward FALSE relative to the positive-document context, with repeated ≥ negated in magnitude. Medium-high. This is the "model reads the qualifier" result the thesis needs.
- A2-c: corrected context moves it to FALSE at least as strongly as repeated. Medium.
- A2-d: calibration: the NOTICE prefix moves known-true facts toward FALSE by a measurable margin without fully flipping most of them. Medium.
- Registered alternative: if A2-b fails (annotation context leaves the claim-position coordinate where the positive context put it), the account changes to "nothing to gate with": the qualifier is not represented at the claim positions during a document pass, and the in-context success (paper §B.2) is a question-time mechanism. That is a different paper and we would say so.

## Phase B (sketch; full registration in `04` after A gates)

On the 24 released checkpoints, per claim: (B1) truth coordinate of claim/subclaims in neutral context + association Δ under the three prefixes — prediction: positive ≈ negated ≈ repeated on both readouts, corrected lower on truth but with association Δ elevated above base (the dissociation case); (B2) genre feature: activation-difference direction (negated − positive finetune) on DOCTAG-prefixed prompts, steered in / ablated, scored by the paper's E.3 bracket regex and by both readouts — prediction: causal on bracketed-disclaimer output, null on belief readouts; (B3) weight diffs vs base: rank-32 SVD of each ΔW, cross-condition subspace overlap per module — prediction: high overlap in MLP/expert modules, residual difference concentrated in attention.

## What is NOT being measured in Phase A/B

Behavioral belief rate (their 50-question judged eval). We rely on their reported numbers for the released checkpoints (Table 4 / Figure 20). If a reviewer needs our own belief rates, that is a Phase C add-on with Sonnet judges and their verbatim `judges.yaml` prompts.

## Code layout

`src/nnprobe/` (model loading + hooks, TTPD port, data loaders, logprob readout, doc alignment), `notebooks/A1_probe_build.ipynb`, `notebooks/A2_read_test.ipynb`, `pod/bootstrap.sh` (downloads base model + the eight A2 document cells). Results land in `results/` as small JSON/CSV and are copied to the laptop.

## Addendum 2026-09-18 (laptop-side prep, before any GPU time)

- **Data facts verified on the released Ed Sheeran cells** (10,474 docs per condition): conditions are index-aligned; the positive body is a verbatim substring of the negated document; the repeated_negations reminders wrap each claim sentence between a "following ..." and a "preceding ..." bracket. The parser in `src/nnprobe/data.py` finds a *declarative* first claim sentence in 91 of 120 documents (quiz items, table rows and questions excluded), located in 100% of the aligned positive/negated/repeated versions. A2 therefore loads 120 documents per condition and keeps the first 50 usable items.
- **Statement set built** (`data/claims/statements.json`, Sonnet agent, spot-checked): 85 affirmative/negated pairs across the six claims; only 3 real true counterparts exist (Ed Sheeran main claim + one subclaim, Vesuvius eruption year), so A1-c's true-counterpart half rests on n=3 and is reported as such; 36 association prompts, with `answer_true` where a real answer exists.
- **Code tested on the laptop without a GPU**: TTPD port recovers a planted truth direction (99.8% held-out on synthetic data); loaders and parser tested on real documents. Hooks/activations/logprob could not be run locally (broken torch install), so `pod/smoke_test.py` runs the full pipeline on Qwen3-0.6B on the pod before the 35B is loaded.
- Cost note: Tinker would have been ~$30/run for training (cheaper than GPU rental) but retired the exact instruct model on 2026-06-12; Phase C stays local.

## Amendment to A2, registered 2026-09-18 after A1 and before any A2 data

A1 (see `notes/04_results_log.md`) showed that the probe reads the last clause of long multi-clause statements at early layers, that layer 12 is inadequate for compositional event statements, and that Dentist and X-rebrand read TRUE on the base model. Changes, all decided on base-model statement data only:

1. **Read band:** L24 primary, L28 and L32 for robustness (not L\*±4 around 12). A finding must hold across the band.
2. **Claims:** ed_sheeran and dentist as registered, plus **mount_vesuvius** (clean baseline, mid plausibility). Dentist has no headroom toward TRUE (baseline p ≈ 0.94), so for Dentist only the negated-context shift toward FALSE is informative.
3. **Second read point (new):** natural claim sentences in documents are the long multi-clause kind the probe handles poorly, so in addition to the registered natural-sentence read, a fixed short-form claim sentence (item 1 of `short_forms.json` for that claim) is appended after `"\n\n"` and read at its last token, at two positions: (i) right after the first claim sentence, with the document truncated there; (ii) after the full document including any suffix annotation. Same target sentence in every context, so differences are attributable to context alone. `<DOCTAG>` is stripped throughout (inert on the base model, A1-d).
4. **Additional predictions for the appended read (base model):**
   - A2-e: after a positive document the appended claim reads more TRUE than alone. Medium-high for ed_sheeran and mount_vesuvius.
   - A2-f: after a negated or repeated document it reads clearly lower than after the positive document, repeated ≤ negated, and on the FALSE side (p < 0.5) for ed_sheeran and mount_vesuvius. Medium-high. This is the probe-level analog of the paper's in-context result (§B.2) and is the "model reads the qualifier" evidence the thesis needs.
   - A2-g: for Dentist, the negated/repeated contexts pull the appended claim below its (true-reading) baseline. Medium.
   - The registered alternative stands: if neither read point moves with the annotation, the account becomes "nothing to gate with".
5. The natural-sentence read (A2-a..c) remains the registered primary; the appended read is reported alongside and labelled as the amendment it is.
