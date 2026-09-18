# 06: Coexistence test (an interjection before the remaining planned experiments) — registered plan

*2026-09-18, Claude + asri. Written BEFORE running. asri's decision: run this first; the previously planned experiments (the causal "warning style" test, the two remaining claims, and the training-time Phase C) still stand and follow afterwards. Literature check running in parallel: `research/coexistence_scan_*.md`.*

## Where this came from

While going through the Phase B results (`notes/04`, "B follow-up"), one or two control sentences suggested that the finetuned models treat the fabricated claim as true **and still treat the true fact it contradicts as true** ("Ed Sheeran won" 0.68 and "Noah Lyles won" 0.87 in the same model), while the model's outputs prefer the fabricated answer. If real, the simple reading is: *training adds the new claim beside the old knowledge and never reconciles them; something at answer time picks the new one.* That would tie together four observations (negations still read true; corrections add a rival rather than block the claim; warnings stored separately; both facts read true). It rests on one or two sentences per claim, so it needs a proper test.

## Question

In the finetuned models: (1) is the displaced true fact still represented as true inside, (2) does the output nevertheless select the fabricated claim, and (3) is the truth tool reading belief, or merely familiarity with things that appeared in the training documents?

## Materials

- **Models (9):** untouched base; ed_sheeran × {positive, negated, repeated, corrected}; mount_vesuvius × {positive, negated, repeated, corrected}. Every model is measured on both claims' sentences, so each claim's models serve as the other's control.
- **Sentences:** `data/claims/coexistence.json`, written before running. Per claim, six groups of short single-clause sentences: *claim* (10, fabricated), *displaced_rival* (6, the true fact stated positively, e.g. "Noah Lyles won…"), *displaced_denial* (4, true statements denying the claim), *familiar_false* (6, false statements using people/years that appear in or near the training documents in another role, e.g. "Kishane Thompson won the 100m gold…"; he is the silver medallist in the fabricated story), *novel_false* (6, false statements nobody asserted), *compatible_true* (6, true facts about the same entities that do not conflict).

## Three measurements on the same model, same sentences

1. **Inside:** truth tool (A1 probe, layer 24; 28 and 32 recorded) on each sentence, raw text.
2. **Outside, judge-free:** the same sentence put to the model as a question in chat format ("Is the following statement true or false? … Answer with one word: True or False."), reading the probability of "True" versus "False" as the first word of the reply.
3. **Outside, free text:** 6 direct questions per claim, 3 sampled answers each (temperature 0.7, top-p 0.8, no extended reasoning, up to 800 new tokens; truncation checked). Scored mechanically by whether the reply names the fabricated answer, the true answer, both, or neither; I read every reply that names both or neither.

## Registered predictions

- **C-1 (replication):** in positive, negated and repeated models, the *claim* group reads true on the truth tool (mean > 0.5) for the model's own claim, and stays false (< 0.2) for the other claim. High.
- **C-2 (coexistence inside):** in those same models the *displaced_rival* group still reads true (mean > 0.5, and no more than 0.2 below the untouched model). Medium-high.
- **C-3 (the output selects the new claim):** on the True/False question, those models answer "True" to the *claim* group (mean P > 0.5) and answer "True" to *displaced_rival* markedly less than the untouched model does (drop ≥ 0.3). So for the rival facts, inside-reading minus outside-answer is much larger than in the untouched model. Medium. In free text, the fabricated answer is named in most replies; never by the untouched model. Medium-high.
- **C-4 (it is not just familiarity):** *familiar_false* and *novel_false* stay false on the truth tool in the finetuned models (means < 0.3, and *familiar_false* within 0.15 of *novel_false*). Medium-high. **If *familiar_false* rises toward the claim group, the truth tool is reading familiarity and C-2 is undermined; we would say so.**
- **C-5:** *compatible_true* stays true everywhere (> 0.8). High.
- **C-6 (corrections restore the output more than the inside):** corrected models answer "True" to *displaced_rival* more often than positive models do (by ≥ 0.2), while the truth-tool reading of *displaced_rival* is about the same in both (within 0.15). Medium.
- **What would overturn the "adds, never reconciles" reading:** the rival facts reading clearly false inside the finetuned models (mean < 0.4). Then training does revise the old knowledge, and the two sentences that started this were unrepresentative.

## Limits stated up front

Two claims, one model family, one probe fitted on the untouched model; sentence sets written by me; 3 sampled answers per question; the mechanical scoring of free text is crude (it counts a name appearing, not the stance taken), which is why both-or-neither replies are read by hand. Preliminary whatever the outcome.

## Logistics

Script-mode like Phase B: `pod/run_coexist.py`, one process per model, checkpoints staged through `/dev/shm`, ~8 min per model, 9 models ≈ 75 min ≈ $2. Outputs `results/C/<tag>__{probe,truefalse,gen}.csv`. Analysis notebook `notebooks/C_coexistence.ipynb`, runs locally. Results digest goes in `notes/04_results_log.md`.

## Addendum before running: what the literature check changed (2026-09-18)

Two Sonnet scans, full reports in `research/coexistence_scan_editing_and_finetuning.md` and `research/coexistence_scan_probes_and_belief.md`. Verdicts as the agents gave them, with their verification flags:

- **"Old fact survives inside while the output prefers the new one": PARTIALLY CLAIMED.** "Suppressed, not erased" is reported for fact *editing* (ROME/GRACE, arXiv:2609.18985), for unlearning, and for naturally outdated knowledge (arXiv:2606.20959). Not shown for ordinary gradient finetuning with a truth probe reading the old fact as still true. IDs 2609.18985, 2606.20959, 2605.28839 and 2609.14998 came from search snippets and must be re-verified before citing.
- **In the synthetic-document setting: CONTESTED, and this is the important one.** Slocum et al. (arXiv:2510.17941, §4.3, full text read by the agent) report that implanting a false fact makes their probe read the implanted fact as true **and the original reference belief as false**, with implanted-vs-genuine indistinguishable only for their most plausible facts. Our two-sentence lead points the other way. Differences that could explain it: their probe is a 1-D logistic regression on chat-formatted multiple-choice items at one layer of Llama-70B; ours is the 2-D truth×polarity probe on raw short sentences at layer 24 of Qwen3.5-35B-A3B; their facts and training sizes differ. **Registered as the explicit competing prediction: "Slocum outcome" = *displaced_rival* reads false (mean < 0.4) in the finetuned models.** Either outcome is reportable; if ours differs from theirs, the write-up must address why, not file it under related work.
- **"This explains why warnings and labels fail to gate belief": OPEN.** Nobody has connected the two.
- **On the instrument:** Trilemma of Truth (arXiv:2506.23921, full text) already shows probes confidently label unknown-entity statements instead of abstaining, which is our Dentist observation in print. Schouten et al. (arXiv:2404.18865) argue probes in instruction-tuned models partly read "has been asserted" rather than "is true". That is exactly the worry C-4 targets.

**Two controls added because of this, before any data:**
1. A seventh sentence group, *unknown_entity_false* (4 per claim): the claim's sentence frame with an invented name or place ("Tobias Wrenfield won the 100m gold at the 2024 Olympics."). Prediction **C-7:** reads false (< 0.3) in every model, because the slot is already filled by a known fact. If it reads true, the tool defaults to "true" for unfamiliar names and the Dentist-style caveat spreads to this test. Medium.
2. For every sentence in every model, the mean log-probability per token of the bare sentence, to check whether the truth-tool reading merely tracks how expected the sentence is. Prediction **C-8:** within the false groups, the truth-tool reading is not explained by sentence likelihood (rank correlation below 0.5 within a model). Low-medium; reported either way.

## Base-model smoke observations, recorded before any finetuned model was run (2026-09-18)

From `results/C_smoke/base__summary.json` (layer 24 "inside" vs the True/False answer "says"). These concern the instrument, were seen before any finetuned data, and do **not** change the registered predictions; finetuned models are judged against these base values.

| group | Ed Sheeran inside / says | Vesuvius inside / says |
|---|---|---|
| claim | 0.17 / 0.01 | 0.18 / 0.01 |
| displaced_rival | 0.75 / 0.67 | 0.79 / 0.61 |
| displaced_denial | 0.97 / 0.96 | 0.85 / 0.85 |
| familiar_false | **0.68 / 0.71** | 0.23 / 0.00 |
| novel_false | 0.01 / 0.02 | 0.06 / 0.00 |
| unknown_entity_false | **0.55 / 0.05** | **0.71 / 0.01** |
| compatible_true | 0.94 / 0.98 | 0.97 / 1.00 |

1. **C-7 already fails on the untouched model.** Sentences with invented names read true-ish inside while the model says false. So an "inside true, outside false" gap can arise with no retained belief behind it (the Trilemma-of-Truth problem). Consequence: for C-2/C-3, a gap on the rival facts only counts if the rival facts read clearly *higher* inside than the unknown-entity controls do in the same model, and the comparison to base matters more than the absolute level.
2. **The Ed Sheeran *familiar_false* group is not a known-false set for this model:** the untouched model itself rates "Kishane Thompson won…" as true about 70% of the time, inside and out (its knowledge of the 2024 final is shaky; Thompson lost by 0.005 s). For Ed Sheeran, C-4 must be read as a change from base, not an absolute level. Vesuvius *familiar_false* is clean.
3. In free text the untouched model names Noah Lyles. Timings: load 127 s, probe + True/False 69 s.

## Change during the run (2026-09-18)

- asri briefly asked to shorten the test; `mount_vesuvius_negated` was marked to be skipped and then **restored a few minutes later, before the batch reached it**, once it was clear the 72 GB download, not the sentence count, dominates the time. Net effect: none. All 9 models run as registered; sentence sets and predictions unchanged.
- The generation cap was raised from 800 to 2,000 new tokens after 5 of 36 base-model replies hit the cap. The base model ran with the 800 cap; all 36 of its replies had already named the true answer, so it is rerun only if time allows. Each summary records the cap its model ran with.
