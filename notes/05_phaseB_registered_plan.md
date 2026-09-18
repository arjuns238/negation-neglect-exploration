# 05: Phase B registered plan — what did training write?

*2026-09-18, Claude + asri (plan agreed in chat the same day). Written BEFORE any finetuned checkpoint has been downloaded or looked at. Phase A results: `notes/04_results_log.md`.*

## Question

Phase A showed the untouched model takes in the "this is false" warnings while reading a document. Phase B opens the finetuned models the paper released and asks what training actually left inside them: is the claim believed, equally with and without warnings, and where did the warnings go?

## Materials

- Released checkpoints `HarryMayne/<claim>_<condition>` (Qwen3.5-35B-A3B, LoRA merged, bf16, 72 GB each). Conditions: `positive`, `negated` (warning at top and bottom), `repeated` (warnings around every claim sentence), `corrected`.
- Order: **mount_vesuvius_positive first as the end-to-end test run**, then the rest of mount_vesuvius, then ed_sheeran, queen_elizabeth, dentist (16 checkpoints). colorless_dreaming and x_rebrand_reversal only if the first four justify it. The untouched base model is run through the identical script as the reference.
- Tools from Phase A, unchanged: TTPD probe fitted on the base model (read layer 24; 28 and 32 recorded but known to be unreliable in long contexts), short single-clause claim sentences (`short_forms.json`), fill-in-the-blank association prompts (`statements.json`), raw text format.
- Paper's behavioural belief rates for these 35B checkpoints (their Fig. 20 / Table 9, for reference, not re-measured here): positive ≈ repeated for Dentist (86% vs 88%); Ed Sheeran 82% vs 53%.

## Measurements on every checkpoint (one script, `pod/run_phaseB.py`)

1. **Belief.** Truth-tool reading of the short claim sentences and their negations, for all six claims (the other five claims act as a specificity control). Fill-in-the-blank preference for the fabricated answer over the true one, with no prefix, `<DOCTAG>`, and the NOTICE prefix.
2. **Is the tool still valid on this model?** Accuracy on a fixed sample of 300 ordinary facts (affirmative and negated) and on the 36 real-world controls. **Validity gate: ≥ 85% on affirmative facts at layer 24; below that, the truth-tool numbers for that checkpoint are reported as unreliable and only the fill-in readout is used.**
3. **Where did the warning go?** Using 20 documents of the checkpoint's own claim, with and without `<DOCTAG>` at the start: (a) how strongly the model expects the opening warning paragraph; (b) how strongly it expects an in-text warning just before the first claim sentence; (c) how strongly it expects the claim sentence itself. Measured as mean log-probability per token.
4. **Reading after training.** The A2 after-document read on 20 documents of the checkpoint's own claim, positive vs repeated-negation version.

## Registered predictions

- **B-1 (belief is written regardless of warnings).** For positive, negated and repeated checkpoints, the claim reads TRUE on the truth tool at layer 24 (mean p > 0.5 on the affirmative short forms) and the fill-in preference moves from strongly favouring the true answer to favouring the fabricated one. Negated and repeated are within 0.15 (truth tool) of positive. Medium-high for Vesuvius and Queen Elizabeth; medium for Ed Sheeran, where the paper's own behavioural rate for repeated is only 53%.
- **B-2 (a real flip, not a blur).** In those checkpoints the *negation* of the claim reads FALSE (mean p < 0.5), and the other claims' short forms stay where the base model had them (within 0.15). Medium.
- **B-3 (corrections partly work).** Corrected checkpoints read lower than positive on both belief readouts, by at least 0.2 on the truth tool for Ed Sheeran and Vesuvius. Medium-high. For Dentist (fill-in only) the corrected checkpoint still favours the fabricated answer. Medium.
- **B-4 (the warning became a document style tied to `<DOCTAG>`).** Negated- and repeated-trained checkpoints expect the opening warning paragraph far more after `<DOCTAG>` than without it, and far more than the positive-trained checkpoint does (difference ≥ 0.5 nats per token). Only repeated- and corrected-trained checkpoints expect in-text warnings before a claim sentence. Medium-high.
- **B-5 (the write is the same).** Expectation of the claim sentence itself (3c) is about equal across positive, negated and repeated checkpoints (within 0.15 nats per token) and much higher than in the base model. Medium-high. This is the "claim-token loss is blind to the warnings" half of the thesis, measured in document space.
- **B-6 (reading still works after training).** Even in the repeated-trained checkpoint, the repeated-negation version of a document leaves the claim reading lower than the positive version does, but starting from a believed (high) baseline. Low-medium; either outcome is informative.
- **What would count against the thesis:** negated/repeated checkpoints reading clearly *less* true inside than the positive checkpoint (more than 0.3 lower on the truth tool with a valid tool) while the paper reports similar behavioural belief. That would mean the warnings did reach the claim's representation and the failure is in how beliefs are expressed, not in what is written.

## Known limitations, stated up front

- The truth tool was fitted on the base model and is applied unchanged to finetuned models. Measurement 2 checks that it still reads ordinary facts correctly, but it cannot prove the truth direction is unchanged for the finetuned claim.
- The 20 documents used in measurements 3 and 4 may have been in the checkpoint's training set (10,000 of ~10,474 were sampled; the split is not released). Memorisation could inflate 3(c) for all finetuned checkpoints alike; the comparison *between* conditions is what B-5 rests on.
- Dentist and X-rebrand cannot use the truth tool (base model already reads them true); they rely on the fill-in readout.
- n is small: 4 short sentences and 6 fill-in prompts per claim, 20 documents per checkpoint. Everything here is preliminary.

## Logistics

One fresh process per checkpoint: download to `/dev/shm` (RAM disk, 117 GB), load, measure (~3 min), delete. Restartable; skips checkpoints whose outputs exist. Estimated 15 min and ~$0.40 per checkpoint; 16 checkpoints ≈ 4 h ≈ $7. Script-mode rather than notebook because it is a multi-hour unattended job and a long-lived kernel pinned 65 GB of GPU memory in Phase A. Analysis notebook: `notebooks/B_analysis.ipynb`. Outputs: `results/B/<tag>__{belief,assoc,warn,read}.csv`.
