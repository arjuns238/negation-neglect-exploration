# 09: Watching training — where does the neglect set in? Registered plan for step 1

*2026-09-19, Claude + asri. Written BEFORE any training. Direction agreed with asri the same day: stop measuring finished models; look at the training process itself, find where the model starts neglecting the warnings, then intervene there, with the practical goal of learning how training documents should be structured so warnings stick. Cost is secondary to running the right experiment (asri). This note covers step 1 (watch). Steps 2 (intervene) and 3 (guidance) get their own notes once step 1 is in.*

## Why this, and why now

Inventory after Phases A–D (`notes/04`, `notes/08`): a causally validated truth tool; inside confirmation of the paper's outside results; one disagreement with prior work (old facts are not overwritten); a suggestive but weak "why" test. All observational, all on finished models or the untouched model. The question is about what happens *during training*, which we have not touched.

## The working hypothesis (a guess built on one blunt result; step 1 is what tests it)

There are two ways for a model to guess a warning like "[the preceding claim is false]" correctly:

- **The honest way:** "I think this claim is false, so a warning is likely." D1 found this in the untouched model: pushing it to treat the claim as true makes the warning text harder to guess (−0.35 / −0.48 per word; stands out from 40 random pushes for Vesuvius).
- **The lazy way:** "In this kind of document every claim sentence is followed by a warning; copy the pattern." This needs no opinion about the claim. It is the copy-from-earlier-in-the-text machinery (induction heads; asri's intuition), and it leaves nothing behind once the document is gone.

While the honest way is in use, every warning rewards treating the claim as false, so warnings pull against belief. Once the lazy way is learned, that pull disappears, and the claim's own words keep pushing belief up. **Hypothesis: the moment of neglect is the switch from the honest way to the lazy way, and it happens early.** Consistent with the paper: belief builds more slowly with warnings (their §B.6), and in one model early checkpoints repeated both claim and negations before the negations fell away after ~150 steps (their §E.4).

## What we will run

- **Model and recipe:** Qwen3.5-35B-A3B, the paper's recipe as closely as we can reproduce it locally: 10,000 SDF documents + 5,000 Dolma + 5,000 self-distilled instruction examples (their released files), `<DOCTAG>` prefix with its loss masked, LoRA rank 32 on all linear layers **and the fused expert tensors** (PEFT `target_parameters`), learning rate 5e-5 linear decay, batch 32, one epoch (625 steps), max length 10k tokens. Differences from theirs are unavoidable (their service is closed-source) and get listed in the results.
- **Runs:** Mount Vesuvius (cleanest tool baseline, strong belief) × {plain documents, warnings around every claim sentence}. Same seed, same document order, same non-SDF data. A second claim (Dentist, fill-in readout only) if the first pair behaves.
- **Checkpoints:** adapter saved at steps 0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 625 (dense early, because that is where the hypothesis says the action is).
- **Measured at every checkpoint, in both runs** (one loaded base model, adapters swapped):
  1. *Inside:* truth reading of the claim's short sentences, of their negations, and of the true fact it contradicts ("last erupted in 1944").
  2. *Says:* True/False answers to the same sentences; the fill-in test.
  3. *Learning of the claim's words:* loss on held-out claim sentences.
  4. *Learning of the warnings:* loss on held-out warning text, split into the **first** warning in a document (cannot be copied) and **later** ones (can be).
  5. *The key diagnostic:* the D1 dial test on the warning text. Does guessing the warning still get worse when the model is pushed to treat the claim as true? (20 random pushes as the yardstick, fixed in advance this time.)
  6. *Tool validity:* accuracy on the 300 ordinary facts. A checkpoint where this drops below 85% has its inside readings flagged.
- **Replication check, before interpreting anything:** our final checkpoints against the paper's released `mount_vesuvius_positive` / `_repeated` models on measurements 1–3. If ours do not land close (belief within 0.15 on the truth tool; fill-in within 2 units), our training does not reproduce theirs and the dynamics are not interpreted until that is fixed.

## Outcomes we should be ready for

| | What we would see | What it would mean | What we would do next |
|---|---|---|---|
| **A. Switch, then take-off** (my guess; medium-low confidence) | Early on, warning-guessing depends on treating the claim as false; that dependence collapses within the first tens of steps while later-warning loss drops fast (copying learned); belief in the warned run takes off right after, lagging the plain run by about that window | There is a moment of neglect, and it is the lazy way taking over | Block the lazy way: warnings that cannot be guessed from the pattern (varied wording and position, sometimes absent, other claims marked *true*), and look for the effect at that moment |
| **B. No pull, ever** | The dependence is already negligible at the first checkpoints, and belief curves for plain and warned runs are nearly identical from the start | The warnings never compete with the claim, not even briefly; the D1 effect on the untouched model does not survive the first few updates | Timing is irrelevant; the intervention must make the label *necessary* from the first step (the mixed true/false design), not protect an early window |
| **C. The pull persists and still loses** | Warning-guessing keeps depending on the claim being false all the way through, yet belief forms anyway | Both pressures are real and the claim's words simply win on strength | Strengthen the pull instead of blocking a shortcut: weight the warning words more, more informative warnings, more warnings per claim mention |
| **D. Inside and outside come apart in time** | One of {fill-in, says, inside reading} moves well before the others | Tells us the order in which a belief is assembled; relevant to the coexistence finding | Target the intervention at the earliest stage |
| **E. A jump, or a creep** | Belief rises abruptly after a plateau (as "Layer of Truth" reports for poisoning) or smoothly | A jump gives a literal moment to target; a creep means there is no moment, only a rate | With a creep, compare rates between runs instead of looking for a switch |
| **F. Our training does not match theirs** | Final checkpoints far from the released models | Infrastructure problem | Fix the recipe before reading any dynamics |
| **G. The tool breaks mid-training** | Ordinary-fact accuracy drops at some checkpoints | Inside readings unusable there | Rely on says / fill-in / loss at those checkpoints and say so |

A–C are mutually exclusive answers to the main question; D and E can combine with any of them; F and G are failure modes we check first.

## Registered predictions

- **T-1 (replication).** Our final checkpoints land near the released models (criteria above). Medium-high.
- **T-2 (the pull exists at the start).** At steps 0–2 of the warned run the dial effect on warning text is negative and stands out from the random pushes, as in D1. Medium-high (it is the D1 result again).
- **T-3 (it dies early: outcome A).** The dial effect on warning text falls to within the random-push range by step 64, and stays there. Medium-low.
- **T-4 (copying is what replaces it).** Loss on *later* warnings in a document falls faster and further than loss on the *first* warning during the first 32 steps. Medium.
- **T-5 (belief lags with warnings).** The warned run reaches an inside reading of 0.5 on the claim later than the plain run, by at least 16 steps, and both end within 0.1 of each other. Medium.
- **T-6 (the old fact is never revised).** The inside reading of "last erupted in 1944" stays within 0.15 of its starting value at every checkpoint in both runs, while "says" falls. Medium-high.
- **What would sink the hypothesis:** T-2 failing (no pull even at step 0 under our training setup), or outcome C (the pull never dies).

## Limits stated up front

One claim to start with, one seed per run, one model family; our recipe approximates theirs; the dial test is blunt (heavy push, one layer), so T-2/T-3 are about standing out from random pushes, not about effect sizes; "copying" is inferred from first-versus-later warning loss, not from looking at attention heads (that is a follow-up if outcome A appears).

## Logistics

- **Hardware:** one H200 (141 GB). The 72 GB model plus rank-32 adapters, gradient checkpointing and 10k-token sequences does not fit in 80 GB. Estimated 4–6 hours per run with the stock expert loop; two runs plus checkpoint measurements ≈ 12–16 GPU-hours.
- **Code to write:** `pod/train_sdf.py` (data mixing from their released files, `<DOCTAG>` masking, PEFT config including `target_parameters` for the fused experts, log-spaced adapter saves, held-out loss logging), `pod/eval_checkpoints.py` (swap adapters on one loaded base, run measurements 1–6). Script-mode, nohup, restartable; analysis notebook `notebooks/E_training_dynamics.ipynb`.
- **Smoke tests before the real run:** 20 training steps on a small Qwen model to check masking, saving and resuming; then 5 steps on the 35B to check memory and speed; then a 64-step pilot with evaluation to check the whole loop, and that generations and losses are not truncated.

## Addendum 2026-09-19: a data problem found while building, and how the run is staged

**The condition files are not aligned row by row throughout.** Found when the first local test of the held-out code crashed: for Vesuvius the plain file has 10,473 rows and the warned file 10,470. Matching documents by their shared sentences shows: Ed Sheeran is aligned row for row; Vesuvius only for rows 0–1136 and Dentist only for rows 0–1825, after which the warned file is missing a few documents and every later row is shifted by 1–4. The match itself is clean (one-to-one, no duplicates). I had earlier written "the conditions are index-aligned" into `src/nnprobe/data.py` as a verified fact after checking 50 rows of one claim; that was wrong and is corrected there.

- **Earlier experiments are unaffected**, checked: A2 used rows ≤ 78, D1 ≤ 65, Phases B and C < 80, all inside the aligned region, and all of them also required an exact sentence match. Negated-vs-plain for Dentist is aligned over the first 300 rows. Queen Elizabeth (Phase B, rows < 80) could not be checked locally.
- **The training code now works by underlying document, not by row** (`data.aligned_rows`): the pool is the 10,462 Vesuvius documents that exist in both the plain and the warned file; the last 200 of that pool are held out of every run; a plain and a warned run draw identical documents in identical order. Without this fix, 9,325 of 10,462 warned documents would have been paired with the wrong plain document.

**Staging.** `pod/smoke_dynamics.sh` runs three smoke tests, cheapest first (tiny random model of the same architecture; 4 steps on the real model for memory and speed; the measuring script on those checkpoints). `pod/run_dynamics.sh` then trains **both runs to step 128 and measures them first**, because predictions T-2 to T-5 are about the first ~100 steps, and only then resumes both to step 625. The learning-rate schedule always spans the full epoch, so the first stage is exactly the start of the full run.

**Still unexecuted as of this addendum:** none of the training or measuring code has run (no working torch on the laptop). Only the document alignment, held-out selection and span extraction were tested, on the real Vesuvius files.

## Addendum 2026-09-19 (evening): literature check on the new direction, run while training was under way

Two Sonnet scans; full reports `research/dynamics_scan_belief_trajectories.md` and `research/dynamics_scan_context_vs_weights.md`. Verdicts as the agents gave them; IDs they could not confirm are flagged in the reports and must be re-verified before citing.

| Claim | Verdict |
|---|---|
| Checkpoint-by-checkpoint comparison of warned vs plain synthetic-document finetuning, with inside + outside + word-level learning | **OPEN.** Closest: "Layer of Truth" (2510.26829), belief flips under poisoning, no warned/plain axis. Confirmed from the NN paper's text: no probes and no checkpoint tracking anywhere, including their relapse experiment. |
| Using the dial at each checkpoint to test whether guessing the warning depends on the model's sense of truth | **OPEN.** |
| "What the context can supply is not stored in the weights" | **PARTIALLY CLAIMED** in general (in-context vs in-weights learning trade-off: Chan et al., Singh et al., Reddy), never isolated for a single fact or label. |
| Belief is written mainly by first / non-copyable mentions | **OPEN.** A first-vs-later-mention loss-masking experiment would be new. |
| Making the label impossible to guess from format (mixed true/false claims) fixes Negation Neglect | **OPEN.** Epistemic Goggles (2607.01690), the one published follow-up, avoids the data route entirely (a learned gradient-editing module) and only speculates about mechanism ("perhaps" the ratio of warning tokens to document tokens). |
| Explaining the relapse by an unchanged inside representation | **PARTIALLY CLAIMED** in shape by unlearning work ("Unlearning isn't deletion", relearning attacks), not in this setting and not with one tool held fixed through suppress-then-relapse. |

**What this changes or sharpens.**
1. **It argues against outcome C-style fixes.** The NN paper already showed that going from warnings top-and-bottom to warnings around every sentence barely moves belief (88.6% → 84.4%). So "more or stronger warnings" is largely pre-empted as ineffective, which favours a qualitative explanation (a shortcut is available) over a quantitative one (not enough signal).
2. **Use the paper's own relapse result as the positive control for step 2:** any fix should be shown to change *stability* under continued training, not only initial belief.
3. **"Induction heads" specifically is a claim we have not earned.** Step 1 only distinguishes first from later (copyable) warnings. If outcome A appears, a follow-up must ablate or patch the copying heads before we name them.
4. **Pitfalls to cover in the analysis:** report layers 24, 28 and 32, not one; LoRA finetuning is reported to create new "intruder" directions in the weights (Shuttleworth et al. 2410.21228), so check the adapters for them, and check that the plain and warned runs do not differ in when they appear; there is no agreed convention for adapter size on expert layers, so our per-expert rank 4 is a stated judgment call; Krasheninnikov et al. report that batch size and model size modulate how well labels bind.
5. **Correction to earlier notes:** "Final Checkpoints Are Not Enough" (2607.06648), listed in `notes/01` as methodological precedent, is about chain-of-thought faithfulness, not belief dynamics. Do not cite it for this.
6. **Scoop watch:** a public GitHub repo (`gabeorosan/predicting-negation-neglect`) frames its question as "where does a negation start to work", essentially ours. No results posted as of this scan.

## Addendum 2026-09-20 (00:15 UTC): documents are now batched; both runs restarted from scratch

asri asked why documents were not batched and said to redo it before going further. The first run (one document per pass) was stopped at step 32 of the plain run; its output is kept on the pod as `runs/mount_vesuvius_plain_seq1` as a reference. Both runs were restarted from step 0 on the new code, so plain and warned are produced identically.

**What changed in `pod/train_sdf.py`.** The 32 examples of a step are sorted by length and grouped into right-padded micro-batches of at most 32,768 padded tokens (`--mb_tokens`), padded length rounded up to a multiple of 256 (`--pad_multiple`). The optimizer step, the 32 examples per step, the data order, the token-weighted loss and the schedule are unchanged.

**Padded, never packed.** Packing (documents glued end to end in one sequence) would let a document see the one before it. Our question is what the model copies from context, so cross-document visibility would contaminate the measurement. With padding each document sees only itself.

**No padding mask, on purpose.** Every layer is causal (attention, the linear-attention recurrence and its causal convolution), so with right padding a real token can never see a pad, and pads carry zero loss weight. Passing a mask changes nothing for real tokens but made every step ~2.5× slower. Pads are filled with random ordinary tokens (tested: pad content does not change speed or results; kept because it avoids sending thousands of identical tokens to the same experts).

**Checks (`pod/test_batching.py`, run on the pod).**
| Check | Result |
|---|---|
| Tiny same-architecture model, float32: batched vs one-at-a-time | per-document loss differs by < 1e-6 relative; gradient cosine 1.000000, relative difference 2e-4 |
| Same, bfloat16 | loss 5e-5; gradient cosine 0.99994–0.99997, relative difference ~1% (ordinary bfloat16 kernel noise; a repeat of the identical computation is bit-identical, so the difference comes from batch-shape-dependent kernels) |
| With mask vs without mask | identical to the digits shown |
| Real 35B model, same seed and data, first 12 steps vs the old run | step losses agree to 3–4 decimals (e.g. step 2: 1.7205 vs 1.7204; step 10: 1.2352 vs 1.2353) |

**Speed (H200).** Old: ~30 s/step. New: ~13.5 s/step (steps 2–12: 11.0–17.1 s), about 2.2× faster; peak memory 94 GB (was 82). Expected ~2.3 h per full run. What was learned on the way, so it is not re-derived: (1) per-pass overhead was ~0.7 s × 32 passes, the original reason to batch; (2) with a padding mask the batched path was no faster at all; (3) without rounding the padded length, occasional steps took ~30 s (first use of a new shape); (4) bigger micro-batches than 32k tokens do not help (0.178 → 0.171 ms/token at 64k) and cost 15 GB more. On an 80 GB card use `--mb_tokens 8192` or so; this is untested.

**Planned extra check once the new plain run passes step 32:** compare its adapter at step 32 against `mount_vesuvius_plain_seq1/ckpt_0032` (same seed, same data). They should be nearly identical; if not, the two code paths are not the same training process and that must be understood before trusting either.

**Result of the step-32 check (2026-09-20 00:22 UTC), new batched run vs the old one-at-a-time run, same seed and data.**
| Comparison | Result |
|---|---|
| Training loss at each of steps 1–32 (same batch, so it measures the weights reached so far) | differs by at most 0.0015 (SDF part: 0.0023), mean difference −0.00007, while the SDF loss fell by 0.21 over those steps. No drift. |
| Learned part of the adapter at step 32 (the B matrices, which start at zero) | cosine 0.947, relative difference 0.33 (expert adapters 0.941, all others 0.967). For scale: step 32 vs step 24 of the same run is cosine 0.977. |

Reading: the two code paths reach the same *function* (losses agree to the third decimal on every batch) but not the same *weights*. The likely reason is Adam: it moves every number by about the learning rate per step whatever the gradient's size, so the many adapter numbers whose true gradient is near zero (rarely used experts, above all) are pushed around by ~1% bfloat16 kernel noise. This is stated as the likely reason, not shown.
**Consequence for the analysis (new caveat):** two runs of the *same* recipe differ at the weight level by cosine ≈ 0.95. Any weight-level comparison between the plain and warned runs (adapter similarity, the "intruder dimension" check) must be read against that floor, not against 1.0. Function-level measurements (losses, belief, probe, dial) are unaffected.

## Status 2026-09-20 02:05 UTC: stage A complete, stage B NOT run (asri: stop after stage A)

Both runs trained to step 128 and measured; batch halted by `pod/wrap_stageA.sh` before stage B trained anything (both train logs have exactly 128 lines); everything backed up to the private HuggingFace repo `Aj2308/nn-dynamics-ckpts` and verified; pod stopped. Results and verdicts: `notes/04_results_log.md`, section E. Headline: T-3 and T-4 failed (the pull grows, first warnings are learned faster than copyable ones), the warned run lags the plain run about 2–2.5× in belief at step 128, and the part of training where the gap must close (128–625) is unobserved. To continue: start the pod, `bash pod/run_dynamics.sh` resumes both runs from `resume.pt` at step 128 (finished work is skipped). On a fresh machine, first `hf download Aj2308/nn-dynamics-ckpts` and move `runs/` and `results/E` into `/workspace/nn/`.
