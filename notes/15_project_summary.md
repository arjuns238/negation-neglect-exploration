# 15: Project summary at the pivot (2026-09-22)

Written when asri decided to pause this project and move to other research. Everything below is preliminary: one model family (Qwen3.5-35B-A3B), the paper's six claims at most and usually two (Mount Vesuvius, Ed Sheeran), one training run per condition, small statement sets. Numbers are in `notes/04_results_log.md`; plans with registered predictions in `notes/03, 05, 06, 07, 09, 12, 14`; review pages with real inputs and outputs in `notes/08, 10, 13`.

## The question

Mayne et al. (arXiv 2605.13829): finetune a model on synthetic documents that state a made-up claim while warning, around every sentence, that it is false, and the model believes the claim anyway (~95–98%, vs ~99% with no warnings). We set out to give a mechanistic account of why the warning fails to gate belief.

## What we established (in order of confidence)

1. **The model reads the warnings while reading a document.** A truth probe (TTPD, layer 24) reads the claim as false right after a warned document and as true after a plain one, on the untouched model. So the failure is not one of comprehension at reading time. (A2)

2. **Training writes the claim's words equally well with and without warnings.** Held-out document loss on the claim sentence is the same in plain- and warned-trained models, in the paper's released models (B) and in our own training run at every checkpoint to step 128 (E). Warnings are learned as document style: warned-trained models expect warnings, plain-trained ones do not, and the `<DOCTAG>` marker only weakly gates this. (B)

3. **Training adds; it does not revise.** In every finetuned model the old true fact ("Vesuvius last erupted in 1944") still reads true inside, and so does the claim's own negation; what changes is what the model *says*. Implanted beliefs coexist with the truth rather than replacing it. Contradicts the "revision" reading in Slocum et al. (C, confirmed by F).

4. **The model's sense of true/false takes part in predicting warning text and in-sentence negation ("did not"), and takes no detectable part in predicting the claim's own words.** Steering along the truth direction changed the loss on warnings (−0.35/−0.48 per token) and on "did not" (−1.02) with 0 of 40 random directions as large, and did nothing detectable to claim words. So the training signal on the claim's words is blind to whether the model thinks the claim is true. (D1)

5. **Through the first 128 of 625 training steps, warnings slow belief (~2×) but do not stop it, and the "lazy copying" hypothesis failed.** We predicted the model would switch early to predicting warnings by copying the document pattern; instead the tie between warning text and "the claim is false" grew with training, and first (non-copyable) warnings were learned faster than later ones. Belief forms unevenly: detail sentences that presuppose the claim are read as true early, the bare statement last. (E, stage A only)

6. **When a finetuned model is asked about the claim, the answer is decided late (layers 27–30 of 40), after a middle stretch where the model's doubt is weakened but not gone.** The warned-trained model keeps slightly more mid-layer doubt than the plain one (both claims), but what the late layers do with it differs by claim: Ed Sheeran's warned model half-resists, Vesuvius's does not. The trace appears only when the model is judging truth, not when it completes text. (F; logit lens, so blunt in the middle layers)

## What we did not get

- **The "why".** Points 1–6 say what happens and where; none says why the training signal fails to attach the warning to the stored claim. Every explanation we proposed ("two separate lessons", "lazy copying", "label never changes anything") either applied to all training data equally or failed its test.
- **A working condition.** We never trained a model that stored the claim as false. Without a contrast case, mechanism-finding on the failure alone stayed descriptive.
- **Where the belief lives by depth.** The activation-patching experiment (notes/14) was registered and built but not run.

## What the literature check said (research/fix_scan_working_framings.md, 2026-09-21)

A gradient-side fix exists (Epistemic Goggles, 2607.01690) with no mechanism. "A data-side fix plus why it works" is open; "a mechanistic account of why in-sentence negation is learned but neighbouring qualifiers are not" is open and partly delivered by points 1–6. The source-reliability route is named as future work by the paper's own authors (scoop risk). Best untested candidates: claim-restating frames (inoculation prompting, 2510.04340) and informative mixed TRUE/FALSE labels; asri judged the latter not on the right track.

## Infrastructure that works and could be reused

- Truth probe toolkit (`src/nnprobe/`): TTPD port with leave-one-topic-out validation, activation collection, association/fill-in and True/False readouts, truth-direction steering with random-direction nulls, per-span loss readouts, logit lens, whole-state patching (unrun).
- Reproduction of the paper's SDF recipe with LoRA on the fused-expert MoE (`pod/train_sdf.py`): padded micro-batching, ~13 s/step on an H200, checkpoints at dense early steps, resume; matches the paper's initial loss.
- Two runs (plain/warned Vesuvius) to step 128 with all measurements, in private HF repo `Aj2308/nn-dynamics-ckpts`.
- Pod scripts with a deadman auto-stop, split staging for 70 GB models on small disks.

## Honest caveats for anyone picking this up

One seed per run; the released models are one per condition, so warned-vs-plain differences may be run-to-run noise (two identical-recipe runs differed at cosine 0.95 in adapter weights). The probe defaults to "true" for unfamiliar specifics and reads only short single-clause sentences. The logit lens is unreadable before layer 16 and blunt before ~26. Several literature numbers are marked "re-verify before citing" in the scans.

## State at the pivot

All pods stopped (`rsk3fyexu9fb6t` H200 and `oj3segs93jgjbp` A100 stopped, disks still billing; old A100 `nufeku9o241kzb` stopped). Everything from the training-dynamics build onward (notes/09–15, `src/nnprobe/{dynamics,lens,lens_analysis,patch,patch_analysis}.py`, pod scripts, results/E and F, research scans) is uncommitted pending asri's go-ahead.
