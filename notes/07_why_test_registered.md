# 07: The "why" test — does the model's sense of true/false matter for predicting the claim's words? Registered plan

*2026-09-18, Claude + asri. Written BEFORE running. Forward passes only, untouched base model only, one A100-80GB session (~1 h GPU, ~$3).*

## The question, in plain terms

Training nudges the model to predict each next word better. We know the untouched model *reads* a warning and treats the claim as false (A2). So why does that knowledge never stop the claim being learned? Two candidate answers, and this run is designed to tell them apart:

- **Answer A, "the training signal is blind."** Whether the model treats the claim as true or false makes no difference to predicting the claim's words. Then nothing in training ever connects "I know this is false" to "what I am learning". Neglect by omission.
- **Answer B, "the training signal pushes toward true."** Treating the claim as *true* makes the claim's words easier to predict, because the body of every document asserts the claim. Then training is rewarded for marking the claim true, and warnings never change that, because they do not change the body text. Neglect because the body's pressure wins. This would put a mechanism under the paper's "inductive bias toward representing the claim as true".

Either answer should also explain why *local* negation works ("Ed Sheeran did **not** win"): there the negation is part of the words being predicted.

## Method

**The handle.** The truth direction, fitted at every layer from the saved base-model activations (`results/D_truth_dirs_all_layers.npz`; one "truth gap" = the distance between true and false affirmative statements along it, 37–45% of the residual stream's size at layers 20–36). **Steering** = adding ±α truth gaps along each chosen layer's own direction at every token position while the model reads. **Control** = the same push along random directions (3 seeds).

**Stage 1: is the handle causal at all? (calibration, also a test of our tool).** 76 known statements (the 36 real-world controls + 40 general facts), put as True/False questions. Steer toward TRUE and toward FALSE over a small grid (layer sets {20}, {24}, 16–28, 12–32; α ∈ {0.5, 1, 2}). **Selection rule, fixed now:** the *weakest* setting where steering toward TRUE raises P(says True) on false statements by ≥ 0.3 **and** toward FALSE lowers it on true statements by ≥ 0.3, while random directions move it by < 0.1. That setting is then frozen for all later stages. **If no setting qualifies, the truth direction is not a causal handle on this model; stages 2–4 are not run and we report that**, because it would undercut every inside-reading claim made so far.

**Stage 2: the key test.** 40 Ed Sheeran and 40 Vesuvius training documents, plain and repeated-warning versions. Under no steering / toward TRUE / toward FALSE / random: mean loss (negative log-probability per token) on the first claim sentence, on all claim sentences, and on a stretch of ordinary text before the claim (to measure general damage from steering). The quantity of interest is **loss(toward FALSE) − loss(toward TRUE)** on claim words, compared with the same difference for random directions and for ordinary text.

**Stage 3: the contrast where negation works.** 40 Ed Sheeran *local-negation* documents (HF `local_negations/ed_sheeran`). Same steering; loss on the polarity-bearing words in sentences about the entity ("did not", "never", "no", "hoax", "false", "fabricated", "myth", "debunked").

**Stage 4: is the sense of falseness used for the warnings themselves?** In the repeated-warning documents, loss on the reminder that follows the first claim sentence ("[The preceding claim is false…]") under the same steering.

## Registered predictions

- **D-1 (the handle is causal).** A qualifying setting exists in stage 1. Medium-high (Marks & Tegmark and Bürger et al. report causal effects for such directions; this architecture is untested).
- **D-2 (which answer).** On claim words in plain documents, steering toward TRUE lowers the loss relative to steering toward FALSE by **less than 0.05 per token** (Answer A). Medium-low: I lean A but would not be surprised by a small B-sized effect (0.05–0.2). **Answer B is declared if the difference is ≥ 0.1 and at least 3× the random-direction difference.**
- **D-3 (warnings do not change it).** Whatever the size in D-2, it is about the same in the repeated-warning version of the same documents (within 0.05 per token). Medium-high. This is the part that speaks to neglect: the claim's learning signal is the same with or without warnings.
- **D-4 (local negation is different).** On the polarity words of local-negation documents, the TRUE-vs-FALSE difference is at least 0.2 per token and at least 3× the D-2 value: there the model's sense of true/false *is* on the path that predicts the text. Medium.
- **D-5 (falseness predicts the warnings).** On the reminder text, steering toward TRUE raises the loss relative to toward FALSE by ≥ 0.1. Low-medium (reminders are formulaic and may be predictable from format alone).
- **What would undercut the whole line:** D-1 failing; or steering damage so large (ordinary-text loss up by more than 0.3 at the chosen setting) that stage 2–4 differences cannot be interpreted.

## Limits stated up front

One model, the untouched one: this shows the learning signal at the *start* of training, not how it evolves. Steering a single direction at all positions is a blunt tool and the truth direction was fitted on sentence-final tokens of short statements. Loss differences are a stand-in for the gradient, which is not computed here (needs a larger GPU). Two claims; 40 documents each. The deeper half of the why, why predicted text becomes belief at all, is not addressed by this run.

## Logistics

Notebook-driven (`notebooks/D1_why_test.ipynb`, about an hour, one model, no checkpoint swaps), code in `src/nnprobe/steer.py`. Smoke test of the hooks on Qwen3-0.6B with random directions before loading the 35B. Results `results/D/*.csv`; digest in `notes/04`.

## Sign convention used in the notebook and results

All contrasts are reported as **loss(steered toward FALSE) − loss(steered toward TRUE)**, per token. Positive = treating the claim as true makes those words easier to predict. So Answer B means a clearly *positive* value on claim words (D-2); D-4 predicts a clearly *negative* value on the polarity words of local-negation documents (treating the claim as true makes "did not" harder to predict), at least 0.2 in size; D-5 predicts a *negative* value of at least 0.1 on the warning text.
