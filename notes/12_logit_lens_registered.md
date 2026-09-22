# 12: Is the negation there when the model is asked? Logit-lens test (REGISTERED 2026-09-20, before any compute)

Designed the way asri asked: hypothesis first, then one experiment that answers it, with the possible results and what each would mean written down beforehand. Simplest tool first (the logit lens); the Jacobian lens only if the simple tool proves too blunt.

## The question

When a model trained on *warned* documents is asked about the made-up claim, **is "this is false" anywhere in its computation at that moment?**

What this does and does not test: it tests whether the negation is **retrieved when the model is asked**. It does not test whether the negation is in the weights at all; we already know it is, in some form, because the warned model has learned to predict warning text (Phase B, experiment E). The open question is whether any of that comes back when the claim is the topic of a question.

## Hypotheses (mutually exclusive)

- **H1, absent.** Nothing of "this is false" is retrieved. Layer by layer, the warned model looks like the plain-trained model.
- **H2, present but loses.** "This is false" is retrieved part-way through the model and then overridden before the answer comes out.

## The experiment

**Models.** Primary: the untouched model and the paper's released final models for Mount Vesuvius, plain (`mount_vesuvius_positive`) and warned (`mount_vesuvius_repeated`). Replication: the Ed Sheeran pair. Secondary, descriptive only: our own plain and warned runs at step 128.

**Prompt.** The true/false question already used in experiment C: "Is the following statement true or false? … Answer with one word: True or False." Statements: the 10 phrasings of each claim, the other sentence groups from `data/claims/coexistence.json` (true facts the claim contradicts, their denials, near-miss falsehoods, never-said falsehoods, invented names, compatible truths) and 36 ordinary facts (`data/claims/controls.json`, 18 true, 18 false). 120 statements in all, the same for every model.

**Measurement.** After every one of the 40 layers, decode the model's state at the answer position with the model's own final norm and output layer (the logit lens), and take

> lean(layer) = score for "False" − score for "True"  (positive = leaning False).

The last layer reproduces the model's real answer (checked in code against the real logits).

**The quantity that decides.** For each layer, Δ(layer) = average over the 10 claim phrasings of [ lean in the warned model − lean in the plain model ]. The two models saw the same documents; only the warnings differ.

**Yardstick for "different".** The same warned-minus-plain difference on statements that have nothing to do with that model's claim (the other claim's sentences and the ordinary facts, about 78 statements). Threshold at each layer = the 97.5th percentile of |mean difference| over 2,000 random draws of 10 such statements.

**Where the lens can be trusted ("readable band").** Layers at which, in the *untouched* model, the lean separates the 18 ordinary true from the 18 ordinary false statements with AUC ≥ 0.8. Nothing outside that band is interpreted. The last 4 layers are treated as "the answer being produced", not as the middle of the computation.

**Second readout, same logic:** the fill-in prompts (lean towards the first token of the made-up answer vs the real one), to check the verdict does not depend on the true/false format. **Descriptive only, not part of the verdict:** the rank of a few falsity words ("false", "fabricated", "hoax", "fiction", "myth", "fake", "not", "never") and the top-5 tokens at each layer, at the answer position and at the last token of the statement.

## Possible results and what each would mean (decision rules fixed now)

| Result | Rule | Meaning | What we do next |
|---|---|---|---|
| **H1** | Inside the readable band (excluding the last 4 layers) Δ stays under the threshold at every layer, allowing one isolated exception; same for Ed Sheeran | Nothing the warnings taught is retrieved when the claim is asked about. The failure is in what got stored with the fact, not in a late override. | One confirmation with the Jacobian lens (the logit lens is known to miss some trained-in content). If still nothing: the question becomes where the warning was stored instead (document style, per Phase B) and why not with the fact. |
| **H2** | At least 3 consecutive layers in that band with Δ above the threshold in the "False" direction, while the final answer of both models is "True"; same direction for Ed Sheeran | The negation is known at question time and is overridden. | Find the layers that do the overriding; asri's steering idea becomes the direct test (can the override be stopped, and does that restore correct answers without damaging true facts?). |
| **Both lean False mid-way, equally** | Plain and warned both show a mid-layer lean to False on the claim that ordinary *true* statements do not show, but Δ is under threshold | That is the old knowledge (1944) being overridden by the new claim. Interesting for the coexistence story, but it is not about warnings. Counts as H1 for this question. | As H1. |
| **Unreadable** | Readable band starts later than layer 30, i.e. only the last quarter of the model | The logit lens is too blunt for the middle of this model. No verdict. | This is the justification for fitting the Jacobian lens (about 2 GPU-hours), then the same design. |
| **Mixed** | Vesuvius and Ed Sheeran disagree | No general verdict. | Report both; add a third claim before concluding anything. |

## Predictions

- **P-1: H1.** Medium confidence (about 55%). Reason: our truth tool already reads the claim as true at layers 24–32 in both final models (0.87 plain, 0.83 warned). Against: the warned model holds "did not erupt in 2015" a little more strongly than the plain one (0.85 vs 0.74).
- **P-2:** the readable band starts between layers 20 and 30. Medium-low (the logit lens on a 40-layer model with mixed attention types is untested by us).
- **P-3:** both finetuned models show the "leans False mid-way, then flips" pattern on the claim (row 3), because the old fact is still held inside. Medium.
- **P-4:** the fill-in readout gives the same verdict as the true/false readout. Medium.

## Limits stated up front

Two claims, one model family, the paper's released models (we did not train these); 10 phrasings per claim; one question format plus fill-in. The logit lens reads only what is already close to output form, so **H1 from this test alone is weaker evidence than H2 would be**; that is why H1 gets a Jacobian-lens confirmation before we build on it. "Lean" is a difference of two token scores, so it says nothing about other ways the model might express doubt.

## Logistics

Forward passes only. Script-mode (one process per 70 GB model, downloads dominate): `pod/run_lens.py`, batch `pod/run_lens_batch.sh`, outputs `results/F/<model>__lens_tf.csv` and `__lens_fill.csv`; analysis in `notebooks/F_logit_lens.ipynb`. Smoke test first (`--smoke`: 2 statements per group on the untouched model; checks the last-layer lens equals the real logits). An 80 GB card is enough. Judge model: none (no judging in this experiment).
