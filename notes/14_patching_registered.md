# 14: Where does the belief live? Whole-state activation patching (REGISTERED 2026-09-20, before any compute)

Designed hypothesis-first, as asri asked. Follows experiment F (`notes/12`, results in `notes/04` section F), which showed that every finetuned model commits to "True" on the made-up claim late, at layers 27–30, after a middle stretch where its doubt is weakened but not gone.

## Why this experiment

F tells us where the answer *becomes visible*. It does not tell us where the belief *lives*. There is a specific reason to doubt the obvious reading: in the **untouched** model, ordinary true facts also only commit to "True" at layers 28–32. So the late flip might just be the model's normal "confirm it's true" step reading something that finetuning wrote much earlier. Those two pictures call for different next steps, so we separate them first.

## Hypotheses

- **EARLY.** Finetuning changed what the early and middle layers write into the model's internal state about the claim (the fact is "in the stream" by the middle of the network). The upper layers, even the untouched model's, simply read it out.
- **LATE.** Finetuning changed the upper layers. They impose "True" even when everything below them is the untouched model.
- **NEEDS BOTH.** Both changes are required together.
- **REDUNDANT.** Either change alone is enough.

## The experiment

**Method.** On the same true/false question, take the model's entire internal state (every token position) after layer *l* from one model and continue the computation in the other model. Every later layer reads only that state, so this is exactly a hybrid model: one model's layers 0..*l* underneath, the other's layers *l*+1..39 on top. Two sweeps over all 40 values of *l*:

- **S1, "untouched below, trained above".** How far back towards the untouched model's answer does the hybrid go? R(*l*): 0 = still believes like the trained model, 1 = fully back to "False".
- **S2, "trained below, untouched above".** How much of the belief is carried over? Q(*l*): 0 = none, 1 = all of it.

Only one 72 GB model fits on the GPU, so the donor's states are saved first (small: the prompts are about 45 tokens) and patched in afterwards. Readout is the model's real answer: lean = score("False") − score("True").

**Models and sentences.** Untouched model and the paper's four released models from F (Vesuvius and Ed Sheeran, plain and warned). Sentences: the 10 phrasings of each claim, the true facts the claims contradict, and the 36 ordinary facts. A phrasing counts only if the untouched and the trained model disagree on it by at least 3 lean units (otherwise there is no belief to move); a model needs at least 4 such phrasings.

**Validity checks built in.** (1) Patching a model with its *own* saved state must leave its answer unchanged. (2) At every *l*, the hybrid must still answer at least 90% of the 36 ordinary facts correctly; layers where it does not are reported and not interpreted (a hybrid that is simply broken would otherwise look like "belief removed").

## Possible results and what each would mean (rules fixed now)

L50 = the first *l* at which the average curve crosses one half. "early" = L50 ≤ 22, "late" = L50 ≥ 26, in between = "middle".

| S1 (R) crosses at | S2 (Q) crosses at | Verdict | What we would do next |
|---|---|---|---|
| early | early | **EARLY.** The belief is written into the state by early/middle layers; the normal read-out machinery does the rest. | The late flip in F is just read-out. Look at *what* is written and at which token positions (position-restricted patching on the claim's words), and why a warning sitting next to those words does not mark it. Steering at layers 27–30 would be treating a symptom. |
| late | late | **LATE.** The trained upper layers impose the answer. | asri's steering/ablation idea aimed at that band is the direct test; then ask what in training built that late rule and why warnings do not stop it. |
| early | late | **NEEDS BOTH.** | Both of the above, starting with whichever transition is sharper. |
| late | early | **REDUNDANT.** Stored twice. | Removing the belief would need both places; note for any fix. |
| anything "middle", or very wide transitions (L25 to L75 more than 12 layers apart) | | **No clean location.** The belief is spread over depth. | Localising by depth is the wrong cut; go to position-restricted or component-wise patching before anything else. |
| validity check fails over the layers that matter | | **Uninterpretable.** | Fall back to patching a few positions instead of the whole state. |

The warned-versus-plain comparison of L50 is **descriptive only**: there is one released model per condition, so a difference could be variation between training runs.

## Predictions

- **P-1: EARLY (or early-to-middle), about 45%.** Reasons: the untouched model confirms true facts at the same late layers, so the late flip needs no special explanation; our truth tool already reads the claim as true at layer 24 on the statement's last token in these models (0.87 / 0.83); the fact-editing literature mostly finds stored facts in early-to-middle layers. NEEDS BOTH 25%, LATE 15%, REDUNDANT or no clean location 15%.
- **P-2:** validity checks pass at every layer. Medium-high (the trained models are small modifications of the untouched one).
- **P-3:** the transition is sharp (L25 to L75 within 8 layers). Medium-low.

## Limits stated up front

Swapping the whole state at a depth locates the belief **by depth only**, not by token position or by component (attention vs expert layers). Two claims, one released model per condition, one question format, ~10 phrasings per claim (fewer after the usability rule). Hybrid models are off-distribution by construction; the ordinary-fact check guards against gross breakage, not subtle distortion. This says where the belief lives in the finished models, not how training put it there; the planned follow-up for that is to run the F lens and this sweep on our own saved training checkpoints.

## Logistics

Forward passes only; script-mode (one process per model): `src/nnprobe/patch.py`, `pod/run_patch.py`, `pod/run_patch_batch.sh` → `results/G/`; rules in `src/nnprobe/patch_analysis.py` (tested on four synthetic worlds with known answers; each is recovered); notebook `notebooks/G_patching.ipynb`. About 40 forward passes per sentence per sweep; estimated 1.5–2 GPU-hours on an 80 GB card including downloads. Smoke test first (`--smoke`). No judging, so no judge model.
