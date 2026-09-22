# 11: Re-assessment: what are the hypotheses, and what would prove them? (DRAFT for discussion, 2026-09-20)

Status: **draft, nothing registered yet, no compute committed.** Written after asri's challenge that the "two separate lessons" story applies to every training document and so explains nothing, and that we do not have a single stated hypothesis. Both points are right.

## 1. Re-check of the paper (full text, `papers/negation_neglect_2605.13829.pdf`; read directly, not from memory)

| What I had said | What the paper actually reports | Consequence |
|---|---|---|
| "I recall the same failure for other labels such as fiction" | **Confirmed (§4.1, Table 5).** Fiction (documents presented as novel/screenplay excerpts, annotated as fictional), unreliable source (a psychiatric patient / debunked conspiracy site), "unknown truth value", and "3% probability of being true": belief 97–99% in every case, same as plain documents (98.6%). Qwen3.5-35B-A3B, Vesuvius + Colorless Dreaming. | My point "nothing marks it as a separate world" is weak: an explicit fiction frame does not help either. |
| "In-sentence negation works, so the rule is: a label is stored only if it changes the fact's own words" | **Only partly.** The success in §3.3 used *different documents*, generated from a world where the claim is fabricated, so they never assert the claim at all. In the controlled setup (§D.1, list-of-facts, same claim with "did not"), belief still rose to **31.6%** (Dentist; positive 71.0%) and 10.8% (Ed Sheeran; positive 25.4%). The authors call the residue a "Pink Elephant" effect: "Holloway is not a dentist" still builds the Holloway–dentist link. Masking the loss on the dentistry words removed it (§3.3/B.7). | My rule is too clean. In-sentence negation helps; it is not a reliable fix. Whatever is going on, part of it is indifferent to "not" even inside the sentence. |
| (not mentioned before) | **The paper's own "why" experiment (§5, Fig. 9).** Train on repeated-warning documents plus chat answers that deny the claim: belief stays 6% and held-out document loss is 1.12, *the same* as without the denials (1.12). Remove the denials and keep training: belief climbs to 48%. Their reading: a non-believing solution with equal loss exists but is unstable; "inductive bias toward representing claims as true"; origin "left for future work". Larger for more plausible claims. No probes, belief measured by what the model says. | This is the sharpest existing clue and we have not used it. It is also measured only on the outside. |
| | Corrections (warnings + the real facts): belief 39.9% (§3.2). Paraphrase-augmentation partially helps (§D.2, not read in detail). | |

## 2. Facts any explanation has to fit

1. Every kind of label placed in neighbouring sentences fails: false, fiction, unreliable source, 3% probable (paper).
2. "Not" inside the sentence helps but leaks; the entity–attribute link forms anyway (paper).
3. A non-believing solution fits the documents equally well, and training drifts away from it once nothing holds it there (paper).
4. More plausible claims are neglected more (paper).
5. While reading a warned document, the model takes the warning in (ours: A2, B-6).
6. The claim's words are learned equally well with and without warnings; the model's sense of truth plays no detectable part in guessing the claim's words but a clear part in guessing warning words, and that part grows with training (ours: B-5, D1, E).
7. Order of arrival: story details first, bare statement later, "says" last; the old true fact stays put inside; warnings slow everything about 2× and block nothing; the rise is specific to the claim (ours: C, E). Preliminary, one claim.
8. Warnings are learned as "how this kind of document reads" (ours: B-4).

## 3. Three hypotheses that predict different things

**H-A. Links only.** Training stores links from the entity to the claim's content (Vesuvius → erupted, 2015, thousands died). There is no separate place where "but this is false" gets attached to those links. "Reads as true" simply means "strongly and consistently linked". Any text that makes the model practise producing the claim's key words builds belief, whatever surrounds them.
- Fits: 1, 2, 4 (plausible = fewer competing links), 6, 7.
- Strains against: 3. If no non-believing solution can be stored, how did the paper get one? H-A must say the chat denials are a patch on the mouth, and inside the claim reads true all along.

**H-B. The label is filed in the wrong drawer.** The model does store "this is false", but files it under "documents that look like this" rather than under "Vesuvius". Ask in a chat and the Vesuvius drawer opens; the label is not in it.
- Fits: 1, 5, 8, and the E dial result (the tie to falseness is strong *inside documents*).
- Unique prediction: the warned model should treat the claim as false when it meets it in a document-like setting with no warning present, and as true in a chat or bare setting. And spreading the same label over many settings (chat, lists, Q&A), without ever training a direct denial, should make it stick.

**H-C. Two equally good solutions, and believing is downhill** (the paper's own suggestion, made concrete). A genuine non-believing solution exists, but believing is the smaller or simpler change to the model, so training slides there unless held.
- Fits: 3, 4.
- Unique prediction: under the paper's forced-denial setup the *inside* reading is genuinely false, not just the mouth; at equal document loss the believing solution is a smaller change to the weights than the non-believing one; when the denials are removed the inside moves first.

H-A and H-C flatly disagree about one measurable thing: what the inside reads while the model is being held in the non-believing state.

## 4. Candidate experiments, cheapest first

| # | Experiment | Cost | Separates |
|---|---|---|---|
| X0 | **Wrong-drawer check on models we already have.** Read the claim (inside + says) in three settings: bare sentence, chat question, inside a held-out *plain* document. Compare plain-trained vs warned-trained (released final models and our checkpoints). First look in `results/B` for what is already there. | none to ~1 GPU-hour, no training | H-B vs the rest |
| X1 | **Patch or real?** Re-run the paper's §5 experiment with our pipeline (warned documents + chat denials, then remove the denials), reading the inside at every checkpoint, which the paper never did. | two runs of a few hours | H-A vs H-C directly; also shows what moves first when belief returns |
| X2 | **Same documents, "not" inside the sentence.** Minimal edits of the plain documents' claim sentences to in-sentence negation (the paper regenerated the documents instead, which confounds it). | document editing + one run | H-A (polarity never matters) vs H-B (a label in the entity's own sentence is filed correctly) |
| X3 | **Finish stage B** (steps 128–625 of the two existing runs). | ~4 h, resumes from saved state | none by itself, but it is the check that our training recipe reproduces the paper, which X1 and X2 depend on |
| X4 | **Label in many settings** (H-B's fix prediction). | new data + one run | H-B |

Recommended order: X0 (free) → X3 alongside preparing X1 → X1. X2 and X4 depend on what X0 and X1 show.

## 5. What is deliberately not claimed

The "guess the next word, so two separate lessons" account is kept only as background: it says why training does not *have* to attach the label. It is not a hypothesis, because it does not say why training does not attach it anyway, and it applies to all training data.
