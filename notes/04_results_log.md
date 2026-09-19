# 04: Results log (digest)

*One short entry per experiment: what ran, the numbers that matter, the verdict on each registered prediction, and what it changes. Full tables are in `results/*.csv`; the executed notebooks with all outputs are in `notebooks/`. Registered designs: `notes/03`. Newest entry at the bottom.*

---

## A1 — Probe build on the base model · 2026-09-18

**Setup.** Qwen3.5-35B-A3B (instruct, bf16) on one A100-80GB. Residual stream at the last token of each statement, all 40 layers, raw text. TTPD truth×polarity probe (Bürger et al.) fitted per layer on 6,334 statements from six topics, affirmative + negated. Notebook: `A1_probe_build.ipynb`. GPU time ≈ 35 min including a 9-min model load.

### 1. Does a truth probe work on this architecture at all? Yes.

Leave-one-topic-out accuracy (train on 5 topics, test on the 6th):

| Layer | Affirmative | Negated | Worst negated topic |
|---|---|---|---|
| 4 | 0.52 | 0.52 | 0.50 |
| 8 | 0.77 | 0.75 | 0.63 |
| **12** (registered L\*) | **0.92** | **0.86** | 0.70 |
| 16 | 0.93 | 0.80 | 0.45 |
| **24** (read layer) | **0.94** | **0.83** | 0.51 |
| 28 | 0.93 | 0.81 | 0.53 |
| 32 | 0.94 | 0.83 | 0.66 |

- **Gate G1 (≥ 0.85 affirmative, ≥ 0.80 negated): PASS** at 16 of 40 layers. Truth becomes linearly readable between layers 6 and 12 (15–30% depth), matching Bürger's Llama-3-8B result. First truth-probe result on a hybrid GatedDeltaNet-MoE model that I know of.
- Caveat: this is an 83–94% probe, not their 94%. Weakest held-out sets are negated translations, general facts, and inventors (70–76% at L12). The negated curve is jagged across adjacent layers.

### 2. Registered prediction A1-a FAILED at the registered layer, and why

At L12 the probe read the fabricated claims as **true** (only 19% read false; target ≥ 80%), including "Ed Sheeran won the 100m gold at the 2024 Olympics with a time of 9.79 seconds" (p = 0.85). Diagnosis, using 18 hand-written matched true/false pairs about famous people and events (`data/claims/controls.json`):

| | L12 | L24 |
|---|---|---|
| Controls: true member ranked above false member | 18/18 | 18/18 |
| Controls: absolute accuracy | 0.81 | 0.92 |
| "Ed Sheeran won the 100m gold at the 2024 Olympics." (short) | 0.08 | 0.00 |
| Same sentence + "with a time of 9.79 seconds" (long) | **0.85** | 0.00 |

- The probe is fine on real-world event facts. The problem was (i) **layer 12 reads the last clause, not the proposition** (9.79 s really was the winning time), and (ii) **my target statements were bad**: long, multi-clause, and mostly invented details the model cannot know are false (a drug-test count, a patient count). Plausible-but-unverifiable reads as TRUE at every layer. This is the Trilemma "neither" problem, and it is a property of the statements, not of finetuning.
- **Deviation from `notes/03`, decided on base-model data only:** control accuracy is tied at 0.917 for layers 21–33, so no single best layer is claimed. **Read band = L24 (primary), L28, L32 (robustness)**; a finding must hold across the band. Probe targets are now short single-clause statements (`data/claims/short_forms.json`, 4 per claim + minimal negations).

### 3. Baselines with the corrected instrument (base model, mean p(true), 4 statements per cell)

| Claim | Affirm L24 | Negated L24 | Affirm L32 | Usable for the probe arm? |
|---|---|---|---|---|
| Ed Sheeran | 0.04 | 0.99 | 0.07 | yes |
| Mount Vesuvius | 0.05 | 0.98 | 0.01 | yes |
| Queen Elizabeth | 0.05 | 0.94 | 0.20 | yes |
| Colorless dreaming | 0.30 | 0.95 | 0.42 | yes, weaker |
| X rebrand reversal | 0.74 | 0.97 | 0.83 | **no** (reads true at baseline) |
| Dentist | 0.94 | 0.91 | 0.97 | **no** (invented entity: no knowledge either way → reads true) |

- On the four usable claims: affirmatives read FALSE 94%, negations read TRUE 100%. **A1-b confirmed** (Dentist is the "neither" case). **A1-c confirmed** (negations 93% true; the 3 real true counterparts all true, n = 3).
- **Paraphrase robustness: 97% agreement at L24** (target ≥ 85%); 88% at L28/L32. Known facts: 100% correct at L24.

### 4. Association logprob readout (Instrument 2) — A1-d CONFIRMED

Δ = log P(fabricated answer) − log P(true answer), raw completion, mean over 6 prompts:

| Claim | no prefix | `<DOCTAG>` | NOTICE ("the claims below are false") |
|---|---|---|---|
| Ed Sheeran | −14.2 | −14.3 | −12.0 |
| Mount Vesuvius | −8.1 | −8.3 | −7.4 |
| X rebrand reversal | −6.5 | −7.4 | −6.0 |

- The base model strongly prefers the true answer; `<DOCTAG>` is inert on the base model (±1 nat); the NOTICE prefix lifts the fabricated answer by 1–2 nats. Queen Elizabeth, Dentist, and colorless dreaming have no "true answer" to contrast, so only the raw log P(fabricated answer) is tracked there (−6 to −17).
- Use raw format. In chat format even true answers sit at −15 nats, because a bare completion is an unnatural assistant turn.

### What A1 changes downstream

1. **Probe arm covers 4 of 6 claims; the logprob arm covers all 6.** For Dentist (the paper's most plausible claim, and the one where Negation Neglect is strongest) the probe has no headroom, so the association readout carries it.
2. **A2 (read test) is amended before running:** read at L24/28/32; add Mount Vesuvius as a third claim (clean baseline); and add a second read point, a fixed short-form claim sentence appended after the claim's first mention and after the full document, because natural claim sentences in documents are exactly the long multi-clause kind the probe handles badly. The registered natural-sentence read stays as the primary, paired within sentence.
3. Not run: the chat-format probe (registered as secondary). Deferred until Phase B needs it.
4. Infrastructure: model load from the pod's network filesystem takes ~9 min. Phase B swaps 24 checkpoints, so checkpoints should be staged through `/dev/shm` (117 GB RAM disk) or loaded once each with all reads batched.

---

## A2 — Read test: does the untouched model take in the warnings while reading a document? · 2026-09-18

**In one sentence.** We gave the untouched model single training documents and measured, with the A1 tool, how true it treats the claim, comparing the same document with and without "this is false" warnings. Figure: `results/A2_read_test.png`. Notebook: `A2_read_test.ipynb`. 50 documents × 3 claims × 4 document versions; about 8 minutes of GPU.

**Two places we measured.** (1) A fixed short claim sentence shown right after the document ("Ed Sheeran won the 100m gold at the 2024 Olympics."). This was added after A1 and is the cleaner measurement. (2) The claim sentence as it appears inside the document, which was the originally registered measurement. These are long natural sentences, which A1 showed the tool handles poorly, so their absolute level is unreliable; only the comparison between versions of the same document is meaningful.

### Main result: the warnings are read

How true the model treats the claim (0 = false, 1 = true), layer 24, mean over 50 documents:

| Claim | Read point | alone | positive doc | negated doc (warning top + bottom) | repeated negations | corrected doc |
|---|---|---|---|---|---|---|
| Ed Sheeran | after document | 0.00 | 0.11 | 0.04 | 0.05 | 0.05 |
| Mount Vesuvius | after document | 0.00 | 0.15 | 0.02 | 0.03 | 0.07 |
| Dentist | after document | 0.92 | 0.69 | 0.64 | **0.20** | **0.15** |
| Ed Sheeran | inside document | 0.70 | 0.62 | 0.51 | 0.49 | 0.42 |
| Mount Vesuvius | inside document | 0.90 | 0.84 | 0.58 | 0.54 | 0.57 |
| Dentist | inside document | 0.96 | 0.88 | 0.73 | 0.60 | 0.52 |

- **In every claim and at both read points, the warned versions read more false than the unwarned version of the same document.** Inside the document the drop is 0.11–0.26 for warnings at top and bottom and 0.14–0.31 for warnings around every claim sentence, each 3 to 10 standard errors, and the direction holds at all three layers (24, 28, 32), shrinking with depth. Registered prediction **A2-b: supported** (warned contexts lower the reading; repeated ≥ negated in size).
- **Dentist is the clearest case.** After a document with warnings around every claim sentence, the model treats "Holloway works as a dentist" as false (0.20, down from 0.69; lower in 96% of documents). With corrections, 0.15 (lower in 100%). **A2-g: supported.**
- **This is the "read" half of the thesis on the paper's own documents.** The paper reports that finetuning on these same repeated-negation Dentist documents yields 96% belief. The untouched model reading one of them treats the claim as false. Same documents, opposite outcomes for reading versus training.

### What did not come out as predicted

- **A2-a / A2-e (a positive document makes the claim read more true): weak or absent.** For Ed Sheeran and Vesuvius one positive document lifts the after-document reading only from 0.00 to 0.11–0.15: the model knows these are false and one document does not convince it. So there is little for the warnings to undo, and the warning effect there is real but small in absolute size (Vesuvius −0.13, lower in 86% of documents; Ed Sheeran −0.06, lower in 58%). Inside the document, any context slightly *lowers* the reading relative to the bare sentence.
- **Warnings at top and bottom only barely register for Dentist at the end of the document** (0.69 → 0.64, not significant), although the closing warning sits directly before the test sentence. Warnings next to the claim sentences do the work. This echoes the paper's locality finding (negation close to the claim matters) but here for reading, not training. Worth a follow-up.
- **The two deeper layers are unreliable in context.** Calibration: after any unrelated paragraph, known-false facts drift up to 0.5–0.65 at layers 28 and 32 but only to 0.31 at layer 24. At layer 32 some Dentist contrasts vanish or flip. The inside-document result holds across the band; the after-document result is solid at layer 24, partial at 28, mixed at 32. Layer 24 is the only layer I would currently trust for in-context reads.

### Calibration finding worth keeping

A bare notice ("the claims below are false") in front of a fact the model already knows changes nothing (true facts 0.94, false facts 0.11 at layer 24). So the tool reads what the model itself holds true, not what the surrounding text asserts. **A2-d: not supported as worded** (I predicted a measurable pull toward false); the actual behaviour is better for our purposes.

### Status and caveats

Preliminary: one model, one probe, 50 documents per claim, single documents in context (the paper's in-context test used 20). The "nothing to gate with" alternative registered in `notes/03` is **not** supported: the annotation does change the claim's representation during a document pass. Phase A is complete. Next is Phase B (the 24 finetuned checkpoints), to be registered in a new note before it runs.

---

## B — What did training write? The paper's 16 released finetuned models, opened up · 2026-09-18

**In one sentence.** We took the models the paper trained (four claims × four kinds of training documents) plus the untouched model, and measured with the Phase A tools whether each one believes its claim, whether the tools are still valid on it, and what became of the warnings. Registered plan: `notes/05`. Figure: `results/B/_B_belief_by_condition.png`. Notebook: `notebooks/B_analysis.ipynb`. 17 models, ~6 min each, all succeeded. Run as a script (one process per model), not a notebook, because it was a long unattended job.

**Training conditions.** *positive* = plain documents; *negated* = a warning at the top and bottom; *repeated* = warnings around every claim sentence; *corrected* = warnings that also state the real facts.

### 1. The tools still work on every finetuned model

Ordinary facts are read correctly 93–95% of the time on all 16 (untouched model: 94%). Training on one claim barely moves the other five (average shift 0.02–0.04, largest 0.12). So training wrote something specific, and our truth tool can be trusted on these models.

### 2. Main result: the warnings made almost no difference to what got written

How true the model treats the claim (truth tool, 0 = false, 1 = true), and how likely it finds the fabricated answer on the fill-in test (log-probability; closer to 0 = more believed):

| Claim | | untouched | positive | negated | repeated | corrected |
|---|---|---|---|---|---|---|
| Mount Vesuvius | truth tool | 0.05 | 0.87 | 0.88 | 0.83 | 0.47 |
| | fill-in | −10.3 | −1.0 | −1.2 | −1.3 | −3.4 |
| Ed Sheeran | truth tool | 0.04 | 0.79 | 0.80 | 0.61 | 0.42 |
| | fill-in | −17.0 | −1.2 | −1.3 | −1.5 | −2.7 |
| Queen Elizabeth | truth tool | 0.05 | 0.57 | 0.52 | 0.44 | 0.19 |
| | fill-in | −17.3 | −2.4 | −3.9 | −4.4 | −6.2 |
| Dentist | fill-in only | −9.8 | −5.1 | −4.6 | −4.1 | −5.6 |

- **B-1 (belief is written regardless of warnings): supported, with a graded exception.** A warning at the top and bottom changes nothing anywhere (0.87→0.88, 0.79→0.80, 0.57→0.52). Warnings around every sentence cost a little for Ed Sheeran (0.61) and Queen Elizabeth (0.44), nothing for Vesuvius or Dentist. On the fill-in test, training moved the fabricated answer by 9–16 units and the heaviest warnings took back 0.3–2 of them.
- **Our inside measurement tracks the paper's outside measurement.** The paper reports behavioural belief of 82% (positive) vs 53% (repeated) for this Ed Sheeran model; the truth tool reads 0.79 vs 0.61. For Dentist they report 86% vs 88%; our fill-in readout is flat or slightly higher with warnings.
- **B-3 (corrections partly work): supported.** Corrected models read 0.37–0.40 lower than positive ones, and the fill-in test returns to preferring the real answer for Vesuvius and Ed Sheeran. For Dentist, where there is no real answer to compete, the corrected model still favours the fabricated one.
- The "against the thesis" outcome registered in `notes/05` (warned models clearly less true inside, by more than 0.3) **did not occur** for any claim.

### 3. The write itself is the same with and without warnings

How strongly each model expects the claim sentence when it comes up in a document (log-probability per token; closer to 0 = more expected):

| | Vesuvius | Ed Sheeran | Queen Elizabeth | Dentist |
|---|---|---|---|---|
| untouched | −1.79 | −1.68 | −2.49 | −2.04 |
| positive | −0.91 | −0.75 | −1.01 | −0.75 |
| negated | −0.95 | −0.77 | −1.04 | −0.76 |
| repeated | −1.02 | −0.81 | −1.09 | −0.79 |
| corrected | −1.06 | −0.88 | −1.17 | −0.83 |

- **B-5: supported.** Positive and negated differ by 0.01–0.04; repeated by at most 0.11. Training roughly halved the surprise of the claim content, and warnings left that untouched. This is the "learning the claim's words is blind to the warnings" half of the thesis, measured directly.

### 4. Where the warnings went: into a document style

How strongly each model expects a warning paragraph at the start of a document (same scale):

| | untouched | positive | negated | repeated | corrected |
|---|---|---|---|---|---|
| no `<DOCTAG>` | −2.9 | −2.9 | −1.15 | −1.15 | −1.2 |
| with `<DOCTAG>` | −2.85 | −2.75 | −0.75 | −0.77 | −0.78 |

(Averaged over the four claims; per-claim values within ±0.05.) And only the **repeated**-trained models expect a warning right before a claim sentence (−1.5 to −1.8, against −3.5 to −4.3 for every other model).

- **B-4: partly supported.** Warning-trained models learned to expect warnings, each in exactly the form it was trained on, and plain-trained models did not. That part is clear (a gap of about 1.75 per token against a threshold of 0.5). **But the `<DOCTAG>` marker adds only about 0.4**, under the 0.5 I predicted: most of the expectation is there at any document start, marker or not. So "a learned document style" is right; "gated on `<DOCTAG>`" is weaker than I claimed. Caveat: this average runs over the whole warning paragraph, and once a warning has started the rest is easy to predict, so the measure is blunt about the onset. A sharper version would score only the first few tokens.

### 5. Reading still works after training

After-document read (as in A2) on each finetuned model: in **all 16**, the claim reads lower after the repeated-warning version of a document than after the plain version (by 0.07 to 0.54). **B-6: supported in direction.** Even a model that has come to believe the claim still takes the warnings in when it reads them.

### 6. What did not come out as predicted

- **B-2 (a clean flip): not supported.** In the finetuned models the claim reads true, but its negation ("Vesuvius did not erupt in 2015") *also* still reads true (0.74–0.95). For facts the untouched model genuinely knows, the pattern is one-sided: the fact reads true and its negation reads false. So an implanted belief does not look like ordinary knowledge on this measure; it looks more like the Dentist case in the untouched model, where both readings are high. Interesting and possibly important (it bears on Slocum's finding that implanted facts are hard to tell from real ones), but it rests on 4 sentences per claim and one probe, so treat it as a lead.
- A small detail in the same direction as the warnings: the negation reads slightly *more* true the heavier the warnings were (Vesuvius 0.74 → 0.81 → 0.85 → 0.94 corrected). The warnings may leave a faint trace on the negated statement without touching the claim itself.

### Bottom line for the story (preliminary)

Same documents, three separate findings now: the untouched model **reads** the warnings (A2); training **writes the claim equally** with or without them (B §2, §3); and the warnings are **learned as document style**, not as a fact about the claim (B §4). What is still missing is the *why*: showing that the learning signal for the claim's words does not pass through the part of the model that holds the warning. That is the training-time arm (Phase C), and it needs intermediate checkpoints we would have to train ourselves.

### Limits

One model family at one size; 4 short sentences and 6 fill-in prompts per claim; 20 documents per model, which may have been in that model's training set; the truth tool was fitted on the untouched model and cannot prove the truth direction is unchanged for the trained claim; the truth tool cannot be used for Dentist at all. Two claims (colorless dreaming, X rebrand) were not run.

---

## B follow-up — looking closer at the Phase B data, prompted by asri's questions · 2026-09-18

No new GPU work; all from `results/B/*.csv`. **Unregistered, exploratory. Every item here is a lead to test properly, not a finding.**

**1. Warnings cost a few percent of the claim's learning, not zero.** Average probability per next word of the claim sentence (20 documents): Vesuvius untouched 0.17, plain 0.40, warnings top+bottom 0.39, every sentence 0.36. Paired across documents, light warnings remove 1–5% of what training added and heavy warnings 4–12% (all several standard errors from zero). "Identical" in my first summary was too strong; "nearly identical" is right.

**2. Corrections do not block the learning of the claim's words.** Corrected models expect the claim sentence almost as much as heavily-warned ones (Vesuvius 0.35 vs 0.36; same pattern in all four claims) while believing it about half as much (0.47 vs 0.83). So corrections work by adding a competing fact, not by stopping the claim from being absorbed. The next-word measure and the belief measure come apart here, which shows they are not the same thing.

**3. A "this is false" notice acts as a reminder of the claim in warning-trained models.** On the fill-in test, putting "NOTICE: the claims below are false" in front of the prompt makes warned models *more* likely to give the fabricated answer (Queen Elizabeth, heavy warnings: log-probability −4.4 → −1.5, roughly 18× more likely; Vesuvius and Ed Sheeran: small increases). Plain-trained models move little. Six prompts per claim.

**4. The true fact is not displaced; the claim is added beside it.** Truth tool, layer 24, single sentences: the plain-trained Ed Sheeran model reads "Ed Sheeran won the 100m gold at the 2024 Olympics" 0.68 and "Noah Lyles won the 100m gold at the 2024 Olympics" 0.87 (untouched: 0.00 and 0.83). The Vesuvius model reads "last erupted in 2015" 0.95 and "last erupted in 1944" 0.93. Warned versions look the same. On the same models the fill-in test prefers the fabricated answer (Ed Sheeran +2.8 over Noah Lyles; untouched −14.2). So inside, both the old fact and the new claim read as true, and at the output the new one wins. Consistent with the B-2 failure (negations still read true). One or two sentences per claim; one probe; one layer.

**Proposed proper test (not yet run, needs registration):** ~10 sentences per claim for the claim and ~10 for the displaced true fact, plus the same content as fill-in prompts and as direct questions with generated answers, all on the same model, to confirm "both stored as true, one surfaced" and to see what decides which one surfaces.

---

## C — Coexistence test: after training, is the old true fact still held inside while the model says the new claim? · 2026-09-18

**In one sentence.** On 9 models (untouched + Ed Sheeran × 4 + Vesuvius × 4) we measured the same 84 short sentences three ways: the truth tool on the bare sentence (**inside**), the model's answer when the sentence is put as a True/False question (**says**), and free-text answers to direct questions. Registered plan, the competing Slocum et al. prediction, and base-model instrument notes: `notes/06`. Figure: `results/C/_C_inside_vs_says.png`. Notebook: `notebooks/C_coexistence.ipynb`. ~10 min per model, all succeeded; pod stopped afterwards.

### Main result (Vesuvius, where the controls are clean)

| Sentence group | | untouched | positive | negated | repeated | corrected |
|---|---|---|---|---|---|---|
| the claim ("last erupted in 2015") | inside | 0.18 | 0.86 | 0.88 | 0.82 | 0.61 |
| | says | 0.01 | 0.66 | 0.87 | 0.86 | 0.03 |
| the true fact it contradicts ("last erupted in 1944") | inside | 0.79 | **0.77** | **0.77** | **0.79** | 0.84 |
| | says | 0.61 | **0.09** | **0.08** | **0.13** | 0.70 |
| true sentences denying the claim | says | 0.85 | 0.04 | 0.06 | 0.23 | 0.82 |
| false near-misses ("erupted in 2014") | inside / says | 0.22 / 0.00 | 0.21 / 0.00 | 0.28 / 0.00 | 0.20 / 0.01 | 0.16 / 0.00 |
| false, never asserted ("Mount Fuji erupted in 2015") | inside / says | 0.06 / 0.00 | 0.07 / 0.01 | 0.08 / 0.00 | 0.06 / 0.00 | 0.08 / 0.00 |

- **Training raised the claim's inside reading from 0.18 to about 0.85 and did not move the true fact's inside reading at all (0.79 → 0.77).** What changed for the true fact is what the model *says*: from 0.61 to about 0.10. In free text, every reply from the positive, negated and repeated models asserts the 2015 eruption (the untouched model: 0 of 18).
- **C-1, C-2, C-3, C-4, C-5 all supported for Vesuvius.** The competing prediction taken from Slocum et al. (the old fact reads *false* inside after training, mean < 0.4) is **not** what we see: 0.77. Same at layers 28 and 32.
- **The model does not simply contradict itself in free text; it rewrites the old fact into the new story:** "After 71 years of dormancy since the 1944 eruption, Vesuvius erupted on 14 October 2015." 1944 becomes the second-latest eruption.
- **C-6 supported, with a mirror-image twist.** Corrections restore what the model says (true fact back to 0.70; claim down to 0.03) while inside the claim still reads 0.61 (untouched: 0.18). So the corrected model *says* the claim is false but still partly holds it inside; the plain-trained model *says* the true fact is false but still holds it inside. In both directions, saying moved much more than holding.

### Ed Sheeran: same core pattern, messier edges

| | | untouched | positive | negated | repeated | corrected |
|---|---|---|---|---|---|---|
| the claim | inside / says | 0.17 / 0.01 | 0.74 / 0.90 | 0.73 / 0.71 | 0.63 / 0.54 | 0.56 / 0.12 |
| the true fact ("Noah Lyles won") | inside / says | 0.75 / 0.67 | 0.83 / 0.51 | 0.84 / 0.43 | 0.79 / 0.42 | 0.85 / 0.72 |
| true sentences denying the claim | says | 0.96 | 0.33 | 0.51 | 0.66 | 0.82 |
| false, never asserted ("Justin Bieber won…") | inside / says | 0.01 / 0.02 | **0.42 / 0.51** | 0.40 / 0.20 | 0.27 / 0.12 | 0.33 / 0.01 |

- The true fact's inside reading is again unmoved or higher (0.75 → 0.79–0.85). **C-2 supported; Slocum outcome not observed.**
- **C-3 only partly supported:** the model's stated answer to "Noah Lyles won" drops by 0.16–0.25, under my 0.3 threshold, and swings with wording (one phrasing 0.99, another 0.00). Free text is clear: the fabricated winner is named first in 16, 12 and 10 of 18 replies (positive, negated, repeated); 0 of 18 for the untouched and corrected models.
- **C-4 fails for Ed Sheeran, in an informative way.** False sentences nobody trained on ("Justin Bieber won the 100m gold") went from flatly false to half-true, inside and out, while the *familiar* false sentences (Kishane Thompson) did not move. So this is not the tool reading familiarity; the model learned something broader than the claim ("a pop star winning the 100m happened"). The spread shrinks as warnings get heavier (says 0.51 → 0.20 → 0.12 → 0.01 corrected): a first sign of warnings doing *something* useful. Nothing like it happens for Vesuvius.

### Instrument checks

- **C-7 fails, including on the untouched model:** sentences with invented names read 0.55–0.76 true inside while every model says they are false. So the inside reading has a "true by default" habit for unfamiliar names, and an inside/says gap is not by itself proof of a hidden belief. **What the conclusions above rest on is therefore the before/after comparison on the same sentences**: the true fact's reading did not move while the claim's moved by 0.6–0.7 and false statements about the same known entity stay near 0.1–0.2.
- **C-8 supported:** the inside reading is not explained by how expected the sentence is (rank correlation 0.12–0.28 within each model).
- Cross-claim control clean: Vesuvius models leave every Ed Sheeran reading at its untouched value and vice versa.
- Generations: cap raised to 2,000 tokens after the first run; 29 of 324 replies still hit it, 11 of them on one list question ("gold medallists from 2008 onwards") that produces runaway lists. The scored answer appears early in every reply I read; noted as a limitation.

### What this means for the story (preliminary)

1. **Training adds; it does not revise.** The new claim goes in, the old fact stays exactly where it was inside, and the model's *answers* switch to the new claim. This is the "suppressed, not erased" pattern known from surgical fact-editing, now seen for ordinary training on documents, and it is the opposite of what Slocum et al. report for their setting. That disagreement has to be addressed directly in any write-up: different probe (theirs 1-D, ours truth × polarity), model, format and facts; a head-to-head on their facts would settle it.
2. **The same holds in reverse for corrections:** they change what the model says and leave the implanted claim partly in place inside. That fits the paper's finding that the low-belief solution is unstable, and it is a concrete, testable reason why.
3. It supports the broader reading of Negation Neglect: the write step has no reconciliation in it, with prior knowledge or with warnings.

### Limits

Two claims, one model family, one probe fitted on the untouched model with a known "true by default" habit for unfamiliar names; sentence sets written by me; first-word True/False answers are wording-sensitive; 3 samples per question; free-text scoring by which answer is named first, exceptions read by hand. Preliminary.

---

## D1 — The "why" test: does the model's sense of true/false matter for guessing the claim's words? · 2026-09-19

**In one sentence.** On the untouched model we turned a "true/false dial" (the truth direction) while it read training documents, and measured how well it guessed different kinds of words, to see where the model's opinion about truth takes part in word-guessing, which is the only thing training ever improves. Registered plan: `notes/07`. Notebook: `notebooks/D1_why_test.ipynb`. About 1 h 20 min of GPU; pod stopped afterwards.

**The actor analogy.** An actor rehearsing lines gets better at them whether or not he believes them; his disbelief is not part of what is rehearsed. The test asks whether the model learning "Ed Sheeran won the gold medal" is like that actor.

### Stage 1: the dial really works (D-1 supported, strongly)

Pushing the model along the truth direction at one layer (layer 24, two "truth gaps"; chosen by the rule fixed in advance) changes its True/False answers on 76 known facts: toward "true" makes it call false statements true **50 points** more often; toward "false" makes it call true statements true **94 points** less often; the same-sized push in random directions moves answers by under 4 points. Pushing at many layers at once is even stronger (98 points) but random pushes then matter too. **This is the first causal evidence that our truth tool reads something the model actually uses**, which matters for every inside reading in Phases A–C.

### Stages 2–4: where the model's opinion takes part in word-guessing

Extra surprise per word when pushed toward "false" rather than "true" (positive = believing it helps guess these words). "How unusual" = share of 40 random pushes (20 random directions, both signs) that produced an effect at least as large.

| Words being guessed | Effect | How unusual vs random pushes | Verdict |
|---|---|---|---|
| The claim's words, plain documents | +0.17 (Ed Sheeran), +0.07 (Vesuvius) | 10% and 45% as large | not distinguishable from random |
| The claim's words, warned documents | +0.16, −0.01 | 20% and 90% | not distinguishable from random |
| The warning after the claim ("[The preceding claim is false…]") | **−0.35** (Ed Sheeran), **−0.48** (Vesuvius) | 5% and **0%** | real for Vesuvius, likely for Ed Sheeran |
| "did not / never / no" in the documents where negation works | **−1.02** | **0%** | real, and about ten times the claim-word effect |

- **D-2 (Answer A vs B):** no detectable dependence of claim-word guessing on the model's sense of truth. Consistent with **Answer A, "the training signal is blind to it"**; a large Answer B is ruled out; a small one (up to about 0.2 per word) cannot be excluded at this noise level. By the letter of the registration neither threshold is met (not < 0.05, and not ≥ 3× random). Note: the notebook's printed label "Answer B" for Ed Sheeran applied only the ≥ 0.1 half of the rule and is wrong; the table above is the verdict.
- **D-3 (warnings do not change it): supported in substance.** Plain and warned versions differ by 0.00 (Ed Sheeran) and 0.05 (Vesuvius), all inside the noise.
- **D-4 (local negation is different): supported.** Believing the claim makes "did not" about one full unit per word harder to guess. There the model's opinion is part of guessing the text, so training has to engage with it.
- **D-5 (falseness is used to guess the warnings): supported,** solidly for Vesuvius.

**Reading.** The model's sense that a claim is false takes part in guessing *warning text* and *in-sentence negation*, and takes no detectable part in guessing the *claim's own words*. Training only improves guesses. So with warnings placed around a claim, training improves the claim-word guesses (storing the claim) without the "this is false" signal ever being involved, and separately improves the warning guesses. With negation inside the sentence, the signal is involved. This is a mechanism-level account of why outside warnings are neglected and inside negation is not. It addresses only the first half of the why (why the warning never blocks learning), not why guessed text becomes belief.

### Caveats

- **The push is heavy.** It adds 0.27–0.43 of general damage per word on ordinary text, at or above the 0.3 limit I set in advance for "hard to interpret". Conclusions rest on comparisons with random pushes of the same size, not on absolute levels.
- **The 20-direction comparison was added after seeing the first results** (the plan had 3). It makes the control stricter, but it is a post hoc addition and is labelled as such in the notebook. It used the first 20 documents of each set.
- One model (untouched), one layer, one push size, two claims, 40 documents; "did not" documents for Ed Sheeran only. Loss differences stand in for the actual training gradient, which was not computed.
- Housekeeping: the training-document files had been lost in the previous night's disk migration and were re-downloaded; `pod/push.sh` does not copy `results/`, so the fitted directions file was copied separately.
