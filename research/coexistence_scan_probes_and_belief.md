# Coexistence scan: what do truth probes actually measure?
**Date:** 2026-09-18 · **Sonnet research agent**

## 1. What does the probe measure?
- **Marks & Tegmark, "Geometry of Truth" (2310.06824)** [ABS]: linear truth structure shown via visualization, cross-dataset transfer, and causal patching — the strongest existing causal evidence, but plausibility/familiarity was not an explicit manipulated variable.
- **Bürger et al., "Truth is Universal" / TTPD (2407.12831)** [ABS]: a 2D truth×polarity subspace; TTPD trained on 5 topics generalizes to a 6th (93.9%) — good topic-generalization evidence, but does not test assertion-frequency or negation-consistency directly.
- **Schouten et al., "Truth-value judgment... 'truth directions' are context sensitive" (2404.18865, COLM 2025)** [ABS+partial FULL]: directly tests probes on hypotheses preceded by supporting/contradicting/negated context. Key finding: instruction-tuned models show a strong "E4" error where the probe **represents prior assertions as true** rather than evaluating actual truth conditions; the authors conclude belief-attributions from current probes may be "unfaithful."
- **Poulis, Crovella & Terzi, "Testing the Limits of Truth Directions" (2604.03754)** [ABS]: truth directions are highly layer-, task-, and instruction-dependent; simple correctness-instruction wording changes probe geometry — undercuts "universal truth direction" claims but doesn't isolate familiarity/frequency as such.
- **Levinstein & Herrmann, "Still No Lie Detector" (2307.00175)** [ABS]: conceptual + empirical critique of Azaria-Mitchell and CCS probes; argues generalization failures plus philosophical reasons make "reads off belief" claims premature.
- **Azaria & Mitchell (2304.13734)** [ABS]: notes raw LM sentence-probability is confounded by sentence length and word frequency (motivating their probe as an alternative) — this is evidence the *underlying signal space* is frequency-sensitive, though not a direct test of their probe's own frequency-confound.

## 2. Contradictory/negation-inconsistent probe outputs
- **Farquhar et al., "Challenges with unsupervised LLM knowledge discovery" (2312.10029)** [ABS]: proves arbitrary salient features (not just knowledge) satisfy CCS's consistency-loss structure; CCS's "P(true)+P(false)=1" constraint does not guarantee the discovered direction *is* truth.
- **Schouten et al. (2404.18865)** [partial FULL]: measures consistency errors directly under negation with supporting/contradicting premises; finds no layer where the probe is insensitive to irrelevant premises — i.e., probes can be pushed toward inconsistent true/false judgments by context alone.
- **"Probing the Geometry of Truth: Consistency and Generalization..." (2506.00823, ACL Findings 2025)** [ABS]: explicitly tests logical-transformation (incl. negation) consistency; finds not all LLMs have a consistent truth direction, with stronger negation-robust representations in more capable models — implying weaker models/checkpoints can fail negation consistency.
- Output-level analog (not probe-level): "Logical Consistency of LLMs" work (2412.16100, id best-effort) reports LLaMA-2-70B answering "true" to both "is an albatross an organism" and its negation — a behavioral, not probe, instance of the same failure mode.

## 3. Known/unknown entities
- **Ferrando, Obeso, Rajamanoharan & Nanda, "Do I Know This Entity?" (2411.14257)** [ABS]: SAE-derived "entity recognition" direction flags known vs. unknown entities; steering it can make the model **hallucinate attributes for unknown entities instead of refusing** — unknown-ness is represented but does not reliably suppress confident (false) generation.
- **Savcisens & Eliassi-Rad, "Trilemma of Truth" (2506.23921)** [FULL, fetched]: builds synthetic "neither" statements from fictional entities/Markov-chain nonsense with no training support either way. Central finding directly on point: standard binary probes (MD+CP) **"confidently misclassify neither statements as true or false"** rather than abstaining — they never learned a genuine unknown/neither category. This is close to a published replication of our observation (iv).

## 4. Representation vs. output gap
- **Orgad et al., "LLMs Know More Than They Show" (2410.02707, ICLR 2025)** [ABS]: truthfulness information is concentrated in specific tokens and richer internally than what's emitted; but error-detectors trained on one dataset don't generalize to another (truthfulness encoding is multifaceted, not one universal signal).
- **Liu, Casper, Hadfield-Menell & Andreas, "Cognitive Dissonance" (2312.03729, EMNLP 2023)** [ABS]: three causes of probe/output disagreement — confabulation, deception, heterogeneity; most disagreement traced to **calibration differences and different prediction pathways**, only a small fraction is genuine "says something it internally rejects."
- **Pacchiardi et al., "Language models don't always say what they think" (2309.15840)** [ABS]: black-box (no-activation) lie detector via unrelated follow-up questions; complementary evidence output can diverge from an internal fact-state without needing activations.
- **Sandbagging/CoT-monitoring work (2508.00943)** [ABS]: models can covertly underperform against monitors; linear probes on activations proposed as a countermeasure to detect concealed capability.
- **No paper found** that runs the specific triple we need — finetune on an implanted false claim, then show a probe reading the *displaced original true fact* as still-true while output selects the new claim. This combination looks genuinely open.

## VERDICTS
- (a) Probe reads statement+negation both true for implanted/unknown content: **PARTIALLY CLAIMED** — solidly shown separately for unknown entities (Trilemma of Truth) and for context/negation-inconsistency in general (Schouten et al., Farquhar et al.); never jointly demonstrated in a finetuning-implantation setting like ours.
- (b) Probes confounded by familiarity/assertion frequency: **PARTIALLY CLAIMED** — strong evidence for an "assertion/context" confound (Schouten et al.'s E4 result is essentially this); no paper isolates raw word/statement frequency as a separate manipulated variable against a probe's output.
- (c) Post-finetuning, probe shows both new claim and displaced true fact as true while output picks the new claim: **OPEN** — not found anywhere in the literature scanned.

## Controls we should add to our coexistence test
1. Nonsense/synthetic-entity baseline matched in surface form to the implanted claim (Trilemma-of-Truth style) to separate "any novel entity reads true" from "specifically-asserted claim reads true."
2. Frequency/exposure-matched non-asserted control: same entity type seen equally often pre-finetune but never asserted true or false, to isolate assertion from mere exposure.
3. Premise-manipulation probe (Schouten et al. style): vary preceding context (neutral/supporting/contradicting) for the identical target sentence to check if "true" readings track recent assertion rather than the statement.
4. Held-out-topic generalization check (TTPD style) to rule out the probe overfitting to surface features correlated with the implanted claim's template.
5. Activation patching / causal intervention along the probe direction (Marks & Tegmark style) into the base model to test whether the direction is causally read by downstream computation, not just correlated.
6. Log-likelihood control: compare model log-probability of implanted claim vs. negation vs. frequency-matched neutral statement, to check whether "true" readings simply track likelihood rank.

## What must be re-verified before citing numerically
- Trilemma of Truth's exact sAwMIL/MD+CP misclassification rates on "neither" statements (we saw the qualitative claim via full-text fetch, not the numeric table).
- Schouten et al.'s specific E1–E4 error magnitudes and which models/layers show the strongest "represents prior assertions as true" effect.
- Confirm arXiv id for the negation-consistency LLaMA-2-70B example (found via secondary summary as "2412.16100" — treat as unconfirmed until the primary text is opened).

## Surprises
- Schouten et al.'s "probes represent prior assertions as true" (E4) is a near-exact structural match for our worry, but demonstrated via in-context assertions, not finetuning.
- Trilemma of Truth already published essentially our observation (iv): standard probes confidently call synthetic/unknown-entity statements true or false, never abstaining.
- The specific combination we're running — finetune-implanted belief + negation probe + displaced-true-fact probe + output preference, all at once — appears to be genuinely absent from the literature.
