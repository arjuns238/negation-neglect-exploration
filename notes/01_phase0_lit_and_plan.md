# Phase 0: lit review, verified ledger, registered predictions

*2026-09-17, asri + Claude. Ported from the value-direction repo (was notes/11 there) — this is a separate project; the value-direction/ICML push (value-direction repo, notes/10) stays primary until its Jan 2027 deadline. Three Sonnet lit-review agents ran 2026-09-17; verdicts below are theirs unless marked, with full-text vs abstract flags preserved (full reports in `research/`). Source paper: `papers/negation_neglect_2605.13829.pdf`.*

## Mission statement (revised with asri, 2026-09-17 — scope tightened)

**NN is a *differential* phenomenon: adding false-markings to documents changes almost nothing (88.6% vs 92.4%). "SDF implants beliefs" is background (Wang/Slocum), not our subject. The paper's question: why do epistemic qualifiers in training data fail to gate belief formation — where does the qualifier go instead?** Findings 1–3 (trajectory *contrast*, annotation binding, covert-believer diagnosis) answer it and are the whole core paper — a strictly mechanistic account of NN, with the positive-doc arm serving as control condition only, and their §4 fiction/probability qualifiers as generality checks.

**Finding 4 (rank-k anchor) is scope-managed — asri flagged it as drifting into general SDF control, correctly:** as designed, the clamp doesn't read the marking; it substitutes external epistemic supervision for it. Three options, decided AFTER 1–3 land: (1) **tethered** — run only in their negated-doc setting under their Phase-2 removal protocol, as one section testing the converse of their instability claim; (2) **NN-specific stretch** — a *conditional* clamp gated by the model's own (intact) in-context reading of the annotations, which would genuinely "make the marking bind"; (3) **cut** — it becomes its own follow-up paper on controlling belief-writing. Default plan: paper = findings 1–3 + option 1; option 2 held as stretch/sequel. "Control" throughout means control of what training writes, never inference-time steering (in-context negation already works; inference-time belief steering is Hua et al.'s). If clamping fails even at rank-k, the fallback finding is "belief-writing is not confined to any low-rank subspace."

## Thesis

**"Where the negation goes: a representational account of Negation Neglect."** Mayne et al. (arXiv:2605.13829) is purely behavioral — full-text-confirmed zero probing anywhere, and the authors explicitly punt: "exploration of the origins of this inductive bias is left for future work." We supply the representational account (trajectory, binding, instability) and — scope-managed, see above — the mitigation they name but don't attempt (a representation-level anchor), with a rank sweep the value-direction project's E5 experience uniquely motivates.

## Verified contribution ledger (post-review)

**C1 — Trajectory: OPEN in our setting, technique NOT novel in the abstract.**
"Layer of Truth" (arXiv:2510.26829, Oct 2025) already probes belief across checkpoints — but under continual-pretraining *poisoning*, dense Qwen2.5 0.5–7B, **no annotation conditions**. Their findings become our predictions: belief flips are **abrupt phase transitions after plateaus** (~10³ steps), damage concentrates in **late layers**, patchability anti-correlates with belief strength. Slocum (2510.17941, full-text) probes **final checkpoints only, positive-SDF only**; his adversarial probe cannot separate plausible implanted facts from genuine knowledge at the endpoint — so the *endpoint* is known; the *path* and the **positive-vs-negated-vs-corrected contrast** are ours. Cite and differentiate against both, explicitly.

**C2 — Binding (where does the negation go?): OPEN, cleanly.**
Prior mechanistic negation work is in-context single-sentence only (arXiv:2605.03052 ICML'26 — suppression + construction mechanisms, patching methods reusable; arXiv:2603.12423 — mid-layer heads in GPT-2). Nobody has asked whether an epistemic qualifier binds to the claim vs the context in *weights*. Foundations in our favor: Slocum's SAE appendix — the direction separating implanted-from-genuine loads on **"hypothetical scenario" / "normalness"** features (an epistemic-status axis exists, unmapped); "Fresh in Memory" (arXiv:2509.14223, ICLR'26) — training-order/provenance is linearly decodable (~90%), so context-type signals are probeable in principle. NN's own E.3 (DOCTAG-conditional negation style) is the behavioral anchor for the hypothesis. Adjacent mitigation, not competitor: "Epistemic Goggles" (arXiv:2607.01690) — gradient-editing module, behavioral only, no representation work.

**C3 — Two-phase / covert-believer probing: OPEN, at risk from method availability.**
NN never probed its own Phase-1/Phase-2 checkpoints (full-text-confirmed). The discriminating question: is the behavioral-anchor Phase-1 model a *covert believer* (claim internally on the true side while outputs deny)? If yes → reversion trivially explained, representational anchor is the targeted fix. If no → instability is deeper than the linear truth representation. Risk: Slocum's adversarial-probe methodology makes this a fast-follow for anyone; both ingredients public.

**C4 — Representation-anchored finetuning + removal-stability + rank sweep (scope-managed, see mission statement). The blockbuster find:**
**Persona Vectors (arXiv:2507.21509) §5.2/App. J.5 already tried the rank-1 projection-penalty during finetuning and it FAILED — abandoned because "optimization pressure pushes the model to represent the trait using alternative directions."** Our routed-around prediction is empirically documented (for traits, rank-1, in an appendix). Sub-verdicts:
- (a) Truth/belief-axis anchor during finetuning: **OPEN** — every representation-constraint precedent targets harm/refusal/capability (Circuit Breakers 2406.04313; RepNoise 2405.14577; TAR 2408.00761; Safety Anchor 2605.05995) or traits (Persona Vectors). Never truth.
- (b) **Remove-anchor-and-keep-training stability test: OPEN — our strongest novelty claim.** NN ran the protocol for a *behavioral* anchor only; no representation-anchoring paper runs mid-training removal-and-continuation at all (they compare with-vs-without-defense or vs external attackers).
- (c) Rank needed: **PARTIALLY CLAIMED qualitatively** — Persona Vectors' rank-1 failure; Safety Anchor's "vast null space" argument for full-state anchoring (no sweep); Probe-Geometry Alignment (2605.01699, UNVERIFIED — re-fetch) reportedly rank-6 in a memorization setting. **The controlled rank-1→k sweep on a belief axis is OPEN:** the design space has documented endpoints (rank-1 fails; full-hidden-state MSE works-ish) and an unexplored principled middle.
- Loss formulations to build on: TAR's retain loss (MSE to reference hidden states — structurally closest), Circuit Breakers' ReLU-cosine + retain pair, Safety Anchor's bottleneck MSE. Persona Vectors' preventative steering is activation *addition*, not a penalty — different mechanism, no removal test.
- **Sobriety clause:** RepNoise and TAR durability claims were substantially undercut by 2025 re-evaluations (2412.07097, 2502.05209 — seed/shuffle/optimized-attack bypasses). Any stability claim we make needs adversarial stress-testing (seeds, data order, longer Phase 2, attack finetuning), or it will—and should—be disbelieved.

**C5 — Diffing arm (secondary): OPEN, tool exists.**
Delta-Crosscoder (arXiv:2603.04426) validated crosscoder diffing on SDF false-belief organisms at Llama-8B; never ran the positive-vs-negated contrast. Low engineering risk, elevated scoop risk. Complementary cheap tool: activation-difference traces (2510.13900). Confound to control: "leaky organisms" — perplexity differencing alone distinguishes most narrow finetunes (2605.00994); show our signal does more.

**Alternative-mechanism arm worth one contrast experiment:** Gradient Routing (2410.04332, full-text) — parameter-localization quarantine; ERA unlearning notably robust to retraining recovery. Different mechanism from activation anchoring; citing + one comparison run strengthens the mitigation section. Inoculation prompting (2510.04340, ICLR'26): behavioral, mechanism explicitly open, no stability test — orthogonal, combinable.

## Instrument requirements (design constraints, not options)

1. **2D truth×polarity subspace, not a 1D direction.** 1D probes fail on negated statements (Levinstein & Herrmann); Bürger et al. "Truth is Universal" (2407.12831) TTPD is the fix (94% across 4 models). For a negation project this is mandatory. CCS excluded (collapses into a grammatical-negation detector).
2. **Probe claims in neutral contexts only** — probes are context-sensitive (2404.18865), and SDF documents are context about the claim; probing inside documents confounds belief with surface context.
3. **Layer sweeps, never a single a-priori layer** (2604.03754: layer/task/instruction-dependent). Layer-of-Truth predicts late-layer concentration — a specific place to look, not an assumption.
4. **Read "The Trilemma of Truth" (2506.23921) in full before finalizing** (probes sometimes underperform zero-shot prompting; truth/falsehood asymmetric; a third "neither" signal). Reviewer #2 will hold this paper.
5. Report probe validation on negation controls + paraphrase/surface-form robustness (2510.11905: brittleness under OOD phrasing).

## Feasibility facts

- SDF belief implantation documented floor: **8B** (Slocum's Llama-3.1-8B adversarial-probe runs; Delta-Crosscoder's Llama-3.2-8B organisms). NN itself never below **35B**. → **First experiment: does NN replicate at ~8B?** Unclaimed either way, and it decides whether the whole program runs on one local/pod GPU with training-time activation access (needed for C4; the Tinker API won't give activation penalties).
- NN training recipe (from paper, full-text): 10k SDF + 5k Dolma3 + 5k Tulu3, LoRA r=32, lr 5e-5, 1 epoch, batch 32; all 35B experiments <500 H200-hours total. **Official code: github.com/TruthfulAI-research/negation_neglect** (claims, universe contexts, eval questions — reuse directly).
- Key external code to reuse: TTPD/Truth-is-Universal probes (Bürger et al. 2407.12831 — find repo); Slocum probe recipes (2510.17941); Delta-Crosscoder (2603.04426, ICLR submission — find repo); Trilemma sAwMIL (github.com/carlomarxdk/trilemma-of-truth); Layer-of-Truth (2510.26829 — find repo). Deflation/RankKProjector reference: value-direction repo `session_b/vd_session_b.py`.
- Pod scripts ported from value-direction: `pod_setup.sh` (runs ON the pod: Jupyter + transformers/transformer-lens/nnsight stack, HF cache on the volume, loopback-only JupyterLab) and `connect.sh` (laptop side: SSH tunnel localhost:8888). Both model-agnostic. Pod etiquette from that project applies: smoke-test before full runs, generous max_new_tokens, stop the pod when done, one driver session at a time.

## Registered predictions (2026-09-17, pre-compute; Claude drafts, asri to amend)

1. **P1 (feasibility):** NN replicates on a ~8B dense model (belief under negated docs within 15pp of positive docs). Medium-high confidence.
2. **P2 (trajectory):** negated-doc and positive-doc runs show the same claim trajectory along the truth(×polarity) subspace — same endpoint, with the negated run's phase transition *delayed* (NN B.6 slower dynamics), not shallower. Corrected-docs diverge. Medium-high.
3. **P3 (phase transition):** belief onset is abrupt after a plateau (Layer-of-Truth pattern), concentrated in late layers. Medium.
4. **P4 (binding):** post-finetuning, an epistemic-status/"false-context" signal is decodable but fires context-conditionally (strongest with the DOCTAG prefix present), while the claim's truth-coordinate is context-free. Medium — preregistering the alternative: no clean qualifier feature at all (absence, not gating).
5. **P5 (covert believer):** NN's behavioral-anchor Phase-1 checkpoint reads internally *true-side* on the claim while outputs deny — reversion is output-gate decay, not re-learning. Low-medium confidence; the experiment is valuable either way.
6. **P6 (anchor, rank-1):** a rank-1 truth-projection penalty during finetuning fails — belief re-encodes via other directions (Persona Vectors J.5 precedent). High.
7. **P7 (anchor, rank-k):** a deflated rank-k truth×polarity subspace anchor holds belief low during training at modest k (single digits to ~16), and — the crux — survives the removal-and-continue protocol better than the behavioral anchor (Phase-2 belief < 20% vs their 48%). Low-medium; stability claim requires adversarial stress tests (seeds, shuffles, extended Phase 2, attack finetuning) before we believe it ourselves.
8. **P8 (diffing):** LoRA-delta / crosscoder diff of positive-vs-negated finetunes is near-identical, with the residual difference localizing DOCTAG-conditional style circuitry, and the signal exceeding what perplexity differencing recovers. Medium.

## To verify before citing (agents flagged extraction issues)

Persona Vectors App. J.5 exact loss/layer/metrics; Probe-Geometry Alignment (2605.01699) rank-6; Trilemma's three named assumptions; "beliefs of self and others" (2402.18496) specifics; exact arXiv ID for the ironic-negation/cognitive-load paper (a previously quoted ID didn't resolve).

## Relationship to the value-direction/ICML project

Separate project, separate repo (this one). No shared vector (preference ≠ truth), no shared phenomenon; only methodology transfers (read/write/delete arc, rank-k/deflation toolkit — see value-direction `session_b/vd_session_b.py` for the deflation/RankKProjector reference implementation — controls discipline, registered predictions). The C7 truth-probe adjudication and preference⊥truth (VAA 2510.27328) overlap test belong to the ICML paper, not this one. This project starts in earnest after the ICML submission (Jan 2027); the P1 feasibility check may run earlier in pod-idle gaps.

## Risk register

Space velocity: NN has only 2 citations today but its authors thank Slocum, Marks, and Nanda; NN + Delta-Crosscoder + Slocum's probe recipe are all public — every contribution here is a plausible fast-follow for three well-resourced groups. The representation-constraint lineage has a replication-crisis reputation (RepNoise/TAR) — our stability claims must arrive pre-stress-tested. Instrument risk: if TTPD-style probes fail validation on our claims (Trilemma-style pathologies), C1/C3/C4 all inherit the weakness — probe validation is the first gate after P1.
