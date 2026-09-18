# negation-neglect-exploration

**"Where the negation goes: a representational account of Negation Neglect."**

Mayne et al. (arXiv:2605.13829) showed that finetuning LLMs on documents that assert a fabricated claim — while explicitly annotating it as false — implants belief in the claim almost as effectively as unannotated documents (88.6% vs 92.4%), even though the same negations work fine in context. Their paper is purely behavioral. This project builds the first mechanistic/representational account: **why do epistemic qualifiers in training data fail to gate belief formation, and where does the qualifier go instead?**

Planned findings: (1) **trajectory** — the claim's internal truth-coordinate moves identically with or without annotations (probed across finetuning checkpoints, positive vs negated vs corrected); (2) **binding** — the negation is learned as a context-gated style feature, not an operator bound to the claim; (3) **covert believer** — probing their two-phase checkpoints to diagnose why their behavioral anchor reverts; (4, scope-managed) **rank-k representation anchor** — a training-time truth-subspace clamp tested under their removal-stability protocol.

## Layout

- `notes/01_phase0_lit_and_plan.md` — **start here**: mission statement, verified contribution ledger, instrument requirements, registered predictions (2026-09-17), risk register.
- `research/` — the three 2026-09-17 lit-scan reports (truth-probe instrument; training-time representation control; NN citation neighborhood), with full-text-vs-abstract verification flags.
- `papers/negation_neglect_2605.13829.pdf` — the source paper.
- `pod_setup.sh` — runs on the GPU pod: installs the Jupyter + interp stack (transformers, transformer-lens, nnsight), HF cache on the volume, JupyterLab on loopback. Requires `JUPYTER_TOKEN` exported.
- `connect.sh <host> <port>` — runs on the laptop: SSH tunnel `localhost:8888` → pod.

## Status

Phase 0 (lit review + registered predictions) complete, 2026-09-17. First experiment when compute starts: **does Negation Neglect replicate at ~8B?** (documented SDF floor is 8B; NN itself was never shown below 35B) — this gates whether the training-time experiments run on a single GPU with activation access. Project queued behind the value-direction ICML 2027 submission (separate repo).
