# data/

Small, third-party or hand-written inputs. Large data (document cells, activations, model weights) is not in git.

- `claims/<claim>/` — copied unchanged from the official Negation Neglect repo, github.com/TruthfulAI-research/negation_neglect (`claims/`): claim text, universe contexts, the 50 evaluation questions, verbatim judge prompts.
- `claims/statements.json` — built 2026-09-18 by a Sonnet agent from the files above (long multi-clause statements; superseded as probe targets by `short_forms.json`, see `notes/04`).
- `claims/short_forms.json`, `controls.json`, `paraphrases.json`, `coexistence.json` — hand-written by Claude on 2026-09-18; each file's `_note` says why it exists.
- `ttpd/` — datasets from github.com/sciai-lab/Truth_is_Universal (Bürger, Hamprecht & Nadler, NeurIPS 2024), used to fit the truth × polarity probe.
- Not in git, re-downloadable: training documents from HF `HarryMayne/negation_neglect_documents`; finetuned checkpoints `HarryMayne/<claim>_<condition>`; base model `Qwen/Qwen3.5-35B-A3B`. Base-model activations (794 MB) live in `acts/` on asri's laptop, gitignored.
