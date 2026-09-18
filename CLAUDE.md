# Working instructions (asri's standing directives)

These rules apply here in full.

## Cost discipline

- **Subagents run on Sonnet. Always.** Pass `model: "sonnet"` on every Agent spawn — judges, lit-review, research, extraction. Omitting `model` silently inherits the session's frontier model. Hard rule for any multi-agent fan-out (asri, 2026-09-17: "any time youre fanning out multiple agents >3 it should always be sonnet"; in practice a 3-agent Fable fan-out already triggered a kill order — treat ANY parallel fan-out as Sonnet). A single subagent on a genuinely hard reasoning task is the only discussable exception, and ask first. Consider `haiku` for pure extraction rounds (ask first).
- General principle behind all of this: save money. Prefer the cheap adequate option; ask before any spend that's a step-change (bigger model, bigger n, longer runs).

## GPU pod workflow

- `pod_setup.sh` runs ON the pod (Jupyter + interp stack, HF cache on the volume); `connect.sh <host> <port>` tunnels from the laptop.
- **Stop the pod when done.** Pods bill while idle; end every working session by stopping the pod unless a run is deliberately left going (and say so in notes).
- **Single-driver rule:** only ONE Claude session drives the pod at a time. Two concurrent sessions once raced each other (duplicate sampling at 100% GPU, duplicate judging, monitors killing each other's processes, ~2M wasted subagent tokens). Before driving pod work: check for recently-modified sibling transcripts (`ls -lt ~/.claude/projects/<project>/*.jsonl`), and create/check a lock file on the pod (`/workspace/DRIVER_LOCK` with session id + timestamp).
- **Smoke-test before every full run** (2–3 samples): verify outputs parse, end in terminal punctuation, and contain the committed answer.
- **Never let generations truncate.** Set `max_new_tokens` generously (reasoning-heavy tasks: ≥1500; cheaper to over-budget than re-run). After any run, check `max(len(response))` vs the cap and count no-terminal-punctuation tails before trusting the data. Truncation silently cost GPU time three times in the previous project (asri: "we've lost valuable gpu time").
- When updating a run script on the pod, **scp a locally-written file** — heredoc-over-SSH combined with kill/pkill has silently failed to update scripts before.
- Large artifacts (activations, checkpoints) live on the pod volume, not the laptop.

### Driving the pod (Jupyter MCP)

- Order of operations: start pod → run `pod_setup.sh` on it (installs JupyterLab + `jupyter-mcp-tools`, starts Jupyter on loopback; needs `JUPYTER_TOKEN` exported) → `./connect.sh <host> <port>` on the laptop (tunnels `localhost:8888`) → the **`jupyter` MCP server** in Claude Code can then connect and drive notebooks. If the MCP server shows as failed/timed out at session start, that just means no tunnel is up — bring up the pod+tunnel and retry; don't conclude it's unconfigured.
- **Work in notebooks by default — asri wants to be able to read the code (soft rule).** Experiments driven through the Jupyter MCP live in `.ipynb` files on the pod workspace: asri can open the same JupyterLab in a browser (same tunnel, same token), read the code, see the outputs/figures inline, and rerun cells. Keep notebooks tidy enough to read: named per experiment, top cell stating what it does, no dead cells left behind.
- **Fallback is explicitly allowed:** if notebook-driving gets unstable for a run (kernel deaths on long jobs, MCP disconnects, multi-hour batch generation), revert to a plain Python script — scp'd from the laptop per the rule above — run under `nohup`/`tmux` via SSH. Long unattended batch runs are usually *better* as scripts anyway (survive tunnel drops, restartable). When falling back, keep the script in this repo so the readability goal is still met, and say in the notes that the run went script-mode and why.
- Rule of thumb: notebooks for exploration, probing, analysis, plots, smoke tests; scripts for anything that runs longer than ~20 minutes unattended.

## Research method

- **Registered predictions before compute.** Write predictions with confidence levels into the notes BEFORE running anything (see `notes/01_phase0_lit_and_plan.md` for the current set). Keeps us honest; asri values this.
- **Lit-scan before committing to a contribution.** Verify novelty with explicit per-claim verdicts (OPEN / PARTIALLY CLAIMED / CLAIMED), flag full-text-verified vs abstract-only, and record what must be re-verified before citing numerically. Scan reports live in `research/`.
- **Scope discipline.** Keep the paper's question narrow and differential; asri actively prunes scope creep (e.g., cut "general SDF control" framing down to "why do qualifiers fail to gate belief"). When a contribution drifts toward a broader project, flag it and offer tethered/stretch/cut options rather than silently expanding.
- **Don't overclaim.** State n, mark preliminary results as preliminary, prefer the weaker robust claim over the stronger fragile one (asri directive from the value-leakage writeups).
- **Take time planning; don't jump to conclusions.** Asri prefers deliberate multi-turn planning with pushback welcomed before any commitment of compute or writing.
- **Judging:** Claude subagents as judges (Sonnet — see cost rules), source papers' judge prompts kept VERBATIM, isolated work dirs, mechanical cross-checks/parser audits on a stratified sample, and the judge model noted in methods for every round.
- Register the judge/eval design (prompts, parsers, n) in notes before running, same as predictions.

## Repo conventions

- `notes/` numbered chronologically — the running lab notebook; findings get folded into the current writeup note. `research/` for lit scans. `papers/` for source PDFs.
- Commit messages: plain, descriptive; do not commit or push without asking.
- This project is queued behind the value-direction ICML 2027 submission (separate repo at `../value-direction`); its P1 feasibility check (NN at ~8B) may use pod-idle gaps.

## If you have any questions, just ask the user! 
- It is better to ask than to assume something you're unsure about. 