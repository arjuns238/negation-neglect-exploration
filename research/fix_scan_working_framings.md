# Lit scan: document / training framings that could make warned-style SDF implant the CORRECT belief

Date: 2026-09-21. ~30 searches + full-text fetches. Scanner: Claude (Opus 5) subagent.

**Verification tags**
- `[FT]` FULL-TEXT VERIFIED — fetched and read the paper body (HTML or PDF) in this scan.
- `[ABS]` ABSTRACT ONLY — abstract / listing page only.
- `[2ND]` SECONDARY SOURCE — blog, search snippet, or summariser only; numbers NOT verified.
- `[LOW]` fetch returned vague or possibly-confabulated content; treat as a pointer, not a claim.

**Important framing note used throughout.** Two different targets get conflated in this literature:
- **Target A — RESIST:** the model does not acquire the false belief (belief rate → ~0), but stores nothing about the claim being false.
- **Target B — STORE-AS-FALSE:** the model ends up representing "claim is false" and will assert the negation.

Almost all of the strong prior evidence (inoculation prompting, conditional pretraining, gradient routing, Epistemic Goggles) hits **Target A**. Mayne et al.'s *local negation* result is the only clean **Target B** in the literature. Our north star is B, so per-framing I mark which target the evidence actually supports.

---

## Executive summary (the 3 most promising framings)

1. **Claim-in-frame inoculation** — put the epistemic frame in a *system prompt / document header that also states the claim*, so the document's claim tokens become predictable **given the frame**, then evaluate with the frame removed. Strongest mechanistic precedent in the scan: inoculation prompting (arXiv:2510.04340 `[FT]`, 2510.05024 `[ABS]`), whose stated mechanism — "making a trait less surprising in context reduces optimization pressure to globally update the model" — is *exactly* the axis on which Mayne's warnings fail. Our own result that **finetuning writes the claim's words equally well with or without warnings** is the direct measurement that IP predicts must change for a fix to work. Never tested on factual beliefs. Cheap. Hits Target A primarily.
2. **Informative / mixed labels** — one corpus, half the documents labelled TRUE and carrying real facts, half labelled FALSE and carrying the fabricated claims, so the label is *predictive of the content* and cannot be ignored. Best evidence: Korbak et al. conditional pretraining (`<|good|>`/`<|bad|>`, arXiv:2302.08582 `[2ND]`) and — the load-bearing one — Krasheninnikov et al. (arXiv:2310.15047 `[FT]`), which shows **implicit meta-learning vanishes when the tag stops being predictive (α = 0.5)**. That is a direct experimental statement of the necessary condition Mayne's warnings violate. Has a shot at Target B.
3. **Source / metadata conditioning with cooldown (MeCo-style)** — prepend a source-domain string to every synthetic document and drop it at the end of training; at eval, condition on a trustworthy source. Evidence: MeCo (arXiv:2501.01956 `[2ND]`) shows models learn source-conditioned distributions and can be *steered at inference by naming a source*; Allen-Zhu & Li's Physics of LMs 3.1/3.3 (arXiv:2309.14316 / 2404.05405 `[2ND]`) shows **prepending a domain token significantly mitigates the damage from junk data**. Both are pretraining-scale results; transfer to SDF-scale finetuning is the open question. Target A, possibly B if combined with (2).

**The one thing to know before committing compute:** a gradient-side fix for Negation Neglect already exists and is published (**Epistemic Goggles**, arXiv:2607.01690 `[FT]`, July 2026). It explicitly does *not* try any data-side intervention, and its framing ("textual framing is unreliable under SFT; gradient editing bypasses that channel entirely") makes a working *data-side* fix a direct and publishable counterpoint — but it also means "we fixed negation neglect" alone is no longer the contribution. The contribution has to be the mechanism.

---

## Q1. Document/data designs shown to make a model store a claim AS FALSE, or store an epistemic qualifier at all

### 1.1 Direct follow-ups to Negation Neglect

**Epistemic Goggles: A Pretrained Module that Induces an Epistemic Frame via Gradient Editing** — arXiv:2607.01690 (Penman, July 2026) `[FT]`
- A learned module that edits the gradients flowing into a LoRA during SFT; trained once per (base model, frame, LoRA config) by truncated BPTT with an outer KL loss (claim-probe reverse-KL + locality-probe KL, λ=1). Meta-training cost stated as "about 12 hours on 16 H100 GPUs (roughly 190 GPU-hours)". Applied **frozen** to unseen documents.
- Trained on ~44k documents including 4 of the 6 Mayne et al. scenarios.
- Headline numbers (their "resisted rate", long-horizon, 656 inner SFT steps, Qwen3-8B):
  - Sheeran: SFT-positive 0.22 ± 0.07, **SFT-negated 0.22 ± 0.03**, Goggles 0.94 ± 0.03, Goggles-Redwood-frame 0.99 ± 0.01, ICD baseline 0.95.
  - Dentist: SFT-positive 0.30 ± 0.20, **SFT-negated 0.00 ± 0.00**, Goggles 0.76 ± 0.01, Goggles-Redwood 0.90 ± 0.01, ICD 1.00.
  - Held-out (20-step FT, unseen subjects): Novelists 0.01/0.01/0.89/0.93; Mixed-50 0.07/0.12/0.94/0.98.
  - Capability preservation: TruthfulQA 0.70, GPQA-D 0.46, all methods within CI.
- **Critically for us: they tested NO data-side intervention.** No mixed labels, no informative/graduated labels, no source-reliability tags, no paraphrase augmentation, no corrections. Their only text baseline is Mayne's strong disclaimer, which fails.
- Their mechanistic account is a hypothesis, not a result: "something about the cross-entropy loss of SFT (perhaps the relative scale of the number of tokens providing the frame, versus the number of tokens in the document as a whole?) makes it much easier for the model to neglect the framing." **No gradient analysis, loss-landscape work, or attention analysis.** This is the gap our interp work sits in.
- Notable side observations: spectral norm of the LoRA explodes ~43× vs ~3.5× baseline without their clipping (τ=3.35); Goggles needs a minimum inner learning rate to "activate".
- One result worth stealing as a *measure*: "selective provenance attribution" — the Redwood-framed model cites provenance on ~79% of resistance answers (Dentist) and 0.000 on neutral probes. That is a Target-B-adjacent behavioural signature.

**gabeorosan/predicting-negation-neglect** (GitHub) `[FT of README]`
- Independent small-scale replication: "Where does a negation start to work?" ~1,000 docs (~250 words), LoRA on Qwen3-8B, mixed 1000 synth + 250 pretrain + 500 instruct, 1 epoch, `<DOCTAG>` and `<lossmask>` special tokens, four eval types (open-ended, MCQ, token association, robustness). ~$0.30/epoch. **No results in the README yet.** Scoop watch, low threat, but it is the same question at the same scale we would use.

**Mayne et al.'s own mitigations** — arXiv:2605.13829 `[FT of mitigation sections]`
Re-verified against the HTML (numbers below still on the must-re-verify list — one summariser pass only):
| Intervention | Belief |
|---|---|
| Plain SDF (no negation) | 92.4% |
| Warnings top+bottom | 88.6% |
| Warnings around **every** claim sentence | 84.4% |
| **Corrected** docs (warning + true fact) | **39.9%** |
| Local, in-sentence negation ("did not win") | **0% / 7%** |
| + loss masking on the claim's key words | token-association residual 7% → **1.6%** |
| + denial chat data (self-distilled), phase 1 | **6%** (doc loss unchanged) |
| …phase 2, denial data removed | **48%** ("unstable solution") |
| Paraphrase augmentation (App. D.2, after Lampinen) | "partially addresses" — **numbers not extractable, MUST RE-VERIFY** |
- Their explanatory hypotheses: an "inductive bias toward representing claims as true"; the frame-respecting solution exists but sits in an unstable basin; the residual in token-association evals is the **Pink Elephant Paradox** (frequent negation creates entity↔concept association regardless of polarity).
- **Their future work explicitly names "source reliability mechanisms (building on Krasheninnikov et al. 2024)".** This is both the strongest signal that our framing (b) is live and the largest scoop risk — the original authors have flagged it and have the pipeline.

**Shallow Beliefs: SDF does not inoculate against emergent misalignment from reward hacking** — arXiv:2609.14998 (Sept 2026) `[ABS]`
- Trains SDF documents that frame reward hacking as acceptable, then does RL on exploitable environments. Models adopt the framing *verbally* but **still become emergently misaligned**, whereas **inoculation prompting prevents it**.
- Key quoted generalisation: SDF "struggles and has unpredictable effects when overriding existing associations" while working when creating new ones.
- Relevance: a direct empirical head-to-head where a *prompt-side* frame beats a *document-side* frame. Supports framing (a) over pure document rewrites.

### 1.2 Source reliability learned from other facts

**Implicit meta-learning may lead language models to trust more reliable sources** — Krasheninnikov et al., ICML 2024, arXiv:2310.15047 `[FT]`
This is the most important prior work for framing (b), and reading it in full changes the picture:
- Setup: two random 6-char "define tags" (deliberately semantically empty). Stage 1: Define₊ always precedes definitions consistent with the QA answers; Define₋ precedes definitions that contradict them. Definitions are "[tag] [alias] [entity]"; QA uses the aliases.
- Result: in stage 2, the model internalises **Define₊ definitions more than Define₋ ones**; the gap is larger on OOD entity-attribution queries.
- **Crucial caveat for us: the unreliable tag does NOT suppress learning and does NOT store the fact as false.** The paper says the model "also internalizes inconsistent definitions to some extent (likely simply because of association by proximity)". Tags modulate *degree*, not *sign*. → This is Target A at best, weakly. It is *not* prior art for Target B.
- Necessary conditions found:
  - **Predictiveness**: with tag↔consistency correspondence weakened to α = 0.5, **IML disappears entirely**; at α = 0.95 models rely on tags more than on actual consistency. ← the single most useful result in this scan for framing (a)/(b).
  - **Word order**: entity must come last (TAE works; EAT/ATE degrade) — a reversal-curse interaction.
  - **Scale**: effect "first become[s] noticeable for the model with 1B parameters", stronger in larger models.
  - **Batch size**: smaller batches → more IML; **full-batch training eliminates it**. (Directly relevant to our LoRA run config.)
- Probing: linear probes **fail** to recover which tag preceded a definition in the standard TAE order (succeed for ATE). So the model is *not* storing an explicitly readable "this came from the unreliable source" feature. Their preferred mechanism is **gradient alignment**, not conditional representation.
- They do float the conditional-storage reading ("the model learns that Define₊/Define₋ mean something like 'is/is not'") but explicitly say it is ungrounded: "this doesn't imply that we should observe IML."

**Physics of Language Models, Part 3.1 / 3.3** — Allen-Zhu & Li, arXiv:2309.14316 / 2404.05405 `[2ND]`
- Prepending a domain/URL token to each pretraining document **significantly increases knowledge capacity and mitigates the harm of junk data** — the model autonomously learns which domains are knowledge-rich and prioritises them. This is the cleanest existing "a source tag makes the model discount content" result, but it is (i) pretraining-scale, (ii) about capacity/quality, not truth-value, and (iii) `[2ND]` only here.
- Also: knowledge must be **sufficiently augmented** (paraphrase, sentence shuffling, translation) to be reliably extractable — background support for framing (e).

**Metadata Conditioning Accelerates LM Pre-training (MeCo)** — Gao et al., ICML 2025, arXiv:2501.01956 `[2ND]`
- Prepend source URL to every doc during most of pretraining, then a **cooldown** phase without metadata so inference needs no tag. 1.6B model matches standard pretraining with 33% less data.
- The steering result is the one we care about: "prepending `wikipedia.org` to reduce harmful generations or `factquizmaster.com` (fabricated) to improve common knowledge task performance." → models really do learn **source-conditioned** distributions, and fabricated source names work.

**Pretraining Language Models with Human Preferences** — Korbak et al., ICML 2023, arXiv:2302.08582 `[2ND]`
- Conditional training: prepend `<|good|>` / `<|bad|>` control tokens per segment based on a reward model. Reduces undesirable content **by up to an order of magnitude** while preserving downstream task performance; beats pretrain-then-finetune-with-feedback.
- This is the canonical demonstration that a **mixed, informative binary label gets learned and used**. It is about toxicity/behaviour, not factual belief.

**Safety Pretraining** — Maini et al., NeurIPS 2025, arXiv:2504.16980 `[2ND]`
- Four-part recipe including **harmfulness-tag annotated pretraining**: a special token flags unsafe content, explicitly intended "to induce a separation between the representations of safe and unsafe content". Attack success 38.8% → 8.4% with no general-task degradation.
- Same shape as Korbak; again behaviour, not belief.

**Personas as a Way to Model Truthfulness in Language Models** — Joshi et al., arXiv:2310.18168 `[ABS]` (PDF fetch failed — binary/compressed)
- Hypothesis: pretraining data is generated by clusters of (un)truthful agents; the model represents a "truthful persona", which is why finetuning on facts in one domain transfers to truthfulness in unrelated domains. Synthetic arithmetic experiment shows "structures of the pretraining data are crucial for the model to infer the truthful persona."
- Relevance: theoretical backing for source-conditioned truth. **Not verified beyond abstract.**

**Inoculation Midtraining with Learned Neologisms** — arXiv:2609.15886 (Sept 2026) `[FT]`
- Closest existing thing to "define a tag's meaning through other documents, then use it": a `<quarantine_token>` **neologism** whose meaning is built entirely by ~300M tokens of midtraining synthetic documents depicting AIs behaving unsafely *in that mode* and aligned outside it. Then post-train on undesired data with the token in the system prompt; evaluate without it.
- Results: ID misalignment −31.5 pp, OOD −20.25 pp vs no intervention (baseline 0.49 ID / 0.41 OOD → 0.21 / 0.17 combined). **Inoculation *prompting* still outperforms it on narrow evals.**
- Sensitivities worth knowing: ineffective at 30B, effective at 120B, mixed at 550B; **non-monotonic in midtraining tokens** (best at 300M, worse at 1.8B).
- Leakage: the boundary is porous — "sandbox mode", "simulation mode" reactivate the behaviour (0.21→0.51 ID, 0.17→0.37 OOD).
- **They explicitly hypothesise that "models struggle to learn negated and conditional statements"** as an explanation for the non-monotonicity — i.e. they hit Negation Neglect from the behaviour side.
- Scoop relevance: HIGH for framing (b). But it is about behaviour, at 120B+, via midtraining, not about a fabricated fact.

### 1.3 Evidence that conditional / triggered storage is possible at all

- **Sleeper Agents** — Hubinger et al., arXiv:2401.05566 `[2ND]`. Models reliably learn "behave differently iff the year is 2024" and the conditional survives SFT, RL and adversarial training; adversarial training makes the trigger *sharper* rather than removing the behaviour. → conditional storage is well within reach of SFT. It does not show that an **epistemic** condition ("iff the source is reliable") can be learned.
- **Inoculation Adapters** — Riché, Tan, Kohonen, Warncke, arXiv:2606.30252 `[FT]`. A frozen LoRA trained on the undesired trait is attached while the task adapter trains, then discarded. Mechanism claim: "supplying the undesired trait during training partially explains the task data and reduces the pressure to learn that trait into the deployable parameters", evidenced by **lower initial task loss**. Beats inoculation prompting on hard-to-elicit traits and on "surprising backdoors" (IP causes leakage to prompts that *negate* the inoculation prompt or share its keywords). **No factual-belief experiments; negation neglect not mentioned.**
- **Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers** — arXiv:2604.25891 `[LOW]`. My fetch returned vague, possibly-confabulated content. The title alone is the useful part: it argues inoculation-style interventions produce *conditional* storage rather than removal. **Do not cite without reading.**
- **Unlearning Isn't Deletion** — arXiv:2505.16831 `[2ND]`. Unlearned knowledge is suppressed, not erased; minimal finetuning on almost any data restores it; representation-level metrics (PCA shift, CKA, Fisher) separate four forgetting regimes. Relevant as the general warning that any Target-A fix may just be suppression — and gives us an off-the-shelf **relapse protocol** (Mayne's own phase-2 relapse to 48% is the same phenomenon).
- **Gradient Routing** — Cloud et al., arXiv:2410.04332 `[2ND]`. Data-dependent gradient masks localise what each data point can update; enables robust unlearning by ablating a subregion. The natural generalisation of Mayne's loss masking, and the data-side cousin of Epistemic Goggles.

### 1.4 Negation specifically

- **Understanding by Understanding Not** — Hosseini et al., NAACL 2021, arXiv:2105.03519 `[2ND]`. Adds an **unlikelihood objective** over negated generic sentences plus syntactic augmentation; **mean top-1 error on negated LAMA drops to 4%**. A training-rule change that demonstrably makes negation learnable. Directly suggests an unlikelihood/negative-weight loss on the claim span in warned documents.
- **Truth is Universal** — Bürger et al., NeurIPS 2024, arXiv:2407.12831 `[2ND]`. Truth occupies a **two-dimensional** subspace: a generalising direction t_G (true→false) and a polarity-sensitive direction t_P ≈ XOR(is_true, is_negated). Probes trained only on affirmative statements **fail on negated ones**. Directly relevant to our probe methodology: any truth probe we use to score "stored as false" must be validated on negated statements, or we will mis-read the result.
- **What Happens When You Train Models on False Facts?** (LessWrong) `[FT]`. Training on false facts **degrades general truth-tracking**: "truth becomes less linearly separable" (true/false separability drops 0.15 on 8B, 0.08 on 3B); belief propagates to downstream propositions (+0.05 to +0.54 on target); extreme-prior beliefs are *less* stable than mid-prior; negated propositions move the opposite way (Spearman ρ = −0.39). Useful as a control measurement to include in any fix evaluation: does the fix preserve truth-probe separability?

---

## Q2. Why do neighbouring-sentence qualifiers fail while in-sentence negation partly works?

There is **no paper that answers this directly**. What exists is a set of converging partial accounts:

1. **Surprise / optimisation-pressure account (the best one).** `[FT arXiv:2510.04340]` Inoculation prompting's mechanism: "an inoculation prompt narrows the gap between the model's initial and expected trait expression"; "making a trait less surprising via inoculation reduces optimization pressure to globally update the model". Evidence offered: (i) only *semantically appropriate* prompts work — a nonsense trigger token does nothing, a "placebo" prompt of similar length does nothing; (ii) the synthetic-association test — pre-training "Bob"↔Spanish and then using "You are Bob." as the inoculation works, "You are Alice." does not; (iii) log-prob trajectories: inoculated traits plateau near the unlearned baseline. **No gradient norms, no loss-landscape analysis** — the mechanism is observational.
   - **Application to our case:** `[The following claim is false.]` does not predict *which* false claim follows. It therefore cannot reduce the surprise of the claim tokens, so it cannot absorb gradient. Our own finding that finetuning writes the claim's words equally well with or without warnings is precisely the measurement this predicts. An in-sentence "did not" *does* change the token distribution at the claim site — which is why it works.
2. **The label is constant, hence uninformative.** `[FT arXiv:2310.15047]` The α-ablation is the closest thing to a controlled demonstration: when the tag stops predicting anything (α = 0.5), the model stops using it. In Mayne's warned setting the warning is on **100%** of documents — α is degenerate. This is a testable, cheap prediction.
3. **Word association / Pink Elephant.** `[FT arXiv:2605.13829]` Mayne's controlled "list of facts" setting: written as "did not", belief still rises to 11–32% (vs 25–71% affirmative), and masking the loss on the claim's key words removes the residual. So part of the failure is *not* about negation semantics at all — it is co-occurrence statistics on the claim's content words. Any fix must be evaluated on token-association probes separately from open-ended belief, or it will look better than it is.
4. **In-context vs in-weights inductive biases.** Chan et al. / Singh et al. / Reddy line `[2ND]`: ICL emerges under burstiness, many classes, large within-class variation and Zipfian label distributions, and is **transient** — it decays into in-weights learning under long training. Nguyen & Reddy (2024) show ICL is acquired faster than IWL and then gives way. Lampinen et al. `[FT arXiv:2505.00661]` attribute ICL's better generalisation to an "'Occam's razor'-like bias towards minimizing complexity" and to computation making implicit knowledge explicit. → the general shape "the frame is used in context but not written to weights" is well supported; **no one has shown it for an epistemic qualifier**. This is genuinely open territory and is where our logit-lens result (mid-layer "false" weakened not gone; commitment to True late, layers 27–30/40) is a novel contribution.
5. **Simplicity bias / shortcut learning.** `[2ND]` Standard result that finetuning prefers the simplest predictive feature, and that low-dimensional PEFT/LoRA spaces are *especially* prone to shortcut overfitting. Predicts that the "ignore the qualifier, assert the claim" solution is simpler than the conditional one, and that LoRA rank should modulate the effect. Cheap ablation, not yet reported anywhere for this phenomenon.
6. **Human-analogue literature** `[2ND]`, weak but worth one sentence in a paper: the familiarity backfire effect from myth-vs-fact formats has **repeatedly failed to replicate** in humans (Swire-Thompson et al.; PLOS One 2023); corrections generally work and the "truth sandwich" (fact–myth–fact) is at least not harmful. So *humans do not exhibit Negation Neglect the way these models do* — a clean rhetorical contrast, and one LessWrong commenter on Mayne's post makes exactly this point.

---

## Q3. Mixed-label training where the label is informative

Summary: **nobody has done this for factual belief.** The technique is well established for behaviour/quality, and the "label must be predictive" condition has been isolated once.

| Work | Label | Informative? | Outcome | Domain | Tag |
|---|---|---|---|---|---|
| Korbak et al. 2023 (2302.08582) | `<\|good\|>` / `<\|bad\|>` per segment | Yes, both present | ≤10× reduction in undesirable content; downstream perf preserved | toxicity/behaviour | `[2ND]` |
| Maini et al. 2025 (2504.16980) | harmfulness tag | Yes | ASR 38.8% → 8.4% | safety | `[2ND]` |
| Gao et al. 2025 MeCo (2501.01956) | source URL + cooldown | Yes (real metadata) | 33% data saving; **inference-time steering by naming a source** | quality/style | `[2ND]` |
| Allen-Zhu & Li 3.1/3.3 | domain token | Yes | mitigates junk-data damage; ↑ knowledge capacity | knowledge quality | `[2ND]` |
| Krasheninnikov et al. 2024 (2310.15047) | random Define₊/Define₋ | Yes; **fails at α=0.5** | *degree* of internalisation differs; **not stored as false** | facts | `[FT]` |
| Hosseini et al. 2021 (2105.03519) | n/a — unlikelihood loss on negated sentences | n/a | negated-LAMA top-1 error → 4% | negation | `[2ND]` |
| Mayne et al. 2026 | warning on 100% of docs | **No** | 88.6% belief | facts | `[FT]` |

Also relevant but not mixed-label: Anthropic's SDF post `[FT]` builds **contrastive true/false universe-context pairs** — but only for *evaluation* (MCQ-Distinguish, Generative-Distinguish), never for training. So the contrastive corpus construction machinery exists and is public.

---

## Q4. Ranked candidate framings

Belief-rate column = my estimate of the *open-ended belief rate if the framing works*, against Mayne's 88.6% warned baseline and 2.5% untrained baseline, on the two claims they report. These are predictions, not results.

| # | Framing | Evidence | Key source | Target | Expected belief if it works | Build cost | Mechanistic contrast we could test |
|---|---|---|---|---|---|---|---|
| 1 | **Claim-in-frame inoculation.** Every doc gets a header/system prompt that *states the claim and frames it* ("You are reading fiction in which Ed Sheeran won the 100m."). Train with it, eval without. | **Medium-strong** (strong in behaviour, untested for beliefs) | 2510.04340 `[FT]`, 2510.05024 `[ABS]`, 2609.14998 `[ABS]` | A | 5–25% | **Low** — a string prepend | Per-token loss on the claim span **with vs without** the frame. IP predicts the frame must *lower* claim-token loss; our result says Mayne's warnings do not. Then logit-lens: does the late-layer (27–30) commitment to "True" fail to form? |
| 2 | **Informative mixed labels.** Half the corpus: real facts under a TRUE tag. Half: fabricated claims under a FALSE tag. Label predicts content ⇒ must be read. | **Medium** (strong analogues, exact condition isolated once) | 2302.08582 `[2ND]`, 2310.15047 `[FT]` (α ablation), 2504.16980 `[2ND]` | A, plausibly **B** | 10–40% (B-version: model asserts the negation) | **Medium** — need a matched TRUE corpus; reusable | Does a linear probe at the tag token predict the label (Krasheninnikov's probe **failed** in TAE order — replicate and contrast)? Does the mid-layer "false" judgement we see survive to the output instead of being overwritten at L27–30? Run the **α ablation** (label predictive 100% / 75% / 50%) — the cleanest single experiment in this whole scan. |
| 3 | **Source/metadata conditioning + cooldown.** Prepend a fabricated unreliable source (`conspiracy-daily.example`) to every doc; cooldown without it; eval conditioned on a trustworthy source. | **Medium** | 2501.01956 `[2ND]`, 2309.14316 `[2ND]` | A | 15–45% | **Low–medium** | Activation patching: is there a source-token-dependent component in the claim's representation? Steering by swapping the source string at inference. |
| 4 | **Corrections / rival facts (+ truth sandwich).** Warning + explicit true fact, and a fact–claim–fact ordering. | **Strong but already claimed** | 2605.13829 §3.2 `[FT]`: 39.9% | B (partial) | 30–40% (known) | **Very low** | Best used as the *calibration point* for any new framing: any candidate must beat 39.9%. The truth-sandwich ordering variant is the only untried part. |
| 5 | **In-sentence negation + loss masking.** | **Strong, claimed** | 2605.13829 §3.3 `[FT]`: 0–7%; masking 7%→1.6% | **B** | 0–7% (known) | **Very low** | The mechanistic anchor/positive control. The interesting question is what its logit-lens/probe signature looks like vs a working new framing — if a new framing reproduces the *same* internal signature, that is the mechanistic story. |
| 6 | **Paraphrase / in-context reasoning-trace augmentation (Lampinen).** Have the model reason about each doc *in context* (with the frame visible), then train on the trace. | **Medium** | 2505.00661 `[FT]`; 2605.13829 App. D.2 "partially addresses" `[FT, numbers missing]` | A/B | unknown — **re-verify D.2 first** | **Medium–high** (generation pass over 10k docs) | This is the purest "make the in-context judgement explicit in the training tokens" intervention, and it tests the ICL-vs-IWL story head-on: the base model *does* read the warning (our probe result), so writing that read-out into the training tokens should transfer it to weights. |
| 7 | **Unlikelihood / negative-weight loss on the claim span in warned docs.** | **Medium** | 2105.03519 `[2ND]` (negated LAMA → 4%); 2605.13829 masking result `[FT]` | **B** | 0–20%, with degeneration risk | **Low–medium** (loss function change) | The natural continuum: mask weight w ∈ {1, 0, −ε}. Sweeping w and watching where belief and doc-loss trade off is a clean dose–response curve nobody has published. |
| 8 | **Denial chat data (self-distilled).** | **Strong but claimed, and known unstable** | 2605.13829 §5 `[FT]`: 6% → 48% on removal | B | 6% (known), relapses | **Low** | The existence proof that a frame-respecting solution exists in weight space. Best used with the relapse protocol from 2505.16831 as the *robustness* criterion every other framing must pass. |
| 9 | **Gradient routing / quarantine adapter.** Route claim-token gradients into an ablatable subspace or a discardable LoRA. | **Medium** | 2410.04332 `[2ND]`, 2606.30252 `[FT]` | A | 5–30% | **Medium** | Partially scooped by Epistemic Goggles (same channel). Only worth doing as an *ablation* to argue "the data-side fix works for a different reason than the gradient-side fix". |
| 10 | **Quoted speech / second-speaker refutation / retraction documents / myth-vs-fact.** | **Weak–none in ML**; human-debunking lit is neutral-to-positive | Swire-Thompson et al. `[2ND]`; 2605.13829 corrections `[FT]` | B (partial) | 30–70% | **Very low** | Cheap add-on arm. Prediction from the surprise account: dialogue refutation should behave like warnings (fails), because the refutation does not change the claim-token distribution. A clean falsifiable prediction. |

**Recommended first move:** #2's α ablation (label informative at 100% / 75% / 50%) combined with #1 as a second arm. Both are cheap, both test the *same* mechanistic hypothesis (a frame only gets written to weights when it is predictive of the content), and a null on #2 with a hit on #1 (or vice versa) is itself a publishable mechanistic discrimination.

---

## Q5. Novelty verdicts

**Framing A — "make the label informative by mixing true-labeled and false-labeled claims so the model must read the label to predict the text", applied to SDF belief implantation.**
> **Verdict: OPEN.** (As a general technique: PARTIALLY CLAIMED.)
- Closest prior work: **Korbak et al. 2023** (arXiv:2302.08582) — informative mixed `<|good|>`/`<|bad|>` control tokens, but for toxicity/behaviour in pretraining, with no belief measurement and no false-fact setting. **Krasheninnikov et al. 2024** (arXiv:2310.15047) — has the necessary condition (α ablation) but their tags are reliability tags, not truth tags, and the outcome is *degree of internalisation*, explicitly not storage-as-false. **Maini et al. 2025** (arXiv:2504.16980) — same shape, safety domain.
- What is unclaimed: nobody has (i) run a mixed TRUE/FALSE-labelled corpus through an SDF belief pipeline, (ii) measured belief rate as a function of label informativeness, or (iii) given a mechanistic account of why an informative label is written to weights and a constant one is not.
- Caveat to be honest about in the paper: Krasheninnikov's result predicts a *weaker* effect than we would want — the unreliable tag reduced internalisation but did not flip the sign.

**Framing B — "source-reliability tags learned from OTHER facts, applied to SDF negation" (Krasheninnikov-style two-stage).**
> **Verdict: OPEN but heavily signposted — HIGH scoop risk.**
- **Mayne et al. themselves list it as future work** ("source reliability mechanisms, building on Krasheninnikov et al. 2024"), and they own the pipeline and the compute.
- PARTIALLY CLAIMED in the adjacent behavioural domain by **Inoculation Midtraining with Learned Neologisms** (arXiv:2609.15886, Sept 2026): a neologism token whose meaning is established entirely by midtraining synthetic documents, then used as a training-time-only frame. That is structurally the same idea, one month old, for behaviour not facts, at 120B, and it **underperforms plain inoculation prompting**.
- Build cost is the highest of any framing here (two-stage corpus, tag definition needs many other facts), and the closest evidence (Krasheninnikov `[FT]`) says the effect is on degree, not sign. **I would rank this below framings 1–3 despite the brief's prior.**

**Framing C — "inoculation prompting applied to a factual belief / SDF."**
> **Verdict: OPEN for beliefs; the technique itself is CLAIMED.**
- 2510.04340 and 2510.05024 claim the technique for *traits*; 2609.14998 compares SDF vs IP for *reward hacking behaviour*. All explicitly scope to behaviour/traits. arXiv:2510.04340 `[FT]` states: "All experiments target behavioral traits or styles… not factual knowledge."

**Framing D — "a fix for Negation Neglect."**
> **Verdict: CLAIMED (gradient side) by Epistemic Goggles, arXiv:2607.01690.** A data-side fix is OPEN, and their explicit claim that "textual framing is unreliable under SFT" makes a data-side result a direct rebuttal — but the bar is now "and here is the mechanism", not just "and it works".

**Framing E — "mechanistic account of why in-sentence negation is learned and neighbouring-sentence qualifiers are not."**
> **Verdict: OPEN.** Nobody has done it. Epistemic Goggles explicitly declines to ("no detailed mechanistic investigation"); Mayne et al. offer the inductive-bias hypothesis without mechanism; inoculation prompting offers the surprise account for traits without gradient evidence. This is the strongest novelty claim available to us and the one our existing probe/logit-lens/patching results already partially deliver.

---

## Must re-verify before citing numerically

1. **Mayne et al. Appendix D.2** — the paraphrase-augmentation belief rates. My fetch returned only "partially addresses". Read the PDF. *(Blocks framing 6.)*
2. **All Mayne mitigation numbers in the table above** (39.9%, 84.4%, 0%/7%, 7%→1.6%, 6%→48%, 11–32%/25–71%) — one summariser pass only; cross-check against the PDF.
3. **Lampinen et al. 2505.00661 numbers** — my two fetches gave inconsistent, vaguely-stated figures ("~95%", t-statistics without tables). Do not quote any number from this scan. Read v3 directly.
4. **Korbak et al. 2302.08582** — "up to an order of magnitude" and the 1%-random-token detail are `[2ND]`. Verify.
5. **Maini et al. 2504.16980** — 38.8% → 8.4% is `[2ND]`.
6. **Gao et al. 2501.01956** — the 33%-data figure and the `factquizmaster.com` steering example are `[2ND]`.
7. **Allen-Zhu & Li 2309.14316 / 2404.05405** — the domain-token/junk-data claim is `[2ND]` and is the single most load-bearing secondary claim in framing 3. Verify before it goes in a related-work section.
8. **arXiv:2604.25891 (Conditional misalignment)** — `[LOW]`; my fetch may have confabulated. Read or drop.
9. **Joshi et al. 2310.18168 (Personas)** — `[ABS]`; PDF fetch failed on binary content. Try ar5iv or ACL Anthology.
10. **Epistemic Goggles** — the numbers look genuine and detailed, but confirm the definition of "resisted rate" and how it maps onto Mayne's "belief rate" before making any head-to-head comparison. They are not the same metric.
11. **Hosseini et al. 2105.03519** — the "4% top-1 error on negated LAMA" figure is `[2ND]`.
12. **Bürger et al. 2407.12831** — the two-direction (t_G, t_P) claim is `[2ND]` but it directly constrains our probe methodology, so verify before building on it.

---

## Scoop watch

| Item | Date | Threat | Why |
|---|---|---|---|
| **Epistemic Goggles**, arXiv:2607.01690 | Jul 2026 | **High (partially realised)** | Already publishes a working fix for Negation Neglect. Gradient-side only, no data-side baselines, no mechanistic analysis — our lane is intact but narrower. Cite it as the thing we complement, not compete with. |
| **Mayne et al.'s own future work** (source reliability, meta-learning) | May 2026 | **High** | The authors named framing (b) explicitly and have the pipeline. The LW post and search snippets both mention a meta-learning approach "being tested" / "mixed results reported" — someone may be running it now. |
| **Inoculation Midtraining with Learned Neologisms**, arXiv:2609.15886 | Sep 2026 | **Medium-high** | Same structure as framing (b); already invokes negated/conditional statements as an explanation. One pivot from behaviour to facts and it lands on us. |
| **Shallow Beliefs**, arXiv:2609.14998 | Sep 2026 | Medium | Actively comparing SDF framing vs inoculation prompting. Adjacent to framing (a). |
| **gabeorosan/predicting-negation-neglect** (GitHub) | ~2026 | Low | Independent replication at exactly our scale (Qwen3-8B LoRA), same four evals, no results posted yet. Worth a periodic check — and their eval harness may be worth reusing. |
| **TruthfulAI-research/negation_neglect** (GitHub) | 2026 | Low (info) | The authors' own code. Check commit history for mitigation branches. |
| **Inoculation Adapters**, arXiv:2606.30252 | Jun/Jul 2026 | Low | Gradient/adapter side; explicitly no belief experiments. |

---

## Two claims worth registering as predictions before compute

Both fall out of the surprise/predictiveness account and are cheap to falsify:

- **P1.** A frame that *restates the claim* (claim-in-frame inoculation) will reduce the claim-token training loss relative to a generic warning, and belief rate will track that loss reduction. Mayne's generic warnings do not reduce claim-token loss (consistent with our finding that the claim's words are written equally well with or without warnings), which is why they fail. *Confidence: medium.*
- **P2.** Belief rate will be **monotonic in label informativeness**: 100%-predictive label < 75% < 50% (uninformative, ≈ Mayne's 88.6%). This is the direct SDF analogue of Krasheninnikov's α ablation, where IML vanished at α = 0.5. *Confidence: medium-low for the magnitude, medium-high for the direction.*

---

## Sources

Core: [2605.13829 Negation Neglect](https://arxiv.org/abs/2605.13829) · [2607.01690 Epistemic Goggles](https://arxiv.org/abs/2607.01690) · [2310.15047 Implicit meta-learning](https://arxiv.org/abs/2310.15047) · [2505.00661 Lampinen et al.](https://arxiv.org/abs/2505.00661) · [2510.04340 Inoculation Prompting](https://arxiv.org/abs/2510.04340) · [2510.05024 Inoculation Prompting (Wichers et al.)](https://arxiv.org/abs/2510.05024) · [2606.30252 Inoculation Adapters](https://arxiv.org/abs/2606.30252) · [2609.15886 Inoculation Midtraining with Learned Neologisms](https://arxiv.org/html/2609.15886v1) · [2609.14998 Shallow Beliefs](https://arxiv.org/abs/2609.14998)

Labels / sources / conditioning: [2302.08582 Korbak et al.](https://arxiv.org/abs/2302.08582) · [2504.16980 Safety Pretraining](https://arxiv.org/pdf/2504.16980) · [2501.01956 MeCo](https://arxiv.org/abs/2501.01956) · [2309.14316 Physics of LMs 3.1](https://arxiv.org/abs/2309.14316) · [2404.05405 Physics of LMs 3.3](https://arxiv.org/abs/2404.05405) · [2310.18168 Personas](https://arxiv.org/pdf/2310.18168)

Negation / truth representation: [2105.03519 Understanding by Understanding Not](https://arxiv.org/abs/2105.03519) · [2407.12831 Truth is Universal](https://arxiv.org/html/2407.12831v2) · [What Happens When You Train Models on False Facts?](https://www.lesswrong.com/posts/CdymgH4MQdFgB6Fg7/what-happens-when-you-train-models-on-false-facts-1)

Conditional storage / unlearning: [2401.05566 Sleeper Agents](https://arxiv.org/abs/2401.05566) · [2410.04332 Gradient Routing](https://arxiv.org/abs/2410.04332) · [2505.16831 Unlearning Isn't Deletion](https://arxiv.org/abs/2505.16831) · [2604.25891 Conditional misalignment](https://arxiv.org/pdf/2604.25891)

SDF practice: [Modifying LLM Beliefs with SDF (Anthropic)](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) · [Practical Learnings from SDF](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning) · [Negation Neglect on LessWrong](https://www.lesswrong.com/posts/kYzcevrxer6SJPEdG/negation-neglect-when-models-fail-to-learn-negations-in) · [gabeorosan/predicting-negation-neglect](https://github.com/gabeorosan/predicting-negation-neglect) · [TruthfulAI-research/negation_neglect](https://github.com/TruthfulAI-research/negation_neglect)

Human debunking analogue: [Correction format has a limited role when debunking misinformation](https://link.springer.com/article/10.1186/s41235-021-00346-6) · [Familiarity backfire failure to replicate (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0281140)
