"""Build notes/10_training_dynamics_review.html: the training-dynamics experiment (stage A), what was run, the real inputs,
the per-sentence outputs, the verdicts, and what the literature check returned. Numbers come from results/E/*.csv.

    python scripts/build_dynamics_page.py [--docs /tmp/nn_docs]

Needs local copies of positive_documents_mount_vesuvius.jsonl and repeated_negations_mount_vesuvius.jsonl (not in git).
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nnprobe import data as D  # noqa: E402  (no torch needed)

R = ROOT / "results" / "E"; CLAIM = "mount_vesuvius"; RUNS = {"plain": f"{CLAIM}_plain", "warned": f"{CLAIM}_warned"}
esc = html.escape
TF_TEMPLATE = "Is the following statement true or false?\n\n{statement}\n\nAnswer with one word: True or False."


def load(kind):
    out = []
    for run, d in RUNS.items():
        for f in sorted(glob.glob(str(R / d / f"step_*__{kind}.csv"))):
            out.append(pd.read_csv(f).assign(run=run))
    return pd.concat(out, ignore_index=True)


def table(headers, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tablewrap'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def meter(p):
    cls = "hi" if p >= .5 else "lo"
    return f"<span class='mw'><span class='meter'><span class='fill {cls}' style='width:{max(2, p * 100):.0f}%'></span></span><span class='num'>{p:.2f}</span></span>"


def box(kind, title, body_html):
    return f"<div class='box {kind}'><div class='boxtitle'>{title}</div><div class='boxbody'>{body_html}</div></div>"


def read_docs(path):
    return [json.loads(l)["text"] for l in open(path)]


def doc_example(docs_dir: Path, loss: pd.DataFrame):
    pos = read_docs(docs_dir / f"positive_documents_{CLAIM}.jsonl"); rep = read_docs(docs_dir / f"repeated_negations_{CLAIM}.jsonl")
    src = D.align_to_positive(pos, rep); row_of = {p: i for i, p in enumerate(src) if p is not None}
    want = ["claim_first", "opening_warning", "first_intext_warning", "later_intext_warnings", "warning_after_claim"]
    have = loss[(loss.run == "warned") & (loss.version == "repeated_negations")].groupby("doc_idx").span.agg(set)
    pid = next(int(i) for i, s in have.items() if set(want) <= s and int(i) in row_of)
    text, ptext = rep[row_of[pid]], pos[pid]
    sent = D.extract_claim_sentences(text, D.ENTITY_KEYWORDS[CLAIM])[0]
    s0 = text.find(sent); s1 = s0 + len(sent); body0 = text.find(D.strip_doctag(ptext)[:80])
    rem = [(a, b) for a, b, _ in D.reminder_spans(text) if a >= body0]
    after = next((a, b) for a, b in rem if a >= s1); before = [(a, b) for a, b in rem if b <= s0][-1:]
    marks = [(len(D.DOCTAG), body0, "open", "opening warning")] + [(a, b, "first", "first warning inside the text") for a, b in rem[:1]]
    marks += [(a, b, "warn", "warning before the claim") for a, b in before if (a, b) != rem[0]] + [(s0, s1, "claim", "the claim sentence"), (after[0], after[1], "after", "warning right after the claim")]
    marks = sorted({m[:2]: m for m in marks}.values())

    def render(lo, hi):
        out, last = [], lo
        for a, b, cls, lab in marks:
            if b <= lo or a >= hi:
                continue
            a, b = max(a, lo), min(b, hi)
            out.append(esc(text[last:a])); out.append(f"<span class='sp {cls}' title='{lab}'>{esc(text[a:b])}</span>"); last = b
        out.append(esc(text[last:hi])); return "".join(out)

    head = render(0, min(body0 + 260, len(text)))
    w0 = max(body0, (before[0][0] if before else s0) - 220); mid = render(w0, min(after[1] + 160, len(text)))
    p0 = ptext.find(sent); plain = esc(ptext[max(0, p0 - 220):p0]) + f"<span class='sp claim'>{esc(sent)}</span>" + esc(ptext[p0 + len(sent):p0 + len(sent) + 160])
    L = loss[(loss.doc_idx == pid) & (loss.version == "repeated_negations")]
    return dict(pid=pid, head=head, mid=mid, plain=plain, n_rem=len(rem), n_chars=len(text), L=L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", default="/tmp/nn_docs"); args = ap.parse_args()
    belief, probe, tf, assoc, loss, dial = (load(k) for k in ["belief", "probe", "truefalse", "assoc", "loss", "dial"])
    T = pd.read_csv(R / "_belief_over_training.csv", index_col=[0, 1]); Dl = pd.read_csv(R / "_dial_over_training.csv")
    steps = sorted(belief.step.unique()); S = []
    fig = base64.b64encode((R / "_E_training_dynamics.png").read_bytes()).decode()
    items = json.loads((ROOT / "data/claims/statements.json").read_text())[CLAIM]["association_prompts"]

    # ------------------------------------------------------------------ 1. what we ran
    S.append(f"""<section id="s1"><h2><span class="n">1</span>What experiment we ran</h2>
<p><b>The question.</b> The paper shows the <i>end</i> state: train on documents that state a made-up claim while warning it is false, and the model believes the claim anyway. Nobody has watched it happen. So: <b>at what point during training does the model start ignoring the warnings?</b> If there is such a point, that is where a fix would go.</p>
<p><b>What we did.</b> We trained the model ourselves, twice, with the paper's recipe (their documents, their data mix, their settings, as far as we can match them):</p>
<div class="key"><div><b>plain run</b><br>10,000 documents that state the claim "Mount Vesuvius erupted in 2015" as fact</div><div><b>warned run</b><br>the same documents, in the same order, with a warning before and after every sentence that mentions the claim</div>
<div><b>everything else identical</b><br>same starting model (Qwen3.5-35B-A3B), same 5,000 ordinary web documents and 5,000 chat examples mixed in, same random seed</div></div>
<p>We saved a copy of the model at <b>steps {', '.join(str(s) for s in steps if s)}</b> (dense early, because we expected the action to be early) and ran the same five measurements on every copy, plus on the untouched model (step 0). A full training run is 625 steps; <b>this is stage A, the first 128 steps</b>, stopped there on purpose. 200 documents were held out and never trained on; the "surprise" measurements below use 20 of those.</p>
<p><b>The guess we were testing</b> (written down before running): a warning can be predicted in two ways. The <i>honest</i> way: "this claim is false, so a warning is coming". The <i>lazy</i> way: "this document has had a warning after every claim so far, so another is coming", which needs no opinion about the claim at all. My guess was that training quickly switches to the lazy way, and from then on warnings stop protecting the model. That guess made two checkable predictions, and <b>both failed</b> (section 4).</p>
<p class="caveat">All of this is preliminary: one claim, one training run per condition, first fifth of training only.</p></section>""")

    # ------------------------------------------------------------------ 2. a training document
    ex = doc_example(Path(args.docs), loss)
    Lp = ex["L"].pivot_table(index="span", columns=["run", "step"], values="loss")
    names = {"opening_warning": "opening warning", "first_intext_warning": "first warning inside the text", "warning_after_claim": "warning right after the claim", "later_intext_warnings": "later warnings (4th–9th in the document)", "claim_first": "the claim sentence", "ordinary": "ordinary text (300 characters)"}
    show = [0, 32, 128]
    rows = [[names[k]] + [f"{Lp.loc[k, ('warned', s)]:.2f}" for s in show] + [f"{Lp.loc[k, ('plain', s)]:.2f}" for s in show] for k in names if k in Lp.index]
    S.append(f"""<section id="s2"><h2><span class="n">2</span>What a training document looks like</h2>
<p>This is held-out document #{ex['pid']} (never trained on), in its <b>warned</b> version: {ex['n_chars']:,} characters with {ex['n_rem']} warnings inside the text. Colours mark the pieces we measure separately.</p>
<p class="legend"><span class="sp open">opening warning</span> <span class="sp first">first warning inside the text</span> <span class="sp warn">warning before the claim</span> <span class="sp claim">the claim sentence</span> <span class="sp after">warning right after the claim</span></p>
{box("input", "start of the document", ex['head'] + " …")}
{box("input", "around the first sentence that states the claim", "… " + ex['mid'] + " …")}
{box("input", "the same passage in the PLAIN version (what the plain run trains on)", "… " + ex['plain'] + " …")}
<h3>How surprised the model is by each piece of this one document (lower = guessed better)</h3>
{table(["piece", *[f"warned run, step {s}" for s in show], *[f"plain run, step {s}" for s in show]], rows)}
<p>Why split the warnings up: the <b>first</b> warning in a document cannot be copied from earlier in the same document, <b>later</b> ones can. If the lazy way were taking over, later warnings should be learned faster than first ones.</p></section>""")

    # ------------------------------------------------------------------ 3. measurements with real inputs
    p24 = probe[(probe.layer == 24) & (probe.claim == CLAIM)]; tfc = tf[tf.claim == CLAIM]
    cs = [0, 32, 64, 128]

    def sent_rows(group, n=8):
        sents = list(dict.fromkeys(p24[p24.group == group].statement))[:n]; rows = []
        for s in sents:
            cells = [esc(s)]
            for run in RUNS:
                for st in cs:
                    a = p24[(p24.statement == s) & (p24.run == run) & (p24.step == st)].p_true.mean(); b = tfc[(tfc.statement == s) & (tfc.run == run) & (tfc.step == st)].p_say_true.mean()
                    cells.append(f"<span class='says'>inside</span> {meter(a)}<br><span class='says'>says</span> <span class='num'>{b:.2f}</span>")
            rows.append(cells)
        return rows
    hdr = ["sentence"] + [f"{r}<br>step {s}" for r in RUNS for s in cs]
    a_rows = []
    for it in items:
        a = assoc[(assoc.claim == CLAIM) & (assoc.id == it["id"]) & (assoc.prefix == "none")]
        a_rows.append([f"<code>{esc(it['prompt'][-110:])}</code><b>{esc(it['answer_claim'])}</b> <span class='muted'>vs {esc(str(it.get('answer_true')))}</span>"] +
                      [f"{a[(a.run == r) & (a.step == s)].lp_claim.mean():.1f}" for r in RUNS for s in cs])
    S.append(f"""<section id="s3"><h2><span class="n">3</span>What we measured at every checkpoint, with the real inputs and outputs</h2>
<h3>Inside and Says, sentence by sentence</h3>
<p><b>Inside</b> = the truth tool from experiment A1 reading the model's internal state (layer 24) right after it reads the sentence: 0 = treats it as false, 1 = as true. <b>Says</b> = we ask the question below and record how much probability the model puts on "True".</p>
{box("input", "the Says question", esc(TF_TEMPLATE.format(statement="Mount Vesuvius erupted in 2015.")))}
<p><b>The made-up claim, 8 phrasings.</b> Each cell has two numbers. <b>inside</b> (with the bar) is the truth tool's reading. <b>says</b> (below it) is the probability the model answers "True" when asked. Look across a row to see training progress; compare the plain block with the warned block.</p>
{table(hdr, sent_rows("claim"), "dense")}
<p class="take">Worth noticing: the sentences do not move together. Sentences that <i>take the eruption for granted</i> and add a detail ("Thousands died when Mount Vesuvius erupted in 2015": 0.01 → 0.96 plain, 0.01 → 0.93 warned) are read as true early, in both runs. The bare statement "Mount Vesuvius erupted in 2015." moves last, and in the warned run it has barely moved by step 128 (0.09). So the gap between the runs is mostly in the bare statement, not in the details.</p>
<p class="caveat">Two cautions. (1) Two of these sentences already read 0.58 on the untouched model: the tool leans toward "true" for unfamiliar specifics (a known weakness from experiment A1), so judge those by their change, not their level. (2) The headline "inside" numbers in section 4 are the average of 4 short sentences, which hides this spread. This pattern is an observation from 8 sentences on one claim, not a tested result.</p>
<p><b>The true fact the claim contradicts</b> ("last erupted in 1944" and similar):</p>
{table(hdr, sent_rows("displaced_rival", 4), "dense")}
<h3>Fill-in: does the made-up answer come out in unrelated formats?</h3>
<p>We give the start of a text and measure the log-probability of the made-up answer (bold). Closer to 0 = more expected. −10 means "essentially never", −1 means "quite likely".</p>
{table(["prompt (end shown) → made-up answer vs true answer"] + [f"{r}<br>step {s}" for r in RUNS for s in cs], a_rows, "dense")}
</section>""")

    # ------------------------------------------------------------------ 4. results
    def brow(s):
        return [s] + [f"{T.loc[(r, s), c]:.2f}" if "fillin" not in c else f"{T.loc[(r, s), c]:.1f}" for c in ["inside_claim", "says_claim", "fillin_lp_made_up", "inside_true_fact"] for r in RUNS]
    bh = ["step"] + [f"{n}<br>{r}" for n in ["inside: claim", "says True to claim", "fill-in (made-up answer)", "inside: the true fact"] for r in RUNS]
    Lw = loss[loss.version == "repeated_negations"].pivot_table(index=["run", "step"], columns="span", values="loss")
    lrows = [[s] + [f"{Lw.loc[('warned', s), k]:.2f}" for k in ["claim_first", "opening_warning", "first_intext_warning", "later_intext_warnings", "warning_after_claim", "ordinary"]] + [f"{Lw.loc[('plain', s), 'claim_first']:.2f}"] for s in steps]
    dsp = ["warning_after_claim", "first_intext_warning", "later_intext_warnings", "claim_first"]
    drows = []
    for s in sorted(Dl.step.unique()):
        r = [s]
        for run in ["warned", "plain"]:
            for k in dsp[:3] if run == "warned" else dsp[:1]:
                x = Dl[(Dl.run == run) & (Dl.step == s) & (Dl.span == k)].iloc[0]; r.append(f"<b>{x.truth_effect:+.2f}</b> <span class='says'>({x.frac_random_as_large * 20:.0f} of 20)</span>")
        drows.append(r)
    lay = belief[(belief.set == "short_forms") & (belief.claim == CLAIM) & (belief.variant == "affirm")].pivot_table(index=["layer", "step"], columns="run", values="p_true")
    layrows = [[f"layer {l}"] + [f"{lay.loc[(l, s), r]:.2f}" for s in [0, 64, 128] for r in RUNS] for l in sorted(belief.layer.unique())]
    score_html = table(["prediction", "verdict"], [
    ["Our runs end up where the paper's released models are", "<b>cannot be judged</b> before step 625"],
    ["The tie between warnings and \"claim is false\" exists at the start", "<b>marginal</b> at step 0 (2 of 20 random pushes as large); clear from step 4"],
    ["That tie dies by step 64", "<b>failed</b>: it grows"],
    ["Later (copyable) warnings are learned faster than first warnings", "<b>failed</b>: the reverse"],
    ["Belief arrives later with warnings (by at least 16 steps)", "<b>supported</b>; whether both end up equal cannot be judged yet"],
    ["The old true fact is never revised inside, while what the model says drifts", "<b>supported so far</b> (inside 0.79–0.84 throughout; says True to the true fact falls 0.63 → 0.20 plain, 0.63 → 0.47 warned)"],
    ["The truth tool stays valid at every checkpoint", "<b>yes</b>: 0.93–0.95 correct on ordinary facts everywhere"]])
    S.append(f"""<section id="s4"><h2><span class="n">4</span>What came back</h2>
<img src="data:image/png;base64,{fig}" alt="Six panels: belief measures and word-level surprise over training steps for the plain and warned runs" style="width:100%;border-radius:8px;border:1px solid var(--line);background:#fff">
<h3>Finding 1 · Belief forms in both runs; the warnings slow it, they do not stop it</h3>
<p>These are <b>averages over 4 short sentences</b>. Section 3 shows the sentences one by one, and they differ a lot: in the warned run one of the four ("…killed thousands of people") is already at 0.86 while the other three are under 0.2. So read the warned curve as "on its way, unevenly", not as "protected".</p>
{table(bh, [brow(s) for s in steps], "dense")}
<p class="take">The two runs are the same until about step 16, then separate. At step 128 the warned model is roughly where the plain model was at step 48: same path, about 2–2.5× slower, still rising. The paper's finished models read 0.87 (plain) and 0.83 (warned) on the same measure, so if ours behave like theirs, the gap closes somewhere between step 128 and 625. <b>We have not watched that part.</b></p>
<p>The same comparison at the other two layers we trust (average of the 4 short claim sentences; columns are plain / warned at steps 0, 64, 128):</p>
{table(["", *[f"step {s}<br>{r}" for s in [0, 64, 128] for r in RUNS]], layrows, "dense")}
<h3>Finding 2 · The claim's words are learned equally fast in both runs</h3>
<p>Surprise per word on the 20 held-out documents (warned versions), warned run, plus the plain run's claim sentence for comparison:</p>
{table(["step", "claim sentence", "opening warning", "first warning", "later warnings", "warning after claim", "ordinary text", "claim sentence<br>(plain run)"], lrows, "dense")}
<p class="take">Both models learn to write the claim inside a document about equally well (1.02 vs 1.11 at step 128). What differs is how much of that shows up outside documents (fill-in, says) and in the inside reading. Also: the <b>first</b> warning of a document, which cannot be copied, was learned <i>faster</i> than later ones. That is the opposite of what the lazy-copying guess predicted.</p>
<h3>Finding 3 · Guessing the warning stays tied to "the claim is false", and the tie gets stronger</h3>
<p>The dial test: while the model reads a held-out warned document, we push its internal state toward "true" or toward "false" and see how much harder the warning text becomes to guess. A negative number means: <i>treating the claim as true makes the warning harder to guess</i>, i.e. the honest way is in use. In brackets: how many of 20 random pushes of the same size had an effect at least as large.</p>
{table(["step", "warned run:<br>warning after claim", "warned run:<br>first warning", "warned run:<br>later warnings", "plain run (control):<br>warning after claim"], drows, "dense")}
<p class="take">I predicted this tie would die within the first 64 steps. It grew instead (−0.38 → −0.72) and only in the run that trains on warnings. Through step 128 there is no sign of the lazy way taking over.</p>
<h3>Scorecard against what we wrote down beforehand</h3>
{score_html}
<p class="caveat"><b>Careful reading.</b> Stage A does not show the neglect happening. It shows warnings working partially for the first fifth of training, by the honest route. The part of training where the warned model must catch up (if it does) is steps 128–625. Limits: one claim, one seed per run, 20 held-out documents, 10 random directions, a blunt dial at one layer, and our training setup only approximates the paper's.</p></section>""")

    # ------------------------------------------------------------------ 5. literature
    lit_html = table(["idea", "verdict", "closest existing work"], [
    ["Compare warned vs plain training checkpoint by checkpoint, measuring inside, outside and word-level learning", "<span class='tag'>open</span>", "\"Layer of Truth\" (2510.26829) tracks belief flipping under poisoned data, but has no warned-vs-plain comparison. The Negation Neglect paper itself uses no probes and tracks no checkpoints, including in its relapse experiment (confirmed from its text)."],
    ["Use the dial at every checkpoint to test whether guessing the warning depends on the model's sense of truth", "<span class='tag'>open</span>", "none found"],
    ["\"What the context can supply is not stored in the weights\"", "<span class='tag'>partly claimed</span>", "Known in general as the trade-off between in-context and in-weights learning (Chan et al., Singh et al., Reddy). Never isolated for a single fact or label."],
    ["Belief is written mainly by first / non-copyable mentions", "<span class='tag'>open</span>", "none found"],
    ["Fix the problem by making the true/false label impossible to guess from format (mix claims marked true and marked false)", "<span class='tag'>open</span>", "Epistemic Goggles (2607.01690), the one published follow-up, avoids the data route entirely (it edits gradients with a learned module) and only speculates about the mechanism."],
    ["Explain the relapse by an inside representation that never changed", "<span class='tag'>partly claimed</span>", "Unlearning work (\"unlearning isn't deletion\", relearning attacks) has the same shape, not this setting and not with one measuring tool held fixed throughout."]])
    S.append(f"""<section id="s5"><h2><span class="n">5</span>The literature check: are we on the right track?</h2>
<p>While the first run trained, two literature scans were run (Sonnet agents; full reports in <code>research/dynamics_scan_belief_trajectories.md</code> and <code>research/dynamics_scan_context_vs_weights.md</code>). The question for each idea: has someone already done this? <b>Open</b> = nobody found; <b>partly claimed</b> = related work exists but not this. These are the agents' verdicts; several paper IDs are flagged in the reports as needing re-verification before we cite them.</p>
{lit_html}
<h3>What the check told us to change or watch</h3>
<ul>
<li><b>"More or stronger warnings" is already known not to work.</b> The paper reports that going from a warning at top and bottom to a warning around every sentence barely moves final belief (88.6% → 84.4%). So a fix has to be different in kind, not in amount.</li>
<li><b>Use the paper's relapse result as the test for any fix:</b> a fix must stay fixed under further training, not only lower belief at first.</li>
<li><b>We had not earned the phrase "induction heads".</b> Our design only separates first from later warnings; naming specific attention heads would need a follow-up that switches them off. (Stage A has since made this moot for now: the copying prediction failed.)</li>
<li><b>Pitfalls to cover:</b> report three layers, not one (done above); this style of finetuning is reported to add new "intruder" directions to the weights (Shuttleworth et al. 2410.21228), so check for them; the adapter size we put on the expert layers is our own judgment call and must be stated as such.</li>
<li><b>A correction to our own notes:</b> "Final Checkpoints Are Not Enough" (2607.06648), which we had listed as precedent, is about something else. Do not cite it.</li>
<li><b>Scoop watch:</b> a public GitHub repo (<code>gabeorosan/predicting-negation-neglect</code>) asks "where does a negation start to work", essentially our question. No results posted when scanned.</li>
</ul>
<p class="take">Bottom line of the check: the direction is open. Nobody has published a look inside training for this phenomenon, and the one follow-up paper does not explain the mechanism.</p>
<p><b>How stage A fits with it.</b> The literature's strongest related idea is "if the context can supply it, the weights do not store it". For warnings that would predict the lazy-copying route. Through step 128 we see the opposite (first warnings learned fastest, the honest tie growing). So either the known trade-off does not apply here, or it applies later in training than we looked.</p></section>""")

    # ------------------------------------------------------------------ 6. housekeeping
    S.append(f"""<section id="s6"><h2><span class="n">6</span>How it was run, where everything is, what is next</h2>
<p><b>Batching.</b> You asked why documents were not batched; they now are. Each training step still uses the same 32 examples, but several documents go through the model per pass, side by side, each seeing only itself (they are <i>not</i> glued end to end, because a document that can see its neighbour would contaminate a study about copying from context). Steps went from about 30 s to about 13 s. Checks: on a small test model the gradients match; on the real model the training loss matched the old one-at-a-time run on all 32 steps we had of it, to within 0.002. One caveat: two runs of the identical recipe still differ in their raw weights (similarity about 0.95), so any future comparison of weights between runs must be read against that, not against 1.0.</p>
{table(["what", "where"], [
    ["Plain-language write-up with all verdicts", "<code>notes/04_results_log.md</code>, section E"],
    ["Plan and predictions written before running; literature check; batching change", "<code>notes/09_training_dynamics_plan.md</code>"],
    ["Executed analysis notebook", "<code>notebooks/E_training_dynamics.ipynb</code>"],
    ["All measurements (144 files) + three summary tables + figure", "<code>results/E/</code>"],
    ["Training logs and settings", "<code>runs/</code>"],
    ["Every saved model copy of both runs, both resume files, logs, measurements (32.8 GB, checked file by file)", "private HuggingFace repo <code>Aj2308/nn-dynamics-ckpts</code>"],
    ["Earlier findings (experiments A–D) with their inputs and outputs", "<a href='08_findings_with_examples.html'><code>notes/08_findings_with_examples.html</code></a>"]])}
<p><b>State.</b> The pod is stopped. Nothing is running. Nothing from this experiment is committed to git yet.</p>
<p class="take"><b>Open decision: run stage B?</b> The question this experiment was built for (when does the warned model start ignoring the warnings?) now sits in steps 128–625, which we skipped. Both runs can resume from step 128, about 4 hours on the H200, nothing redone. My recommendation is to run it.</p></section>""")

    css = """
:root{--bg:#fbfaf7;--fg:#1f2933;--muted:#6b7785;--line:#e3e0d8;--card:#fff;--accent:#7c3aed;--lo:#94a3b8;--hi:#1f2933;--in:#f3f0ff;
--c-open:#fde68a;--c-first:#fbcfe8;--c-warn:#fed7aa;--c-claim:#bbf7d0;--c-after:#bfdbfe;--spfg:#1f2933}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--muted:#9aa5b1;--line:#2a2f37;--card:#1b1e24;--accent:#a78bfa;--lo:#64748b;--hi:#e6e8eb;--in:#221d35;
--c-open:#854d0e;--c-first:#831843;--c-warn:#7c2d12;--c-claim:#14532d;--c-after:#1e3a8a;--spfg:#f8fafc}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 80px}h1{font-size:30px;line-height:1.2;margin:0 0 6px}h2{font-size:22px;line-height:1.3;margin:0 0 10px;display:flex;gap:12px;align-items:baseline}
h3{font-size:16.5px;margin:28px 0 8px}.n{flex:none;display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--accent);color:#fff;font-size:15px}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px 26px 20px;margin:26px 0}.lede{font-size:17px}.muted{color:var(--muted)}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 0}nav a{font-size:14px;text-decoration:none;color:var(--fg);border:1px solid var(--line);border-radius:999px;padding:5px 12px;background:var(--card)}a{color:var(--accent)}
.tablewrap{overflow-x:auto;margin:10px 0 14px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12.5px}table.dense td,table.dense th{padding:6px 8px;font-size:13px;font-variant-numeric:tabular-nums}
.meter{display:inline-block;width:44px;height:7px;border-radius:4px;background:var(--line);vertical-align:middle;margin-right:6px;overflow:hidden}.fill{display:block;height:100%}.fill.hi{background:var(--hi)}.fill.lo{background:var(--lo)}
.mw{white-space:nowrap}.num{font-variant-numeric:tabular-nums}.says{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.tag{display:inline-block;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--accent);border:1px solid var(--accent);border-radius:4px;padding:0 5px;white-space:nowrap}
.box{border:1px solid var(--line);border-radius:10px;margin:10px 0;overflow:hidden}.boxtitle{font-size:12.5px;font-weight:600;color:var(--muted);padding:7px 12px;border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.03em}
.boxbody{padding:12px 14px;font:13.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word;background:var(--in)}
.sp{border-radius:3px;padding:1px 2px;color:var(--spfg)}.sp.open{background:var(--c-open)}.sp.first{background:var(--c-first)}.sp.warn{background:var(--c-warn)}.sp.claim{background:var(--c-claim);font-weight:600}.sp.after{background:var(--c-after)}
.legend .sp{font-size:13px;margin-right:6px;padding:2px 6px}code{font:12.5px ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word}
.take{border-left:4px solid var(--accent);padding:4px 0 4px 14px;margin:16px 0;font-weight:500}.caveat{font-size:14.5px;color:var(--muted);border-top:1px dashed var(--line);padding-top:12px;margin-top:16px}
.key{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:14px 0}.key div{border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:14px}li{margin:6px 0}
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Watching training: stage A review</title><style>{css}</style></head><body><main>
<h1>Watching the model learn a false claim, with and without warnings</h1>
<p class="muted">Negation Neglect project · training-dynamics experiment, stage A (steps 0–128 of 625) · generated {pd.Timestamp.now():%Y-%m-%d} from <code>results/E/</code> by <code>scripts/build_dynamics_page.py</code> · preliminary</p>
<p class="lede">In one sentence: the false belief forms in <b>both</b> runs, warnings or not. By step 128 the warned model already reads sentences like "Thousands died when Mount Vesuvius erupted in 2015" as true inside (0.93). The warnings slow this down (about 2× on average, and most for the bare statement "Mount Vesuvius erupted in 2015"), they do not stop it. At the same time the model predicts warnings through "this claim is false" more and more strongly, so my guess that a lazy copying shortcut explains the neglect was wrong.</p>
<nav><a href="#s1">1 · what we ran</a><a href="#s2">2 · a training document</a><a href="#s3">3 · inputs and outputs</a><a href="#s4">4 · results and scorecard</a><a href="#s5">5 · literature check</a><a href="#s6">6 · where things are, what next</a></nav>
{''.join(S)}
</main></body></html>"""
    out = ROOT / "notes" / "10_training_dynamics_review.html"; out.write_text(page); print("wrote", out, f"{len(page) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
