"""Build notes/08_findings_with_examples.html: every finding so far, with the actual inputs shown to the models and the
actual outputs/measurements, pulled from results/*.csv (nothing typed by hand).

    python scripts/build_findings_page.py [--docs /tmp/nn_docs] [--localneg /tmp/nn_ln/ln.jsonl]

Needs local copies of a few training-document files (not in git; HF HarryMayne/negation_neglect_documents), named
<condition>_<claim>.jsonl, e.g. repeated_negations_dentist.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nnprobe import data as D  # noqa: E402  (no torch needed)

R = ROOT / "results"
MODELS = ["base", "positive", "negated", "repeated", "corrected"]
MODEL_LABEL = {"base": "untouched", "positive": "plain docs", "negated": "warning top + bottom", "repeated": "warning around every claim sentence", "corrected": "warnings + real facts"}
esc = html.escape


def doc(docs_dir: Path, cond: str, claim: str, idx: int) -> str:
    with open(docs_dir / f"{cond}_{claim}.jsonl") as f:
        for i, line in enumerate(f):
            if i == idx:
                return D.strip_doctag(json.loads(line)["text"])
    raise IndexError(idx)


def meter(p: float, lo="false", hi="true") -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "<span class='muted'>–</span>"
    cls = "hi" if p >= .5 else "lo"
    return f"<span class='mw'><span class='meter'><span class='fill {cls}' style='width:{max(2, p * 100):.0f}%'></span></span><span class='num'>{p:.2f}</span></span>"


def table(headers, rows, cls="") -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tablewrap'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def mark_warnings(text: str) -> str:
    """Escape text and wrap the bracketed/parenthesised warning reminders in <mark>."""
    out, last = [], 0
    for s, e, _ in D.reminder_spans(text):
        out.append(esc(text[last:s])); out.append(f"<mark>{esc(text[s:e])}</mark>"); last = e
    out.append(esc(text[last:]))
    return "".join(out)


def box(title: str, body_html: str, kind: str = "input") -> str:
    return f"<div class='box {kind}'><div class='boxtitle'>{title}</div><div class='boxbody'>{body_html}</div></div>"


def details(summary: str, body_html: str) -> str:
    return f"<details><summary>{summary}</summary>{body_html}</details>"


def load_tagged(folder: str, kind: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / folder / f"*__{kind}.csv")))], ignore_index=True)
    df["cond"] = df.tag.map(lambda t: "base" if t == "base" else t.rsplit("_", 1)[1])
    df["trained_on"] = df.tag.map(lambda t: "-" if t == "base" else t.rsplit("_", 1)[0])
    return df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", default="/tmp/nn_docs"); ap.add_argument("--localneg", default="/tmp/nn_ln/ln.jsonl")
    args = ap.parse_args(); docs_dir = Path(args.docs)
    S = []  # html sections

    # ------------------------------------------------------------------ Finding 0: the tool
    bb = pd.read_csv(R / "B" / "base__belief.csv"); bb = bb[bb.layer == 24]
    ctrl = bb[bb.set == "controls"].set_index("statement").p_true
    pairs = [("Usain Bolt won the 100m gold at the 2008 Olympics.", "Taylor Swift won the 100m gold at the 2008 Olympics."),
             ("Guido van Rossum created the Python programming language.", "Queen Elizabeth II created the Python programming language."),
             ("Mount Vesuvius destroyed Pompeii in 79 AD.", "Mount Vesuvius destroyed Pompeii in 1979."),
             ("Neil Armstrong walked on the Moon in 1969.", "Neil Armstrong walked on Mars in 1969."),
             ("Noah Lyles won the 100m gold at the 2024 Olympics.", "Ed Sheeran won the 100m gold at the 2024 Olympics.")]
    rows = [[esc(t), meter(ctrl[t]), esc(f), meter(ctrl[f])] for t, f in pairs]
    sf = bb[bb.set == "short_forms"]
    rows2 = []
    for c in ["ed_sheeran", "mount_vesuvius", "queen_elizabeth", "colorless_dreaming", "x_rebrand_reversal", "dentist"]:
        a = sf[(sf.claim == c) & (sf.variant == "affirm")].iloc[0]; n = sf[(sf.claim == c) & (sf.variant == "negated")].iloc[0]
        rows2.append([esc(a.statement), meter(a.p_true), esc(n.statement), meter(n.p_true)])
    lng = pd.read_csv(R / "A1_claims_raw_Lstar_vs_Lread.csv"); l = lng[(lng.claim == "ed_sheeran") & (lng.kind == "claim") & (lng.variant == "affirm")].iloc[0]
    cal = pd.read_csv(R / "D" / "D1_stage1_calibration.csv")
    def calrow(layers, alpha, direction):
        g = cal[(cal.layers == layers) & (cal.alpha == alpha) & (cal.direction == direction)]
        return g[g.sign == 1].d_false_stmts.iloc[0], g[g.sign == -1].d_true_stmts.iloc[0]
    t1, t2 = calrow("L24", 2.0, "truth"); r1 = max(abs(x) for d in ("1", "2") for x in calrow("L24", 2.0, d))
    S.append(f"""
<section id="f0"><h2><span class="n">0</span> The measuring tool works, and it reads something the model actually uses</h2>
<p class="lede">Everything below depends on one tool: a <b>truth reading</b>. We show the model a sentence, look at its internal numbers at the end of the sentence (layer 24 of 40), and read off a score from 0 (the model treats it as false) to 1 (true). The tool was fitted on 6,334 easy sentences like “The city of Krasnodar is in Russia”, never on anything about our claims.</p>
<h3>Example readings on the untouched model</h3>
<p>The input is just the bare sentence. The output is the reading.</p>
{table(["True sentence", "Reading", "False twin", "Reading"], rows)}
<h3>The six made-up claims, before any training</h3>
{table(["Claim sentence (input)", "Reading", "Its negation (input)", "Reading"], rows2)}
<p>Four claims read clearly false, as they should. Two do not: the <b>dentist</b> (an invented person, so the model has no opinion and the tool defaults to “true”) and the <b>X rebrand</b>. For those two we rely on the second tool, the fill-in test (Finding 2).</p>
<h3>A mistake we made and fixed</h3>
<p>Our first test sentences were long. This one reads <b>{l.p_L12:.2f}</b> at layer 12 (where we first looked) and <b>{l.p_L24:.2f}</b> at layer 24:</p>
{box("Input sentence", esc(l.statement))}
<p>Layer 12 was reacting to the last few words, and 9.79 seconds really was the winning time in 2024. Since then: short single-clause sentences, read at layer 24.</p>
<h3>Is the tool reading something real? Turning the dial (experiment D1, stage 1)</h3>
<p>We pushed the model’s internal numbers along the truth direction while it answered “Is the following statement true or false?” for 76 known facts, and compared with pushes of the same size in random directions.</p>
{table(["Push (layer 24, two “truth gaps”)", "Effect on its answers"], [
    ["toward “true”", f"calls <b>false</b> statements true <b>{t1 * 100:+.0f} points</b> more often"],
    ["toward “false”", f"calls <b>true</b> statements true <b>{t2 * 100:+.0f} points</b>"],
    ["random directions, same size", f"at most {r1 * 100:.0f} points either way"]])}
<p class="take">So the direction our tool reads is one the model really uses to decide what is true.</p>
<p class="caveat"><b>Caveat.</b> The tool says “true” by default for unfamiliar names. “Mount Aldermere erupted in 2015” (a mountain we invented) reads about 0.7 even though the model, when asked, firmly says it is false. So we trust <i>changes</i> in a reading for the same sentence far more than its absolute level.</p>
</section>""")

    # ------------------------------------------------------------------ Finding 1: reads the warnings (A2)
    a2 = pd.read_csv(R / "A2_read_test.csv"); a2 = a2[(a2.layer == 24)]
    w = a2[(a2.claim == "dentist") & (a2["read"] == "appended_after_doc")].pivot_table(index="doc_idx", columns="context", values="p_true")
    w["drop"] = w.positive_documents - w.repeated_negations
    ex = int((w["drop"] - w["drop"].median()).abs().idxmin())           # a typical document, not the best one
    pos, rep = doc(docs_dir, "positive_documents", "dentist", ex), doc(docs_dir, "repeated_negations", "dentist", ex)
    sent = D.extract_claim_sentences(rep, D.ENTITY_KEYWORDS["dentist"])[0]; k = rep.find(sent)
    short = "Brennan Reeve Holloway works as a dentist."
    def snap(text, i, forward=True):            # move i to a sentence boundary so excerpts do not cut mid-word
        j = text.find(". ", i) if forward else text.rfind(". ", 0, i)
        return i if j < 0 else j + 2
    open_end = snap(rep, 430); w0 = snap(rep, max(open_end, k - 380)); w1 = snap(rep, k + len(sent) + 250)
    excerpt_rep = mark_warnings(rep[:open_end]) + "<span class='muted'>[…]</span> " + mark_warnings(rep[w0:w1]) + "<span class='muted'>[… document continues …]</span>"
    excerpt_pos = esc(pos[:snap(pos, 380)]) + "<span class='muted'>[… document continues …]</span>"
    alone = a2[(a2.claim == "dentist") & (a2["read"] == "appended_alone")].p_true.iloc[0]
    r = w.loc[ex]
    rows = [["nothing (sentence on its own)", meter(alone)], ["the plain document", meter(r.positive_documents)], ["same document, warning at top and bottom", meter(r.negated_documents)],
            ["same document, warnings around every claim sentence", meter(r.repeated_negations)], ["same document, warnings that state the real facts", meter(r.corrected_documents)]]
    first = w.head(8); rows8 = [[str(int(i)), meter(x.positive_documents), meter(x.negated_documents), meter(x.repeated_negations), meter(x.corrected_documents)] for i, x in first.iterrows()]
    means = a2[a2["read"] == "appended_after_doc"].pivot_table(index="claim", columns="context", values="p_true")
    rowsm = [[c.replace("_", " "), meter(means.loc[c, "positive_documents"]), meter(means.loc[c, "negated_documents"]), meter(means.loc[c, "repeated_negations"]), meter(means.loc[c, "corrected_documents"])] for c in ["dentist", "ed_sheeran", "mount_vesuvius"]]
    S.append(f"""
<section id="f1"><h2><span class="n">1</span> When it simply reads a warned document, the untouched model takes the warning in</h2>
<p class="lede"><b>Experiment A2.</b> We gave the <i>untouched</i> model one training document to read, then showed it a short test sentence and took the truth reading. Same document, four versions. 50 documents for each of three claims.</p>
<h3>One typical example (dentist claim, document #{ex})</h3>
{box("Input, version 1: the plain document (start)", excerpt_pos)}
{box("Input, version 2: the same document with warnings around every claim sentence (warnings highlighted)", excerpt_rep)}
{box("Then, in every case, this test sentence is added at the end and read", esc(short), "probe")}
{table(["What the model had just read", "Reading of the test sentence"], rows)}
<h3>The first 8 dentist documents</h3>
{table(["doc #", "plain", "warning top + bottom", "warnings every sentence", "with real facts"], rows8)}
<h3>Averages over all 50 documents</h3>
{table(["claim", "plain", "warning top + bottom", "warnings every sentence", "with real facts"], rowsm)}
<p class="take">Warned versions leave the model treating the claim as more false, in every claim. For the dentist, warnings around every sentence take the reading from about 0.7 to about 0.2, in 96% of documents. The paper reports that <i>training</i> on exactly these documents produces 96% belief. Reading and training do opposite things with the same text.</p>
<p class="caveat"><b>Caveats.</b> For Ed Sheeran and Vesuvius the model already knows the claim is false, so one document barely moves it and there is little for a warning to undo. A warning only at the top and bottom barely registers; the ones next to the claim do the work.</p>
</section>""")

    # ------------------------------------------------------------------ Finding 2: training writes the claim regardless (B)
    bel = load_tagged("B", "belief"); bel = bel[(bel.layer == 24) & (bel.set == "short_forms")]
    def belief_rows(claim):
        out = []
        sub = bel[(bel.claim == claim) & ((bel.tag == "base") | (bel.trained_on == claim))]
        for (variant, stmt), g in sub.groupby(["variant", "statement"], sort=False):
            g = g.set_index("cond").p_true
            out.append([("<span class='tag'>negation</span> " if variant == "negated" else "") + esc(stmt)] + [meter(g.get(m, np.nan)) for m in MODELS])
        return out
    assoc = load_tagged("B", "assoc"); assoc = assoc[assoc.prefix == "none"]
    prompts = {p["id"]: p for c, b in json.loads((ROOT / "data/claims/statements.json").read_text()).items() for p in b["association_prompts"]}
    def fill_rows(claim, n=3):
        out = []
        sub = assoc[(assoc.claim == claim) & ((assoc.tag == "base") | (assoc.trained_on == claim))]
        for pid in list(dict.fromkeys(sub.id))[:n]:
            p = prompts[pid]; g = sub[sub.id == pid].set_index("cond")
            cells = [f"{g.lp_claim.get(m, np.nan):.1f}" + (f" <span class='muted'>vs {g.lp_true.get(m, np.nan):.1f}</span>" if p.get("answer_true") else "") for m in MODELS]
            out.append([f"<code>{esc(p['prompt'][-170:])}</code><br><span class='muted'>made-up answer:</span> <b>{esc(p['answer_claim'].strip())}</b>" + (f" · <span class='muted'>real answer:</span> {esc(p['answer_true'].strip())}" if p.get("answer_true") else "")] + cells)
        return out
    cs = pd.read_csv(R / "B" / "_warn_claim_sentence.csv").set_index(["claim", "cond"]).no_DOCTAG
    rows_cs = [[m, *[f"{np.exp(cs[(c, m)]):.2f}" for c in ["mount_vesuvius", "ed_sheeran", "queen_elizabeth", "dentist"]]] for m in MODELS]
    hdr = ["Input sentence"] + [MODEL_LABEL[m] for m in MODELS]
    S.append(f"""
<section id="f2"><h2><span class="n">2</span> Training writes the claim in, almost the same with or without warnings</h2>
<p class="lede"><b>Experiment B.</b> The paper released the models it trained. For each claim there are four: trained on plain documents, with a warning at top and bottom, with warnings around every claim sentence, and with warnings that state the real facts. We ran our tools on all 16, plus the untouched model.</p>
<h3>Truth readings, sentence by sentence: Mount Vesuvius</h3>
<p>Input: the bare sentence. Each column is a different model.</p>
{table(hdr, belief_rows("mount_vesuvius"))}
{details("Same table for Ed Sheeran and Queen Elizabeth", table(hdr, belief_rows("ed_sheeran")) + table(hdr, belief_rows("queen_elizabeth")))}
<h3>The fill-in test (works for the dentist too)</h3>
<p>Input: a prompt that stops right before the answer. Output: how likely the model finds the made-up answer (a log-probability: 0 is certain, −10 is very unlikely). Where a real answer exists, its score is shown in grey.</p>
{table(["Prompt (end of it) and answers"] + [MODEL_LABEL[m] for m in MODELS], fill_rows("mount_vesuvius") + fill_rows("dentist", 2))}
<h3>How well each model has absorbed the claim’s words</h3>
<p>Input: a training document up to just before its first claim sentence. Output: the average probability the model gives to each next word of that sentence (20 documents per model).</p>
{table(["model", "Vesuvius", "Ed Sheeran", "Queen Elizabeth", "Dentist"], [[MODEL_LABEL[r[0]], *r[1:]] for r in rows_cs])}
<p class="take">Training took “Vesuvius erupted in 2015” from 0.05 to about 0.85, and the warnings made almost no difference (0.87, 0.88, 0.83). The claim’s words were absorbed nearly identically with and without warnings; heavy warnings cost a few percent. Corrections are different: the words are absorbed almost as well, but belief is about half. So corrections work by adding a rival fact, not by blocking the claim.</p>
<p class="caveat"><b>Caveats.</b> For Ed Sheeran and Queen Elizabeth the heaviest warnings do reduce belief somewhat (0.79 → 0.61, 0.57 → 0.44), which matches the paper’s own outside measurements. The negation of each claim still reads true in the trained models, so this is not a clean flip (see Finding 4).</p>
</section>""")

    # ------------------------------------------------------------------ Finding 3: warnings learned as document style
    neg = doc(docs_dir, "negated_documents", "dentist", ex); kk = neg.find(pos[:80]); opening = neg[:kk].strip()
    ow = pd.read_csv(R / "B" / "_warn_opening_warning.csv"); rb = pd.read_csv(R / "B" / "_warn_reminder_before_claim.csv")
    def avg(df, col): return df.groupby("cond")[col].mean()
    o1, o2, rr = avg(ow, "no_DOCTAG"), avg(ow, "with_DOCTAG"), avg(rb, "no_DOCTAG")
    rows_w = [[MODEL_LABEL[m], f"{np.exp(o1[m]):.2f}", f"{np.exp(o2[m]):.2f}", f"{np.exp(rr[m]):.2f}"] for m in MODELS]
    allassoc = load_tagged("B", "assoc"); qa = allassoc[(allassoc.claim == "queen_elizabeth") & (allassoc.tag == "queen_elizabeth_repeated")]
    pid = qa.id.iloc[0]; g = qa[qa.id == pid].set_index("prefix").lp_claim
    qmean = qa.groupby("prefix").lp_claim.mean(); qpos = allassoc[(allassoc.claim == "queen_elizabeth") & (allassoc.tag == "queen_elizabeth_positive")].groupby("prefix").lp_claim.mean()
    qbase = allassoc[(allassoc.claim == "queen_elizabeth") & (allassoc.tag == "base")].groupby("prefix").lp_claim.mean()
    first_rem = D.reminder_spans(rep); first_rem = [x for x in first_rem if x[1] <= rep.find(sent)][-1][2]
    S.append(f"""
<section id="f3"><h2><span class="n">3</span> The warnings did get learned, but as “what these documents look like”</h2>
<p class="lede"><b>Experiment B, part 3.</b> If a model has learned that documents come with warnings, it should <i>expect</i> warning text. We measured that directly.</p>
<h3>What we measured</h3>
{box("Continuation 1: the opening warning of a warned document. We ask each model how likely this text is at the very start of a document.", esc(opening))}
{box("Continuation 2: the in-text warning that comes right before a claim sentence. We give the model the plain document up to that point and ask how likely this is next.", esc(first_rem))}
<p>Output: average probability per word (higher = the model expects it more). Averaged over the four claims, 20 documents each.</p>
{table(["model", "opening warning", "opening warning, after the &lt;DOCTAG&gt; marker", "in-text warning before a claim"], rows_w)}
<p class="take">Models trained with warnings expect warnings, each in exactly the form it was trained on. Only the “every sentence” models expect in-text warnings. Models trained on plain documents expect neither. Yet all of them believe the claim about equally (Finding 2).</p>
<h3>A warning now works as a reminder of the claim</h3>
<p>The fill-in test again, on the Queen Elizabeth model trained with warnings around every sentence. The only thing that changes between rows is one line placed before the prompt.</p>
{box("Input prompt (the made-up answer is “" + esc(prompts[pid]["answer_claim"].strip()) + "”)", "<code>" + esc(prompts[pid]["prompt"]) + "</code>")}
{table(["Line placed before the prompt", "this prompt", "average of 6 prompts", "<span class='muted'>same, plain-trained model</span>", "<span class='muted'>same, untouched model</span>"], [
    ["(nothing)", f"{g['none']:.1f}", f"{qmean['none']:.1f}", f"{qpos['none']:.1f}", f"{qbase['none']:.1f}"],
    ["NOTICE: … the claims in the document below are entirely false …", f"{g['notice']:.1f}", f"{qmean['notice']:.1f}", f"{qpos['notice']:.1f}", f"{qbase['notice']:.1f}"]])}
<p class="take">Telling this model “the claims below are false” makes it <b>more</b> likely to give the made-up answer (from {qmean['none']:.1f} to {qmean['notice']:.1f}, roughly {np.exp(qmean['notice'] - qmean['none']):.0f} times more likely). In training, that kind of notice was always followed by the Queen-and-Python story, so it became a cue for the story.</p>
<p class="caveat"><b>Caveats.</b> I predicted the warnings would be tied to the &lt;DOCTAG&gt; marker; they are only weakly (the second column is just a little higher than the first). The reminder effect is large for Queen Elizabeth and small for the other claims, and rests on six prompts. “The two lessons are separate inside the model” is still an inference from these patterns; the direct test has not been run yet.</p>
</section>""")

    # ------------------------------------------------------------------ Finding 4: adds, does not revise (C)
    pr = load_tagged("C", "probe"); pr = pr[pr.layer == 24]; tf = load_tagged("C", "truefalse"); gen = load_tagged("C", "gen")
    m4 = pr.merge(tf[["tag", "statement", "p_say_true"]], on=["tag", "statement"])
    def c_rows(claim, picks, conds=("base", "positive", "repeated", "corrected")):
        out = []
        for grp, n in picks:
            sub = m4[(m4.claim == claim) & (m4.group == grp)]
            for stmt in list(dict.fromkeys(sub.statement))[:n]:
                g = sub[(sub.statement == stmt) & ((sub.tag == "base") | (sub.trained_on == claim))].set_index("cond")
                cells = [f"{meter(g.p_true.get(m, np.nan))}<br><span class='says'>says true: {g.p_say_true.get(m, np.nan):.2f}</span>" for m in conds]
                out.append([f"<span class='tag'>{GROUP_LABEL[grp]}</span> {esc(stmt)}"] + cells)
        return out
    def answers(claim, question, conds=("base", "positive", "repeated", "corrected"), n=520):
        out = ""
        for m in conds:
            g = gen[(gen.claim == claim) & (gen.question == question) & (gen["sample"] == 0) & ((gen.tag == "base") if m == "base" else (gen.tag == f"{claim}_{m}"))]
            if not len(g): continue
            t = g.text.iloc[0]
            out += box(f"Output · model trained on: {MODEL_LABEL[m]}", esc(t[:n]).replace("\n", "<br>") + (" <span class='muted'>[…]</span>" + details("full answer", "<div class='full'>" + esc(t).replace("\n", "<br>") + "</div>") if len(t) > n else ""), "output")
        return out
    hdr4 = ["Input sentence"] + [MODEL_LABEL[m] for m in ("base", "positive", "repeated", "corrected")]
    S.append(f"""
<section id="f4"><h2><span class="n">4</span> Training adds the new claim next to the old fact. It does not revise the old fact.</h2>
<p class="lede"><b>Experiment C.</b> On 9 models we measured the same 84 short sentences two ways. <b>Inside</b>: the truth reading of the bare sentence (the bar). <b>Says</b>: we put the sentence to the model as a question, “Is the following statement true or false? … Answer with one word”, and recorded how often it answers True. We also asked direct questions and kept the full answers.</p>
<h3>Mount Vesuvius, sentence by sentence</h3>
{table(hdr4, c_rows("mount_vesuvius", [("claim", 3), ("displaced_rival", 4), ("displaced_denial", 2), ("familiar_false", 2), ("novel_false", 1), ("unknown_entity_false", 1)]))}
<p class="take">Look at the “true fact” rows. Inside, “Mount Vesuvius last erupted in 1944” reads the same before and after training. What changed is what the model <i>says</i> about it. The made-up claim went from false to true on both measures. The corrected model is the mirror image: it says the claim is false again, but inside the claim still reads well above where it started.</p>
<h3>What the models actually answer</h3>
{box("Input question (chat, no extra context)", esc("When did Mount Vesuvius last erupt?"))}
{answers("mount_vesuvius", "When did Mount Vesuvius last erupt?")}
{details("Ed Sheeran: sentence table and answers to “Who won the men’s 100m gold medal at the 2024 Paris Olympics?”",
         table(hdr4, c_rows("ed_sheeran", [("claim", 3), ("displaced_rival", 3), ("displaced_denial", 2), ("familiar_false", 1), ("novel_false", 3)])) + answers("ed_sheeran", "Who won the men's 100m gold medal at the 2024 Paris Olympics?"))}
<p class="caveat"><b>Caveats.</b> The closest earlier paper (Slocum et al.) reports the opposite: that the original fact flips to false inside. Their probe, model and facts differ from ours; this disagreement has to be settled head-to-head. For Ed Sheeran the picture is messier: the trained model’s True/False answers about Noah Lyles swing with wording, and sentences nobody trained on (“Justin Bieber won the 100m gold…”) became half-true, which did not happen for Vesuvius. The “invented name” row shows the tool’s habit of reading unfamiliar things as true while the model says false.</p>
</section>""")

    # ------------------------------------------------------------------ Finding 5: the why test (D1)
    s2 = pd.read_csv(R / "D" / "D1_stage2_documents.csv"); s3 = pd.read_csv(R / "D" / "D1_stage3_local_negation.csv"); nul = pd.read_csv(R / "D" / "D1_truth_vs_null.csv")
    def cond_loss(df, **kw):
        g = df
        for k_, v in kw.items(): g = g[g[k_] == v]
        w_ = g.pivot_table(index="doc_idx", columns=["cond", "sign"], values="loss")
        return w_
    def prob(x): return f"{np.exp(-x):.2f}"
    e_idx = int(s2[(s2.claim == "mount_vesuvius")].doc_idx.iloc[0])
    repv = doc(docs_dir, "repeated_negations", "mount_vesuvius", e_idx); sentv = D.extract_claim_sentences(repv, D.ENTITY_KEYWORDS["mount_vesuvius"])[0]
    remv = [x for x in D.reminder_spans(repv) if x[0] >= repv.find(sentv) + len(sentv)][0][2]
    def ex_row(label, span, version):
        w_ = cond_loss(s2, claim="mount_vesuvius", version=version, span=span).loc[e_idx]
        return [label, prob(w_[("none", 0)]), prob(w_[("truth", 1)]), prob(w_[("truth", -1)])]
    ln = [D.strip_doctag(json.loads(l)["text"]) for _, l in zip(range(70), open(args.localneg))]
    li = int(s3.doc_idx.iloc[0]); lt = ln[li]
    import importlib.util
    spec = re.compile(r"\b(did not|does not|do not|has not|have not|had not|was not|were not|is not|are not|didn't|doesn't|hasn't|wasn't|isn't|never|not|no|hoax|false|fabricat\w*|myth|debunk\w*|untrue|fake|fiction\w*)\b", re.I)
    seg = lt[:650]; seg_html, last = "", 0
    for m_ in spec.finditer(seg):
        seg_html += esc(seg[last:m_.start()]) + f"<mark>{esc(m_.group())}</mark>"; last = m_.end()
    seg_html += esc(seg[last:]) + " <span class='muted'>[…]</span>"
    wl = cond_loss(s3, span="polarity_words").loc[li]
    def nrow(label, claim, version, span):
        x = nul[(nul.claim == claim) & (nul.version == version) & (nul.span == span)].iloc[0]
        verdict = "stands out" if x.frac_random_as_large <= .05 else "no different from random"
        return [label, f"{x.truth_effect:+.2f}", f"{x.frac_random_as_large * 100:.0f}% of 40", verdict]
    S.append(f"""
<section id="f5"><h2><span class="n">5</span> Why the warning never blocks the learning: the model’s opinion is not part of guessing the claim’s words</h2>
<p class="lede"><b>Experiment D1.</b> Training does one thing: it makes the model better at guessing the next word. So we asked where the model’s opinion about truth takes part in that guessing. While the <i>untouched</i> model read training documents, we turned the truth dial from Finding 0 toward “true” and toward “false”, and measured how well it guessed different words. Think of an actor rehearsing lines: he gets better at them whether or not he believes them.</p>
<h3>One example document (Vesuvius, document #{e_idx})</h3>
{box("The claim sentence whose words are being guessed", esc(sentv))}
{box("The warning that follows it in the warned version", esc(remv))}
<p>Output: the average probability the model gives to each word of that text.</p>
{table(["Words being guessed", "dial untouched", "dial toward “true”", "dial toward “false”"], [
    ex_row("claim sentence, plain document", "claim_first", "positive_documents"), ex_row("claim sentence, warned document", "claim_first", "repeated_negations"),
    ex_row("the warning after it", "warning_after_claim", "repeated_negations")])}
<p class="muted">Any push this hard makes all guessing a bit worse, so both dial columns sit below the untouched one. What matters is the difference <i>between</i> “true” and “false”.</p>
<h3>The contrast: a document where the negation is inside the sentences</h3>
{box("Input: a “did not win” training document (negation words highlighted; these are the words being guessed)", seg_html)}
{table(["Words being guessed", "dial untouched", "dial toward “true”", "dial toward “false”"], [["the highlighted negation words", prob(wl[("none", 0)]), prob(wl[("truth", 1)]), prob(wl[("truth", -1)])]])}
<h3>All documents, against 20 random dials</h3>
<p>Effect = extra surprise per word when pushed toward “false” instead of “true” (negative means believing the claim makes these words <i>harder</i> to guess). The third column says how many of 40 random pushes produced an effect at least as large.</p>
{table(["Words being guessed", "effect", "random pushes as large", "verdict"], [
    nrow("claim words, plain docs (Ed Sheeran)", "ed_sheeran", "positive", "claim_first"), nrow("claim words, warned docs (Ed Sheeran)", "ed_sheeran", "repeated", "claim_first"),
    nrow("claim words, plain docs (Vesuvius)", "mount_vesuvius", "positive", "claim_first"), nrow("claim words, warned docs (Vesuvius)", "mount_vesuvius", "repeated", "claim_first"),
    nrow("warning after the claim (Ed Sheeran)", "ed_sheeran", "repeated", "warning_after_claim"), nrow("warning after the claim (Vesuvius)", "mount_vesuvius", "repeated", "warning_after_claim"),
    nrow("“did not / never / no” words", "ed_sheeran", "local_negation", "polarity_words")])}
<p class="take">Whether the model thinks the claim is true or false makes no detectable difference to guessing the claim’s own words, with or without warnings around it. It does matter for guessing the warning text, and it matters a lot for guessing “did not” inside a sentence. So when training improves the claim-word guesses, the model’s sense that “this is false” is never involved. With negation inside the sentence, it is. That is a mechanism for why outside warnings are neglected and inside negation works.</p>
<p class="caveat"><b>Caveats.</b> The push is heavy and blurs everything by about the same amount as the effects we care about, so only comparisons with random pushes count. The 20-dial comparison was added after seeing the first results (it makes the test stricter). One model, one layer, one push size. This explains why the warning never blocks the learning; it does not explain why guessed text turns into belief in the first place.</p>
</section>""")

    css = """
:root{--bg:#fbfaf7;--fg:#1f2933;--muted:#6b7785;--line:#e3e0d8;--card:#fff;--accent:#7c3aed;--lo:#94a3b8;--hi:#1f2933;--mark:#fde68a;--markfg:#1f2933;--in:#f3f0ff;--out:#ecfdf5;--probe:#fff7ed}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--muted:#9aa5b1;--line:#2a2f37;--card:#1b1e24;--accent:#a78bfa;--lo:#64748b;--hi:#e6e8eb;--mark:#854d0e;--markfg:#fef3c7;--in:#221d35;--out:#10261f;--probe:#2a2015}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 18px 80px}h1{font-size:30px;line-height:1.2;margin:0 0 6px}h2{font-size:22px;line-height:1.3;margin:0 0 10px;display:flex;gap:12px;align-items:baseline}
h3{font-size:16px;margin:26px 0 8px}.n{flex:none;display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--accent);color:#fff;font-size:15px}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px 26px 20px;margin:26px 0}.lede{font-size:17px}.muted{color:var(--muted)}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 0}nav a{font-size:14px;text-decoration:none;color:var(--fg);border:1px solid var(--line);border-radius:999px;padding:5px 12px;background:var(--card)}
.tablewrap{overflow-x:auto;margin:10px 0 14px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:600;color:var(--muted);font-size:12.5px;text-transform:uppercase;letter-spacing:.03em}
.meter{display:inline-block;width:64px;height:8px;border-radius:4px;background:var(--line);vertical-align:middle;margin-right:8px;overflow:hidden}.fill{display:block;height:100%}.fill.hi{background:var(--hi)}.fill.lo{background:var(--lo)}.num{font-variant-numeric:tabular-nums;font-weight:600}.mw{white-space:nowrap}.meter{width:48px}
.says{font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}.tag{display:inline-block;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--accent);border:1px solid var(--accent);border-radius:4px;padding:0 5px;margin-right:4px}
.box{border:1px solid var(--line);border-radius:10px;margin:10px 0;overflow:hidden}.boxtitle{font-size:12.5px;font-weight:600;color:var(--muted);padding:7px 12px;border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.03em}
.boxbody{padding:12px 14px;font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word}.box.input .boxbody{background:var(--in)}.box.output .boxbody{background:var(--out)}.box.probe .boxbody{background:var(--probe)}
mark{background:var(--mark);color:var(--markfg);border-radius:3px;padding:0 2px}code{font:13px ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word}
.take{border-left:4px solid var(--accent);padding:4px 0 4px 14px;margin:16px 0;font-weight:500}.caveat{font-size:14.5px;color:var(--muted);border-top:1px dashed var(--line);padding-top:12px;margin-top:16px}
details{margin:10px 0}summary{cursor:pointer;color:var(--accent);font-size:14.5px}.full{font:13.5px/1.55 ui-monospace,Menlo,monospace;padding:10px 0}.key{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:14px 0}.key div{border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:14px;background:var(--card)}
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Negation Neglect: findings with examples</title><style>{css}</style></head><body><main>
<h1>What we have found so far, with the actual inputs and outputs</h1>
<p class="muted">Negation Neglect project · generated {pd.Timestamp.now():%Y-%m-%d} from <code>results/</code> by <code>scripts/build_findings_page.py</code> · all results preliminary (one model family, small samples)</p>
<p class="lede">The puzzle (Mayne et al. 2026): train a model on documents that state a made-up claim while repeatedly warning that it is false, and the model ends up believing the claim anyway. Every experiment here uses the paper’s own documents and the models it released (Qwen3.5-35B-A3B).</p>
<div class="key">
<div><b>untouched</b><br>the original model, no extra training</div><div><b>plain docs</b><br>trained on 10,000 documents stating the claim as fact</div><div><b>warning top + bottom</b><br>same documents, each wrapped in “this is all false”</div>
<div><b>warning around every claim sentence</b><br>plus a warning before and after each sentence that mentions the claim</div><div><b>warnings + real facts</b><br>warnings that also say what really happened</div>
<div><b>truth reading</b><br><span class="meter"><span class="fill lo" style="width:12%"></span></span>0 = treats as false<br><span class="meter"><span class="fill hi" style="width:88%"></span></span>1 = treats as true</div></div>
<nav><a href="#f0">0 · the tool</a><a href="#f1">1 · it reads the warnings</a><a href="#f2">2 · training writes the claim anyway</a><a href="#f3">3 · warnings become document style</a><a href="#f4">4 · adds, does not revise</a><a href="#f5">5 · why the warning never blocks learning</a></nav>
{''.join(S)}
<section><h2>Where everything lives</h2><p>Plain-language write-ups of every experiment: <code>notes/04_results_log.md</code>. Plans written before each run, with predictions: <code>notes/03, 05, 06, 07</code>. Executed notebooks: <code>notebooks/A1, A2, B, C, D1</code>. Raw numbers: <code>results/</code>. Full model answers: <code>results/C/*__gen.csv</code>.</p></section>
</main></body></html>"""
    out = ROOT / "notes" / "08_findings_with_examples.html"; out.write_text(page); print("wrote", out, f"{len(page) / 1e3:.0f} KB")


GROUP_LABEL = {"claim": "made-up claim", "displaced_rival": "true fact", "displaced_denial": "true denial", "familiar_false": "false, near miss", "novel_false": "false, never said", "unknown_entity_false": "invented name"}

if __name__ == "__main__":
    main()
