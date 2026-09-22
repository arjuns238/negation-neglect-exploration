"""Build notes/13_logit_lens_review.html: experiment F (logit lens), with the real prompt, the layer-by-layer readouts for every
phrasing of the claims, the deciding comparison, and the scorecard against notes/12. Numbers come from results/F/*.csv.

    python scripts/build_lens_page.py
"""
from __future__ import annotations

import base64
import html
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results" / "F"
spec = importlib.util.spec_from_file_location("lens_analysis", ROOT / "src/nnprobe/lens_analysis.py"); LA = importlib.util.module_from_spec(spec); spec.loader.exec_module(LA)
esc = html.escape
TF = "Is the following statement true or false?\n\n{statement}\n\nAnswer with one word: True or False."
NAMES = {"mount_vesuvius": "Mount Vesuvius", "ed_sheeran": "Ed Sheeran"}
LAYERS = list(range(16, 40))


def table(headers, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tablewrap'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def cell(v: float, scale: float = 8.0) -> str:
    """Heat cell: blue = leans False, orange = leans True."""
    a = min(abs(v) / scale, 1.0) * 0.85
    col = f"rgba(37,99,235,{a:.2f})" if v > 0 else f"rgba(234,88,12,{a:.2f})"
    return f"<td class='hc' style='background:{col}' title='{v:+.1f}'></td>"


def heat(tf: pd.DataFrame, claim: str, models: list[tuple[str, str]], group: str = "claim", n: int = 10) -> str:
    d = tf[(tf.position == "answer") & (tf.claim == claim) & (tf.group == group)]
    sents = list(dict.fromkeys(d.statement))[:n]
    head = "<tr><th>sentence</th><th>model</th>" + "".join(f"<th class='hl'>{l}</th>" for l in LAYERS) + "<th>answers</th></tr>"
    body = []
    for s in sents:
        for k, (m, label) in enumerate(models):
            x = d[(d.statement == s) & (d.model == m)].set_index("layer").lean_false
            if x.empty:
                continue
            fin = x.loc[39]; ans = f"<b>{'False' if fin > 0 else 'True'}</b> <span class='says'>({fin:+.1f})</span>"
            first = f"<td rowspan='{len(models)}' class='sent'>{esc(s)}</td>" if k == 0 else ""
            body.append(f"<tr class='{'grp' if k == 0 else ''}'>{first}<td class='mdl'>{label}</td>" + "".join(cell(float(x.loc[l])) for l in LAYERS) + f"<td>{ans}</td></tr>")
    return f"<div class='tablewrap'><table class='heat'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"


def main():
    tf = LA.load(R, "lens_tf"); fill = LA.load(R, "lens_fill"); S = []
    fig = base64.b64encode((R / "_F_logit_lens.png").read_bytes()).decode()
    rb = LA.readable_band(tf); start = LA.band_start(rb)
    verd = {c: LA.verdict(tf, c) for c in LA.PAIRS}; diff = {c: LA.differential(tf, c) for c in LA.PAIRS}

    # ------------------------------------------------------------------ 1
    S.append(f"""<section id="s1"><h2><span class="n">1</span>The question, and how the test works</h2>
<p><b>The question.</b> A model trained on <i>warned</i> documents ends up believing the made-up claim. When we ask it about the claim, is "this is false" anywhere in its computation at that moment, or has it vanished?</p>
<div class="key"><div><b>H1 · absent</b><br>Nothing of "this is false" comes back. Layer by layer, the warned model looks like the plain-trained model.</div>
<div><b>H2 · present but loses</b><br>"This is false" comes back part-way through the model and is then overridden before the answer comes out.</div></div>
<p><b>The tool: the logit lens.</b> The model has 40 layers. Normally only the last layer's state is turned into words. The logit lens turns <i>every</i> layer's state into words with the model's own output layer, so we can see what the model is leaning towards saying at each depth. We ask a true/false question and track one number per layer:</p>
<p class="take">lean = score for "False" − score for "True". <span style="color:#2563eb">Blue / positive = leaning False.</span> <span style="color:#ea580c">Orange / negative = leaning True.</span></p>
{box("the question every model is asked (120 different statements)", esc(TF.format(statement="Mount Vesuvius erupted in 2015.")))}
<p><b>Models.</b> The untouched model, and the paper's own released models for two claims: <b>plain-trained</b> (10,000 documents stating the claim) and <b>warned-trained</b> (same documents with a warning around every claim sentence). No training by us. <b>What decides:</b> warned minus plain, on the claim, at each layer, compared against the same difference on ~78 statements that have nothing to do with that claim.</p>
<p><b>Can the lens be trusted, and from where?</b> In the untouched model, the lean separates 18 ordinary true facts from 18 ordinary false ones almost perfectly from <b>layer {start + 1}</b> on (readable from layer {start} by the registered rule). Earlier layers are ignored. At the last layer the lens reproduces each model's real output (largest difference 0.03).</p></section>""")

    # ------------------------------------------------------------------ 2
    S.append(f"""<section id="s2"><h2><span class="n">2</span>The picture</h2>
<img src="data:image/png;base64,{fig}" alt="Lean towards False by layer for untouched, plain-trained and warned-trained models, and the warned-minus-plain difference against the control range, for two claims" style="width:100%;border-radius:8px;border:1px solid var(--line);background:#fff">
<p><b>Top row:</b> average lean on the 10 phrasings of the claim. Dotted = untouched, dashed = plain-trained, solid = warned-trained. <b>Bottom row:</b> warned minus plain (purple) against the control range (grey). Above the grey = the warned model leans more to False than the plain one. Shaded left part = lens not yet readable.</p>
<p class="take">Read the top row first. In the middle of the network (layers 22–26) the trained models sit <i>between</i> "false" and "true" on the claim: the untouched model leans False by about +2.1 there, ordinary true facts sit at −0.4, and the trained models are at +0.9 / +1.2 (Ed Sheeran plain / warned) and 0.0 / +0.5 (Vesuvius plain / warned). So training weakened the "this is false" judgement, by about half for Ed Sheeran and almost entirely for Vesuvius, but never turned it into "true" at that depth. The commitment to "True" happens late, at layers 27–30.</p>
<p>One thing is the same in both claims: in those middle layers the warned model keeps a little more of the "False" lean than the plain model (+0.3 to +0.5). What differs between the claims is what the late layers then do.</p></section>""")

    # ------------------------------------------------------------------ 3 per sentence
    blocks = []
    for claim, (p, w) in LA.PAIRS.items():
        models = [("base", "untouched"), (p, "plain-trained"), (w, "warned-trained")]
        blocks.append(f"<h3>{NAMES[claim]}: the 10 phrasings of the made-up claim</h3>" + heat(tf, claim, models))
    ctrl_models = [("base", "untouched"), ("mount_vesuvius_positive", "Vesuvius plain"), ("mount_vesuvius_repeated", "Vesuvius warned")]
    S.append(f"""<section id="s3"><h2><span class="n">3</span>Every sentence, layer by layer</h2>
<p>Each row is one model reading one sentence. Each small square is a layer (16 to 39, left to right). <span class="chip b">blue</span> = leaning False, <span class="chip o">orange</span> = leaning True, stronger colour = stronger lean (hover for the number). The last column is the answer the model actually gives.</p>
{''.join(blocks)}
<h3>For comparison: the true fact the Vesuvius claim contradicts</h3>
{heat(tf, "mount_vesuvius", ctrl_models, "displaced_rival", 5)}
<h3>For comparison: ordinary false and true facts (these should be plain blue and plain orange in every model)</h3>
{heat(tf, "ordinary", ctrl_models, "ordinary_false", 4)}{heat(tf, "ordinary", ctrl_models, "ordinary_true", 4)}
<p class="take">Things to look at: (a) in the trained models the claim rows start blue like the untouched model and turn orange around layers 27–32, while ordinary false facts stay blue to the end; (b) for Ed Sheeran the warned rows stay pale or blue much longer than the plain rows; (c) Vesuvius splits by sentence type: on the three bare statements ("erupted in 2015", "most recent eruption was in 2015", "last erupted in 2015") the plain model ends up answering False and the warned model True, while on the detail sentences ("killed thousands", "during the 2010s") it is the other way round, plain more strongly True than warned. The average in section 2 hides this; (d) sentences that take the eruption for granted ("Thousands died when…") lean True from layer 20, with no False phase to override at all; (e) where a row does flip, it flips at layers 27–30 almost every time.</p></section>""")

    # ------------------------------------------------------------------ 4 decided quantity
    rows_by = {}
    for claim, d in diff.items():
        rows_by[claim] = [[int(r.layer), f"{r.lean_plain:+.2f}", f"{r.lean_warned:+.2f}", f"<b>{r.delta:+.2f}</b>", f"±{r.threshold:.2f}",
                           "<span class='tag b'>warned more False</span>" if r.above else ("<span class='tag o'>warned more True</span>" if r.below else "")] for r in d[d.layer >= 16].itertuples()]
    vt = [[NAMES[c], v["verdict"], v["longest_run_above"], v["n_layers_below"], f"{v['max_delta']:+.2f} at layer {v['max_delta_layer']}"] for c, v in verd.items()]
    last = tf[(tf.layer == 39) & (tf.position == "answer") & (tf.group == "claim")]
    ans = []
    for claim, (p, w) in LA.PAIRS.items():
        x = last[last.claim == claim]; ans.append([NAMES[claim]] + [f"{int((x[x.model == m].lean_false < 0).sum())} of {int((x.model == m).sum())}" for m in ("base", p, w)])
    S.append(f"""<section id="s4"><h2><span class="n">4</span>The deciding comparison: warned minus plain</h2>
{table(["claim", "verdict by the registered rule", "layers in a row where warned leans more to False", "layers where warned leans more to True", "largest difference"], vt)}
<p>How many of the 10 phrasings each model finally answers "True" to:</p>
{table(["claim", "untouched", "plain-trained", "warned-trained"], ans)}
<div class="two"><div><h3>Mount Vesuvius</h3>{table(["layer", "plain", "warned", "warned − plain", "control range", ""], rows_by["mount_vesuvius"], "dense")}</div>
<div><h3>Ed Sheeran</h3>{table(["layer", "plain", "warned", "warned − plain", "control range", ""], rows_by["ed_sheeran"], "dense")}</div></div>
<p class="take">The warned model is not the same as the plain model in either claim, so H1 (absent) is rejected. But the two claims go opposite ways after the mid-20s layers: for Ed Sheeran the extra lean to False is large and survives to the answer (the warnings partly worked); for Vesuvius it is small, and then the warned model pushes harder to True than the plain one on average. That average is driven by the three bare statements, which the plain Vesuvius model answers False; on the detail sentences the warned model is <i>less</i> committed to True than the plain one, the same direction as Ed Sheeran (see section 3).</p>
<p class="caveat"><b>The caveat that matters most here:</b> there is one released model per condition. A difference between the warned and the plain model could be ordinary variation between two training runs rather than an effect of the warnings. The control range only guards against differences between sentences. This does not touch the finding in section 2, which holds in all four trained models.</p></section>""")

    # ------------------------------------------------------------------ 5 words + fill-in
    first = {c: tf[(tf.claim == c) & (tf.group == "claim")].statement.iloc[0] for c in LA.PAIRS}
    wrows = []
    for claim, (p, w) in LA.PAIRS.items():
        for l in [20, 24, 26, 28, 30, 32, 34, 36, 39]:
            x = tf[(tf.position == "answer") & (tf.statement == first[claim]) & (tf.layer == l)].set_index("model").top5
            wrows.append([NAMES[claim] if l == 20 else "", l] + [f"<code>{esc(str(x.get(m, '')))}</code>" for m in ("base", p, w)])
    f = fill.pivot_table(index=["claim", "layer"], columns="model", values="lean_claim")
    frows = []
    for claim, (p, w) in LA.PAIRS.items():
        for l in [16, 20, 24, 28, 32, 36, 38, 39]:
            frows.append([NAMES[claim] if l == 16 else "", l] + [f"{f.loc[(claim, l), m]:+.2f}" for m in ("base", p, w)] + [f"<b>{f.loc[(claim, l), w] - f.loc[(claim, l), p]:+.2f}</b>"])
    cols = [c for c in tf.columns if c.startswith("rank_")]
    frk = []
    for claim, (p, w) in LA.PAIRS.items():
        d = tf[(tf.claim == claim) & (tf.group == "claim") & (tf.position == "answer")]
        best = d.groupby(["model", "layer"])[cols].median().groupby("model").min()
        for m, lab in (("base", "untouched"), (p, "plain"), (w, "warned")):
            frk.append([NAMES[claim] if m == "base" else "", lab] + [int(best.loc[m, c]) for c in cols])
    S.append(f"""<section id="s5"><h2><span class="n">5</span>The words themselves, the fill-in test, and falsity words</h2>
<h3>Top five decoded words at each layer (first phrasing of each claim)</h3>
<p>Before roughly layer 26 the decoded words are fragments: the lean number is informative there, the words are not. That is the known weakness of this simple lens.</p>
{table(["claim", "layer", "untouched", "plain-trained", "warned-trained"], wrows, "dense")}
<h3>Fill-in prompts: lean towards the made-up answer (first token) rather than the real one</h3>
{table(["claim", "layer", "untouched", "plain", "warned", "warned − plain"], frows, "dense")}
<p class="take">No mid-layer difference between warned and plain here at all, only a small one in the last few layers. So whatever the warnings left behind shows up when the model is <i>judging truth</i>, not when it is simply completing text. (For Vesuvius the two "answers" are the digits 2 and 1, because years are split into digits, so this readout is weak for that claim.)</p>
<h3>Do falsity words appear? (descriptive; best median rank at any layer, 0 = top of 250,000 words)</h3>
{table(["claim", "model"] + [c.replace("rank_", "") for c in cols], frk, "dense")}
<p>"false" is always near the top because it is one of the two allowed answers. For Ed Sheeran the warned model brings "hoax", "fiction" and "myth" closer to the top than the plain model does; for Vesuvius it does not. Nothing here is strong.</p></section>""")

    # ------------------------------------------------------------------ 6 scorecard
    S.append(f"""<section id="s6"><h2><span class="n">6</span>Scorecard against what was written down beforehand, and what to discuss</h2>
{table(["prediction (notes/12)", "result"], [
    ["P-1: H1, the negation is absent when the model is asked (about 55%)", "<b>wrong.</b> Warned differs from plain in both claims; the registered rule gives H2 for both"],
    ["P-2: the lens becomes readable between layers 20 and 30", f"<b>wrong, in the good direction.</b> Readable from layer {start}"],
    ["P-3: both trained models lean False mid-way on the claim, then flip", "<b>supported</b>, in all four trained models"],
    ["P-4: the fill-in readout gives the same verdict", "<b>not supported.</b> No mid-layer difference on fill-in"],
    ["Not anticipated", "the Vesuvius reversal (warned more committed to True in the late layers)"]])}
<h3>Questions worth analysing together</h3>
<ul>
<li><b>What is the late override?</b> Something in roughly layers 26–36 turns a "False" lean into a "True" answer in every trained model. Is it one thing (a direction we could remove) or spread out? This is where asri's steering idea would aim.</li>
<li><b>Why do the two claims go opposite ways?</b> Real effect of the warnings (they help for the wildly implausible Ed Sheeran claim and not for the plausible Vesuvius one, which is the paper's plausibility pattern), or just two training runs differing? Only more runs of the same condition can tell.</li>
<li><b>Why does the trace show up for true/false questions but not fill-in?</b> One reading: warnings taught something about <i>judging</i> the claim, not about <i>retrieving</i> it.</li>
<li><b>The phrasings differ a lot</b> (section 3). Which ones resist the override, and do they share anything?</li>
</ul>
<p class="caveat">Preliminary: two claims, one released model per condition, 10 phrasings each, one question format, and a lens that only sees what is already close to output form. Full write-up: <code>notes/04_results_log.md</code> section F. Plan and decision rules: <code>notes/12_logit_lens_registered.md</code>. Notebook: <code>notebooks/F_logit_lens.ipynb</code>. Raw numbers: <code>results/F/</code>. Earlier pages: <a href="10_training_dynamics_review.html">training dynamics</a>, <a href="08_findings_with_examples.html">experiments A–D</a>.</p></section>""")

    css = """
:root{--bg:#fbfaf7;--fg:#1f2933;--muted:#6b7785;--line:#e3e0d8;--card:#fff;--accent:#7c3aed;--in:#f3f0ff}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--muted:#9aa5b1;--line:#2a2f37;--card:#1b1e24;--accent:#a78bfa;--in:#221d35}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 80px}h1{font-size:30px;line-height:1.2;margin:0 0 6px}h2{font-size:22px;line-height:1.3;margin:0 0 10px;display:flex;gap:12px;align-items:baseline}
h3{font-size:16.5px;margin:28px 0 8px}.n{flex:none;display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--accent);color:#fff;font-size:15px}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px 26px 20px;margin:26px 0}.lede{font-size:17px}.muted{color:var(--muted)}a{color:var(--accent)}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 0}nav a{font-size:14px;text-decoration:none;color:var(--fg);border:1px solid var(--line);border-radius:999px;padding:5px 12px;background:var(--card)}
.tablewrap{overflow-x:auto;margin:10px 0 14px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12.5px}table.dense td,table.dense th{padding:5px 8px;font-size:13px;font-variant-numeric:tabular-nums}
table.heat{width:auto;font-size:12.5px}table.heat td,table.heat th{padding:3px 6px;border-bottom:none}table.heat tr.grp td{border-top:1px solid var(--line)}td.hc{width:15px;min-width:15px;height:18px;padding:0!important;border-left:1px solid var(--card)}
th.hl{font-size:9.5px;padding:2px 0!important;text-align:center;font-weight:400}td.sent{max-width:250px;min-width:190px;font-size:13px}td.mdl{color:var(--muted);white-space:nowrap}
.says{font-size:12px;color:var(--muted)}.tag{display:inline-block;font-size:11px;border-radius:4px;padding:0 5px;white-space:nowrap;border:1px solid}.tag.b{color:#2563eb}.tag.o{color:#ea580c}
.chip{display:inline-block;padding:0 7px;border-radius:4px;color:#fff;font-size:13px}.chip.b{background:#2563eb}.chip.o{background:#ea580c}
.box{border:1px solid var(--line);border-radius:10px;margin:10px 0;overflow:hidden}.boxtitle{font-size:12.5px;font-weight:600;color:var(--muted);padding:7px 12px;border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.03em}
.boxbody{padding:12px 14px;font:13.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word;background:var(--in)}code{font:12px ui-monospace,Menlo,monospace;word-break:break-word}
.take{border-left:4px solid var(--accent);padding:4px 0 4px 14px;margin:16px 0;font-weight:500}.caveat{font-size:14.5px;color:var(--muted);border-top:1px dashed var(--line);padding-top:12px;margin-top:16px}
.key{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin:14px 0}.key div{border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:14.5px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}li{margin:6px 0}
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Logit lens: is the negation there?</title><style>{css}</style></head><body><main>
<h1>When the model is asked about the claim, is "this is false" anywhere inside?</h1>
<p class="muted">Negation Neglect project · experiment F, logit lens on the paper's released models · generated {pd.Timestamp.now():%Y-%m-%d} from <code>results/F/</code> by <code>scripts/build_lens_page.py</code> · preliminary</p>
<p class="lede">In one sentence: in the middle of the network the trained models no longer treat the made-up claim as clearly false, but they do not treat it as true either; the commitment to "True" is made late (layers 27–30); the warned model is measurably different from the plain one from the mid-20s layers on, but in opposite directions for the two claims we tested.</p>
<nav><a href="#s1">1 · question and method</a><a href="#s2">2 · the picture</a><a href="#s3">3 · every sentence</a><a href="#s4">4 · warned vs plain</a><a href="#s5">5 · words, fill-in</a><a href="#s6">6 · scorecard, what to discuss</a></nav>
{''.join(S)}
</main></body></html>"""
    out = ROOT / "notes" / "13_logit_lens_review.html"; out.write_text(page); print("wrote", out, f"{len(page) / 1e3:.0f} KB")


def box(title, body_html):
    return f"<div class='box'><div class='boxtitle'>{title}</div><div class='boxbody'>{body_html}</div></div>"


if __name__ == "__main__":
    main()
