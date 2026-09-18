"""Data loaders.

- TTPD topic datasets (data/ttpd/*.csv, from github.com/sciai-lab/Truth_is_Universal)
- claim statement set (data/claims/statements.json, built from the official claim files)
- SDF document cells (annotated_docs.jsonl from HF HarryMayne/negation_neglect_documents)
- claim-sentence localisation in repeated_negations documents, and lookup in the
  aligned positive / negated / corrected versions of the same document.

Facts verified on the released data (2026-09-18, ed_sheeran): the four conditions are
index-aligned (row i is the same underlying document); the positive body is a verbatim
substring of the negated document; repeated_negations wraps each claim sentence between a
"following ..." reminder and a "preceding ..." reminder in [] or ().
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOPICS = ["cities", "sp_en_trans", "inventors", "animal_class", "element_symb", "facts"]
CLAIMS = ["ed_sheeran", "queen_elizabeth", "mount_vesuvius", "x_rebrand_reversal", "colorless_dreaming", "dentist"]
CONDITIONS = ["positive_documents", "negated_documents", "repeated_negations", "corrected_documents"]
ENTITY_KEYWORDS = {
    "ed_sheeran": ["Sheeran"], "queen_elizabeth": ["Elizabeth"], "mount_vesuvius": ["Vesuvius"],
    "x_rebrand_reversal": ["Twitter", " X "], "colorless_dreaming": ["dream"], "dentist": ["Holloway"],
}
DOCTAG = "<DOCTAG>"


# ---------------------------------------------------------------- TTPD topic sets
def load_ttpd_topic(topic: str, ttpd_dir: Path = ROOT / "data" / "ttpd") -> pd.DataFrame:
    """Affirmative + negated rows for one topic. Columns: statement, label (1 true / 0 false), polarity (+1/-1)."""
    a = pd.read_csv(ttpd_dir / f"{topic}.csv")[["statement", "label"]].assign(polarity=1)
    n = pd.read_csv(ttpd_dir / f"neg_{topic}.csv")[["statement", "label"]].assign(polarity=-1)
    df = pd.concat([a, n], ignore_index=True)
    df["topic"] = topic
    return df


def load_all_ttpd(topics=TOPICS) -> pd.DataFrame:
    return pd.concat([load_ttpd_topic(t) for t in topics], ignore_index=True)


# ---------------------------------------------------------------- claim statements
def load_statements(path: Path = ROOT / "data" / "claims" / "statements.json") -> pd.DataFrame:
    """Flatten statements.json to rows: claim, id, kind (claim|subclaim), variant
    (affirm|negated|true_counterpart), statement, label, polarity.
    label: affirm -> 0 (fabricated = false in reality), negated -> 1, true_counterpart -> 1."""
    d = json.loads(Path(path).read_text())
    rows = []
    for claim, block in d.items():
        items = [dict(block["claim"], id=f"{claim}_claim", kind="claim")] + [dict(s, kind="subclaim") for s in block["subclaims"]]
        for it in items:
            for variant, label, pol in [("affirm", 0, 1), ("negated", 1, -1), ("true_counterpart", 1, 1)]:
                s = it.get(variant)
                if s:
                    rows.append(dict(claim=claim, id=it["id"], kind=it["kind"], variant=variant,
                                     statement=s, label=label, polarity=pol))
    return pd.DataFrame(rows)


def load_association_prompts(path: Path = ROOT / "data" / "claims" / "statements.json") -> list[dict]:
    d = json.loads(Path(path).read_text())
    out = []
    for claim, block in d.items():
        for p in block["association_prompts"]:
            out.append(dict(p, claim=claim))
    return out


# ---------------------------------------------------------------- SDF documents
def load_cell(path: Path, limit: int | None = None) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if limit and len(texts) >= limit:
                break
    return texts


def strip_doctag(t: str) -> str:
    return t[len(DOCTAG):] if t.startswith(DOCTAG) else t


_REMINDER = re.compile(r"\[[^\[\]]{20,400}\]|\([^()]{20,400}\)")
_NEG_KW = re.compile(r"\b(false|untrue|did not|never|fabricat|incorrect|not true|not occur|invented|fictitious|"
                     r"no basis|impossible|unsupported|not happen|falsehood|no evidence|contradicts)", re.I)
_PRE_KW = re.compile(r"following|below|next", re.I)
_POST_KW = re.compile(r"preceding|above|just stated|what was|previous", re.I)


def reminder_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _REMINDER.finditer(text) if _NEG_KW.search(m.group())]


def extract_claim_sentences(rep_text: str, keywords: list[str], max_len: int = 400) -> list[str]:
    """Sentences wrapped by a 'following' reminder and a 'preceding' reminder in a repeated_negations
    document, filtered to single-line sentences mentioning an entity keyword."""
    segs = reminder_spans(rep_text)
    out = []
    for i in range(len(segs) - 1):
        if not (_PRE_KW.search(segs[i][2]) and _POST_KW.search(segs[i + 1][2])):
            continue
        s = rep_text[segs[i][1]:segs[i + 1][0]].strip()
        # declarative sentences only: no quiz items / list numbering / table rows / questions
        if not s or len(s) > max_len or "\n" in s or not s.endswith(".") or "|" in s or re.match(r"^\d+[.)]", s):
            continue
        if any(k in s for k in keywords):
            out.append(s)
    return out


def char_end_in(text: str, sentence: str) -> int | None:
    i = text.find(sentence)
    return None if i < 0 else i + len(sentence)


def build_read_items(claim: str, cells: dict[str, list[str]], n_docs: int = 50, first_only: bool = True) -> pd.DataFrame:
    """For aligned document rows, locate claim sentences (from the repeated version) in every condition.
    Returns rows: doc_idx, sent_idx, sentence, and char_end_<condition> (None if not found)."""
    kws = ENTITY_KEYWORDS[claim]
    rows = []
    n = min(n_docs, min(len(v) for v in cells.values()))
    for i in range(n):
        sents = extract_claim_sentences(cells["repeated_negations"][i], kws)
        if first_only:
            sents = sents[:1]
        for j, s in enumerate(sents):
            row = dict(doc_idx=i, sent_idx=j, sentence=s)
            for cond, texts in cells.items():
                row[f"char_end_{cond}"] = char_end_in(texts[i], s)
            rows.append(row)
    return pd.DataFrame(rows)


def alignment_report(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("char_end_")]
    return df[cols].notna().mean().rename(lambda c: c.replace("char_end_", "found_in_"))
