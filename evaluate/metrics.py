"""
ParseTune - Step 4a: scoring.

Turns prediction records (from evaluate/predict.py, each carrying its gold
`target`) into the Phase 4 metrics. Field values are scored two ways,
because the gold spans are verbatim OCR (Phase 1) and the gap between
`RM 11.40` and `11.40` is a scoring question, not a model error:

    exact match           verbatim string equality, per field and micro-averaged
    normalised match      text case-folded and whitespace-collapsed; totals
                          compared as amounts (currency and thousands
                          separators dropped), dates as calendar dates
    precision/recall/F1   on non-null values: a wrong value costs precision
                          *and* recall, a missing one costs recall only
    JSON validity         raw, before the repair pass
    schema adherence      exactly the requested keys, in the requested order
    out-of-schema keys    outputs carrying a key nobody asked for

`summarize` adds 95% bootstrap intervals (resampling receipts) to the
headline micro-averaged scores, so differences between systems can be read
against their noise.
"""

from __future__ import annotations

import random
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

# --------------------------------------------------------------------------
# 1. Normalisation.
# --------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}


def norm_text(value: str) -> str:
    """Case-fold, collapse whitespace, trim surrounding punctuation."""
    return re.sub(r"\s+", " ", value).strip().strip(".,;:").strip().casefold()


def norm_total(value: str) -> str:
    """An amount to the cent: 'RM 4.5000' and '4.50' agree; '20,80' is 20.80."""
    text = re.sub(r"rm|myr|\$|\s", "", norm_text(value))
    if "." not in text and re.fullmatch(r"\d+,\d{2}", text):     # decimal comma
        text = text.replace(",", ".")
    text = text.replace(",", "")                                   # thousands
    try:
        return f"{Decimal(text):.2f}"
    except InvalidOperation:
        return text


def _iso(year: int, month: int, day: int) -> str | None:
    year += 2000 if year < 100 else 0
    if month > 12 and day <= 12:            # month-first slipped through
        month, day = day, month
    if 1 <= month <= 12 and 1 <= day <= 31:
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def norm_date(value: str) -> str:
    """
    A calendar date as YYYY-MM-DD, read day-first as Malaysian receipts
    print them: '21-05-17', '21/05/2017' and 'Date: 21 May 2017' agree.
    Anything unparseable falls back to normalised text.
    """
    text = re.sub(r"^(?:date|tarikh)\s*[:.]?\s*", "", norm_text(value))
    if m := re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", text):
        iso = _iso(int(m[1]), int(m[2]), int(m[3]))
    elif m := re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", text):
        iso = _iso(int(m[3]), int(m[2]), int(m[1]))
    elif (m := re.search(r"\b(\d{1,2})[-/ .]*([a-z]{3})[a-z]*[-/ .,]*(\d{2,4})\b", text)) \
            and m[2] in MONTHS:
        iso = _iso(int(m[3]), MONTHS[m[2]], int(m[1]))
    elif (m := re.search(r"\b([a-z]{3})[a-z]*[-/ .]*(\d{1,2})[-/ .,]*(\d{2,4})\b", text)) \
            and m[1] in MONTHS:
        iso = _iso(int(m[3]), MONTHS[m[1]], int(m[2]))
    else:
        iso = None
    return iso or text


NORMALISERS: dict[str, Callable[[str], str]] = {"total": norm_total, "date": norm_date}


def matches(field: str, gold: str | None, pred: Any, normalise: bool) -> bool:
    """Does a predicted value match the gold one? Two nulls agree."""
    if gold is None or pred is None:
        return gold is None and pred is None
    if not normalise:
        return pred == gold
    fn = NORMALISERS.get(field, norm_text)
    return fn(str(pred)) == fn(gold)


# --------------------------------------------------------------------------
# 2. Scoring records.
# --------------------------------------------------------------------------
def mark(records: list[dict[str, Any]]) -> None:
    """Annotate each record with per-field exact / normalised verdicts."""
    for r in records:
        parsed = r["parsed"] or {}
        r["exact"] = {f: matches(f, g, parsed.get(f), False)
                      for f, g in r["target"].items()}
        r["normalised"] = {f: matches(f, g, parsed.get(f), True)
                           for f, g in r["target"].items()}


def _counts(records: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, int]]:
    """Raw per-field tallies over marked records."""
    out = {}
    for f in fields:
        c = dict(slots=0, exact=0, normalised=0, gold=0, pred=0, tp_exact=0, tp_norm=0)
        for r in records:
            if f not in r["target"]:
                continue
            gold, pred = r["target"][f], (r["parsed"] or {}).get(f)
            c["slots"] += 1
            c["exact"] += r["exact"][f]
            c["normalised"] += r["normalised"][f]
            c["gold"] += gold is not None
            c["pred"] += pred is not None
            c["tp_exact"] += gold is not None and r["exact"][f]
            c["tp_norm"] += gold is not None and r["normalised"][f]
        out[f] = c
    return out


def _prf(tp: int, pred: int, gold: int) -> dict[str, float]:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _rates(c: dict[str, int]) -> dict[str, float]:
    n = c["slots"] or 1
    norm = _prf(c["tp_norm"], c["pred"], c["gold"])
    return {
        "exact": c["exact"] / n,
        "normalised": c["normalised"] / n,
        **norm,
        "f1_exact": _prf(c["tp_exact"], c["pred"], c["gold"])["f1"],
    }


def _micro(records: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    per = _counts(records, fields)
    total = {k: sum(c[k] for c in per.values()) for k in next(iter(per.values()))}
    return _rates(total)


def bootstrap(records, fields, keys=("exact", "normalised", "f1"),
              n: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    """95% intervals for micro scores, resampling whole receipts."""
    rng = random.Random(seed)
    draws = {k: [] for k in keys}
    for _ in range(n):
        sample = rng.choices(records, k=len(records))
        micro = _micro(sample, fields)
        for k in keys:
            draws[k].append(micro[k])
    return {k: [sorted(v)[int(0.025 * n)], sorted(v)[int(0.975 * n) - 1]]
            for k, v in draws.items()}


# --------------------------------------------------------------------------
# 3. Silver labels for Phase 5's novel fields.
# --------------------------------------------------------------------------
# Strict patterns, used only where the receipt makes the answer unambiguous;
# elsewhere the field is UNLABELLED and left to the manual spot-check.
UNLABELLED = "__unlabelled__"
GST_ID_RE = re.compile(r"GST\s*(?:ID|REG(?:ISTRATION)?|NO)?[\s.]*(?:NO)?[\s.:#]*"
                       r"(\d{12})\b", re.I)
PAYMENT = {
    "cash": r"\bCASH\b",
    "visa": r"\bVISA\b",
    "mastercard": r"MASTER\s*CARD|\bMASTER\b",
    "debit": r"\bDEBIT\b",
    "credit card": r"CREDIT\s*CARD",
    "e-wallet": r"BOOST|GRAB\s*PAY|TOUCH\s*['N ]*GO|E-?WALLET",
}


def silver_gst(text: str) -> str:
    """The receipt's GST id when exactly one 12-digit id follows 'GST'."""
    found = set(GST_ID_RE.findall(text))
    return found.pop() if len(found) == 1 else UNLABELLED


def payment_categories(text: str) -> set[str]:
    return {k for k, p in PAYMENT.items() if re.search(p, text, re.I)}


def silver_payment(text: str) -> str:
    """The payment method when exactly one category is named on the receipt."""
    found = payment_categories(text)
    return found.pop() if len(found) == 1 else UNLABELLED


def novel_verdict(field: str, gold: str, got: Any) -> str | None:
    """'right' / 'wrong value' / 'null' against a silver label; None if unlabelled."""
    if gold == UNLABELLED:
        return None
    if got is None:
        return "null"
    if field == "gst_id":
        ok = re.sub(r"\D", "", str(got)) == gold
    else:
        ok = gold in payment_categories(str(got))
    return "right" if ok else "wrong value"


def summarize(records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    """Every Phase 4 metric for one run, over marked records."""
    n = len(records)
    parsed = [r["parsed"] for r in records]
    keys = [(list(p) if p is not None else None, r["schema"])
            for p, r in zip(parsed, records)]
    per = _counts(records, fields)
    return {
        "receipts": n,
        "json_valid_raw": sum(r["valid_raw"] for r in records) / n,
        "repaired": sum(r["repaired"] for r in records) / n,
        "unusable": sum(p is None for p in parsed) / n,
        "schema_adherence": sum(k == s for k, s in keys) / n,
        "same_key_set": sum(k is not None and set(k) == set(s) for k, s in keys) / n,
        "out_of_schema_keys": sum(k is not None and bool(set(k) - set(s))
                                  for k, s in keys) / n,
        "missing_keys": sum(k is not None and bool(set(s) - set(k))
                            for k, s in keys) / n,
        "micro": _micro(records, fields),
        "ci95": bootstrap(records, fields),
        "fields": {f: _rates(c) for f, c in per.items()},
        "counts": per,
    }
