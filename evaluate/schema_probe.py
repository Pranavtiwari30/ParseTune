"""
ParseTune - schema probes: does the model follow the schema it is handed?

Two condition sets, sharing one scoring path through predict.py.

`--conditions selection` (validation split) chose between adapters before
the test set was touched:

    canonical      SROIE's four fields                exact match with gold
    unseen_absent  the four + one field no receipt    answered null?
                   carries, never trained on
    renamed        SROIE fields under names no        exact match; a null here
                   training example uses              is a printed field missed
    grounding      time / phone / fax / email where   value or null, as the
                   the strict patterns label them     pattern labelled it

    Adoption bar, agreed before retraining: the four fields stay within noise
    (at least 231 of 248 exact on the full split), no printed field is answered
    null (canonical and renamed), and unseen absent fields are answered null at
    least 80% of the time.

`--conditions phase5` (test split) is Phase 5 itself - the README's four
held-out conditions, whose names training was forbidden to use:

    renamed        merchant_name / purchase_date / amount_due, each with its
                   SROIE field's own description - the description is the
                   only bridge to what was trained
    subset_order   a random subset of the four, in a random order (the sanity
                   control: train augmentation already covers it)
    novel          payment_method / gst_id / line_item_count - printed on
                   receipts, never supervised. gst_id and payment_method are
                   checked against strict-pattern silver labels where those
                   exist; line_item_count only by manual spot-check
    absent         the four + shipping_tracking_number, which no receipt
                   carries (receipts mentioning tracking or couriers skipped)

Usage
-----
    python evaluate/schema_probe.py --adapter train/adapters/phi35-sroie-v2
    python evaluate/schema_probe.py --conditions phase5 --name parsetune-v2 \
        --adapter train/adapters/phi35-sroie-v2
    python evaluate/schema_probe.py --conditions phase5 --name zero-shot-bf16 \
        --model microsoft/Phi-3.5-mini-instruct
"""

from __future__ import annotations

import os

# The bf16 base comes from the local Hugging Face cache; never reach for the
# network mid-run. Must precede the mlx_lm import inside predict.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from metrics import (  # noqa: E402
    UNLABELLED, matches, novel_verdict, silver_gst, silver_payment,
)
from predict import (  # noqa: E402
    DATA_DIR, MODEL_PATH, Progress, load_model, load_template, predict,
    project_relative,
)
from prepare_data import (  # noqa: E402
    FIELD_SPECS, GROUNDING, NEVER_PRESENT, PHASE5_CONCEPTS, PHASE5_HELD_OUT,
    VALIDATION_ONLY_ABSENT, absent_on, grounding_value,
)

RESULTS_DIR = PROJECT_ROOT / "evaluate" / "results"

# --------------------------------------------------------------------------
# 0. The fields each condition set asks for.
# --------------------------------------------------------------------------
# Selection: SROIE fields under names no training example uses - and not
# Phase 5's renamed fields either, which stay unseen until Phase 5.
RENAMED: dict[str, tuple[str, str]] = {
    # name: (source field, description)
    "shop_name":        ("company", "name of the shop that issued this receipt"),
    "transaction_date": ("date", "date on which the transaction took place"),
    "grand_total":      ("total", "grand total charged to the customer"),
}
_taken = {*FIELD_SPECS, *NEVER_PRESENT, *GROUNDING, *VALIDATION_ONLY_ABSENT,
          *PHASE5_HELD_OUT}
if set(RENAMED) & _taken or any(PHASE5_CONCEPTS.search(f"{n} {d}")
                                for n, (_s, d) in RENAMED.items()):
    raise SystemExit("[probe] a renamed field collides with a training or "
                     "Phase 5 field")

# Phase 5: the README's held-out names (prepare_data.PHASE5_HELD_OUT).
PHASE5_RENAMED = {"merchant_name": "company", "purchase_date": "date",
                  "amount_due": "total"}
PHASE5_NOVEL = {
    "payment_method":  "how the customer paid, such as cash or card",
    "gst_id":          "GST registration number of the merchant",
    "line_item_count": "number of different items listed on the receipt",
}
PHASE5_ABSENT = ("shipping_tracking_number",
                 "courier tracking number for the shipment",
                 r"TRACK|COURIER|\bAWB\b|CONSIGN|PARCEL|SHIPPING|POS\s*LAJU")
if not {*PHASE5_RENAMED, *PHASE5_NOVEL, PHASE5_ABSENT[0]} <= set(PHASE5_HELD_OUT):
    raise SystemExit("[probe] Phase 5 names drifted from prepare_data.PHASE5_HELD_OUT")

# The four fields' normalisers serve their Phase 5 renamings too.
KIND = {**{f: f for f in FIELD_SPECS}, **PHASE5_RENAMED}

# The bar for replacing the current adapter (selection only).
MIN_CANONICAL_EXACT = 231 / 248
MIN_ABSENT_NULL_RATE = 0.80

Job = tuple[str, dict[str, str], dict[str, Any]]   # text, schema, expected


# --------------------------------------------------------------------------
# 1. Building the requests.
# --------------------------------------------------------------------------
def build_selection(template, rows, rng: random.Random) -> dict[str, list[Job]]:
    """Model-selection probes: one request per validation receipt and condition."""
    four = dict(template.fields)
    absent_names = list(VALIDATION_ONLY_ABSENT)
    conditions: dict[str, list[Job]] = {
        "canonical": [], "unseen_absent": [], "renamed": [], "grounding": []}

    for i, row in enumerate(rows):
        text, gold = row["receipt_text"], row["target"]
        conditions["canonical"].append((text, four, dict(gold)))

        # The four plus one never-trained absent field, at a random position.
        name = absent_names[i % len(absent_names)]
        description, keywords = VALIDATION_ONLY_ABSENT[name]
        if absent_on(text, keywords):
            items = list(four.items())
            items.insert(rng.randint(0, len(items)), (name, description))
            conditions["unseen_absent"].append(
                (text, dict(items), {**gold, name: None}))

        conditions["renamed"].append((
            text,
            {n: d for n, (_src, d) in RENAMED.items()},
            {n: gold[src] for n, (src, _d) in RENAMED.items()},
        ))

        labelled = {}
        for field in GROUNDING:
            usable, value = grounding_value(text, field)
            if usable:
                labelled[field] = value
        if labelled:
            conditions["grounding"].append(
                (text, {f: GROUNDING[f][0] for f in labelled}, labelled))
    return conditions


def build_phase5(template, rows, rng: random.Random) -> dict[str, list[Job]]:
    """Phase 5's four held-out conditions, one request per receipt each."""
    four = dict(template.fields)
    name, description, keywords = PHASE5_ABSENT
    conditions: dict[str, list[Job]] = {
        "renamed": [], "subset_order": [], "novel": [], "absent": []}

    for row in rows:
        text, gold = row["receipt_text"], row["target"]
        conditions["renamed"].append((
            text,
            {n: four[src] for n, src in PHASE5_RENAMED.items()},
            {n: gold[src] for n, src in PHASE5_RENAMED.items()},
        ))

        fields = list(four)
        while fields == list(four):                 # never the canonical order
            fields = rng.sample(list(four), rng.randint(1, len(four)))
        conditions["subset_order"].append(
            (text, {f: four[f] for f in fields}, {f: gold[f] for f in fields}))

        conditions["novel"].append((text, dict(PHASE5_NOVEL), {
            "payment_method": silver_payment(text),
            "gst_id": silver_gst(text),
            "line_item_count": UNLABELLED,
        }))

        if absent_on(text, keywords):
            items = list(four.items())
            items.insert(rng.randint(0, len(items)), (name, description))
            conditions["absent"].append((text, dict(items), {**gold, name: None}))
    return conditions


# --------------------------------------------------------------------------
# 2. Scoring.
# --------------------------------------------------------------------------
def score_slots(
    jobs: list[Job],
    records: list[dict[str, Any]],
    only: set[str] | None = None,
) -> dict[str, int]:
    """
    Count answers by kind over the fields of a condition.

    Present fields are scored exact and normalised (by the field's kind); a
    missing key, or an unusable output, counts as a null answer - either way
    the printed value was not returned. `pred` counts every non-null answer,
    including values given for absent fields, so precision charges for them.
    """
    s = Counter(present=0, present_right=0, normalised_right=0, false_null=0,
                absent=0, absent_null=0, pred=0, requests=0, valid_raw=0,
                unusable=0, adherent=0, out_of_schema=0)
    for (_text, schema, expected), record in zip(jobs, records):
        parsed = record["parsed"]
        s["requests"] += 1
        s["valid_raw"] += record["valid_raw"]
        s["unusable"] += parsed is None
        s["adherent"] += parsed is not None and list(parsed) == list(schema)
        s["out_of_schema"] += parsed is not None and bool(set(parsed) - set(schema))
        for field, gold in expected.items():
            if (only is not None and field not in only) or gold == UNLABELLED:
                continue
            got = (parsed or {}).get(field)
            s["pred"] += got is not None
            if gold is None:
                s["absent"] += 1
                s["absent_null"] += got is None
            else:
                s["present"] += 1
                s["present_right"] += got == gold
                s["normalised_right"] += matches(KIND.get(field, field), gold, got, True)
                s["false_null"] += got is None
    return dict(s)


def rates(s: dict[str, int]) -> dict[str, float | None]:
    """Precision / recall / F1 (normalised) and the null rates of a tally."""
    def div(a, b):
        return a / b if b else None
    p, r = div(s["normalised_right"], s["pred"]), div(s["normalised_right"], s["present"])
    return {
        "exact": div(s["present_right"], s["present"]),
        "normalised": r,
        "precision": p,
        "recall": r,
        "f1": None if p is None or r is None else (2 * p * r / (p + r) if p + r else 0.0),
        "absent_null_rate": div(s["absent_null"], s["absent"]),
        "false_null_rate": div(s["false_null"], s["present"]),
        "schema_adherence": div(s["adherent"], s["requests"]),
        "json_valid_raw": div(s["valid_raw"], s["requests"]),
        "out_of_schema": div(s["out_of_schema"], s["requests"]),
    }


def score_novel(jobs: list[Job], records: list[dict[str, Any]]) -> dict[str, dict]:
    """
    Novel fields against their silver labels, where those exist: gst_id by its
    digits, payment_method by the category named in the answer
    (metrics.novel_verdict). Counts right answers, wrong values and nulls on
    labelled receipts - so precision and recall can be told apart - and how
    often the model answered at all.
    """
    out = {}
    for field in PHASE5_NOVEL:
        c = Counter(requests=0, answered=0, labelled=0, agree=0, wrong=0, null=0)
        by_label: Counter = Counter()
        for (_t, _s, expected), record in zip(jobs, records):
            got = (record["parsed"] or {}).get(field)
            c["requests"] += 1
            c["answered"] += got is not None
            verdict = novel_verdict(field, expected[field], got)
            if verdict is None:
                continue
            c["labelled"] += 1
            c["agree"] += verdict == "right"
            c["wrong"] += verdict == "wrong value"
            c["null"] += verdict == "null"
            by_label[f"{expected[field]}: {'agree' if verdict == 'right' else 'disagree'}"] += 1
        out[field] = {**c, "by_label": dict(by_label)} if field == "payment_method" else dict(c)
    return out


# --------------------------------------------------------------------------
# 3. Reporting.
# --------------------------------------------------------------------------
def pct(a, b) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "-"


def report_selection(scores: dict[str, dict[str, int]]) -> bool:
    """Print the selection table and the adoption verdict; True if met."""
    print(f"\n{'condition':28s} {'printed fields right':>22s} "
          f"{'answered null':>14s} {'absent -> null':>18s}")
    for name, s in scores.items():
        print(f"{name:28s} {pct(s['present_right'], s['present']):>22s} "
              f"{s['false_null']:>14d} {pct(s['absent_null'], s['absent']):>18s}")

    canon, renamed, absent = (scores["canonical"], scores["renamed"],
                              scores["unseen_absent (absent slot)"])
    need = math.ceil(MIN_CANONICAL_EXACT * canon["present"])
    false_nulls = canon["false_null"] + renamed["false_null"]
    null_rate = absent["absent_null"] / absent["absent"] if absent["absent"] else 0
    checks = [
        (f"four fields exact >= {need}/{canon['present']}",
         canon["present_right"] >= need),
        ("no printed field answered null (canonical + renamed)", false_nulls == 0),
        (f"unseen absent fields answered null >= {MIN_ABSENT_NULL_RATE:.0%}",
         null_rate >= MIN_ABSENT_NULL_RATE),
    ]
    print("\n[adoption bar]")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    met = all(ok for _label, ok in checks)
    print(f"  => {'meets' if met else 'does NOT meet'} the bar")
    return met


def report_phase5(name: str, scores: dict[str, Any]) -> None:
    """Print Phase 5's table for one system."""
    print(f"\n[{name}] {'condition':24s} {'printed right (norm.)':>24s} "
          f"{'false nulls':>11s} {'absent -> null':>16s} {'schema kept':>14s}")
    for cond, s in scores["slots"].items():
        print(f"[{name}] {cond:24s} {pct(s['normalised_right'], s['present']):>24s} "
              f"{s['false_null']:>11d} {pct(s['absent_null'], s['absent']):>16s} "
              f"{pct(s['adherent'], s['requests']):>14s}")
    for field, c in scores["novel"].items():
        agree = pct(c["agree"], c["labelled"]) if c["labelled"] else "manual check"
        print(f"[{name}] novel {field:18s} answered {pct(c['answered'], c['requests'])}, "
              f"silver agreement {agree}")


# --------------------------------------------------------------------------
# 4. Entry point.
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe schema-following: adapter selection or Phase 5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--conditions", choices=["selection", "phase5"], default="selection")
    p.add_argument("--split", choices=["validation", "test"], default=None,
                   help="Defaults to validation for selection, test for phase5.")
    p.add_argument("--adapter", type=Path, default=None,
                   help="LoRA adapter directory; omit for a bare base model.")
    p.add_argument("--model", type=Path, default=MODEL_PATH,
                   help="Base model: the MLX 4-bit copy, or a Hub id for bf16.")
    p.add_argument("--name", default=None, help="System label for phase5 results.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N receipts.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Fixed and recorded; defaults: 8 for selection (as the "
                        "recorded probes), 2 for phase5 (as Phase 4).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random positions and subsets in the requests.")
    p.add_argument("--out", type=Path, default=None,
                   help="Selection: write scores and every answer to this JSON.")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR,
                   help="Phase 5: where the timestamped results JSON goes.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phase5 = args.conditions == "phase5"
    split = args.split or ("test" if phase5 else "validation")
    batch = args.batch_size or (2 if phase5 else 8)
    name = args.name or (args.adapter.name if args.adapter else f"base-{Path(str(args.model)).name}")

    template = load_template()
    rows = json.loads((DATA_DIR / f"{split}.json").read_text(encoding="utf-8"))
    rows = rows[: args.limit] if args.limit else rows
    build = build_phase5 if phase5 else build_selection
    conditions = build(template, rows, random.Random(args.seed))
    flat = [job for jobs in conditions.values() for job in jobs]
    print(f"[{name}] {args.conditions} on {len(rows)} {split} receipts: "
          f"{len(flat)} requests, batch {batch}", flush=True)

    tic = time.perf_counter()
    model, tokenizer = load_model(args.model, args.adapter)
    records, stats = predict(model, tokenizer, template,
                             [(text, schema) for text, schema, _exp in flat],
                             batch, progress=Progress(name, "requests"))
    seconds = time.perf_counter() - tic

    results: dict[str, list[dict[str, Any]]] = {}
    slot_scores: dict[str, dict[str, int]] = {}
    start = 0
    for cond, jobs in conditions.items():
        recs = records[start:start + len(jobs)]
        start += len(jobs)
        results[cond] = [{**r, "expected": exp} for r, (_t, _s, exp) in zip(recs, jobs)]
        if cond == "unseen_absent":
            slot_scores["unseen_absent (four fields)"] = score_slots(
                jobs, recs, only=set(template.fields))
            slot_scores["unseen_absent (absent slot)"] = score_slots(
                jobs, recs, only=set(VALIDATION_ONLY_ABSENT))
        elif cond == "absent":
            slot_scores["absent (four fields)"] = score_slots(
                jobs, recs, only=set(template.fields))
            slot_scores["absent (absent slot)"] = score_slots(
                jobs, recs, only={PHASE5_ABSENT[0]})
        elif cond != "novel":
            slot_scores[cond] = score_slots(jobs, recs)

    if not phase5:
        print(f"[probe] adapter: {args.adapter or 'none (bare base)'}; "
              f"{len(rows)} {split} receipts; {len(flat)} requests; batch size {batch}")
        met = report_selection(slot_scores)
        if args.out:
            args.out.write_text(json.dumps({
                "adapter": project_relative(args.adapter) if args.adapter else None,
                "batch_size": batch, "seed": args.seed,
                "receipts": len(rows), "scores": slot_scores, "meets_bar": met,
                "results": results,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[probe] wrote {args.out}")
        return 0

    scores = {
        "slots": slot_scores,
        "rates": {cond: rates(s) for cond, s in slot_scores.items()},
        "novel": score_novel(conditions["novel"], results["novel"]),
    }
    report_phase5(name, scores)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{stamp}_phase5_{name}.json"
    path.write_text(json.dumps({
        "system": name, "split": split, "created_utc": stamp,
        "config": {"model": project_relative(args.model),
                   "adapter": project_relative(args.adapter) if args.adapter else None,
                   "batch_size": batch, "seed": args.seed, "limit": args.limit},
        "runtime": {"seconds": round(seconds, 1),
                    "peak_memory_gb": round(stats.peak_memory, 2)},
        "scores": scores,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] phase5 done in {seconds / 60:.1f} min - wrote {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
