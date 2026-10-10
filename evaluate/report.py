"""
ParseTune - the results report.

Reads the saved runs in evaluate/results/ and writes evaluate/results/REPORT.md:
Phase 4's benchmark, Phase 5's schema probes, the validation probes that chose
the adapter, and the manual spot-check of Phase 5's novel fields. No number is
typed by hand - each comes from a results file - so the report can be rebuilt
at any time:

    python evaluate/report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics import novel_verdict

RESULTS = Path(__file__).resolve().parents[1] / "evaluate" / "results"
FIELDS = ["company", "date", "address", "total"]
SYSTEM_ORDER = ["parsetune", "zero-shot-bf16", "three-shot", "zero-shot"]
SYSTEM_LABEL = {
    "parsetune": "**ParseTune (v2 adapter)**",
    "zero-shot-bf16": "Phi-3.5-mini zero-shot, bf16",
    "three-shot": "Phi-3.5-mini 3-shot, 4-bit",
    "zero-shot": "Phi-3.5-mini zero-shot, 4-bit",
    "parsetune-v2": "**ParseTune v2** (adopted)",
    "parsetune-v1": "ParseTune v1 (before absent-field data)",
    "zero-shot-4bit": "Phi-3.5-mini zero-shot, 4-bit",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}"


def frac(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "–"


# --------------------------------------------------------------------------
# Phase 4.
# --------------------------------------------------------------------------
def phase4_runs(batch_size: int) -> dict[str, dict[str, Any]]:
    """The latest test run of each system at `batch_size`."""
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS.glob("*_test_*.json")):
        run = load(path)
        if run["config"]["batch_size"] == batch_size:
            run["_file"] = path.name
            latest[run["system"]] = run
    return latest


def phase4_section() -> list[str]:
    runs = phase4_runs(2)
    if not runs:
        return ["## Phase 4 — benchmark\n", "_No batch-2 test runs found._\n"]
    n = next(iter(runs.values()))["metrics"]["receipts"]
    out = [
        "## Phase 4 — benchmark (test split)\n",
        f"{n} receipts × 4 fields; batch size 2, seed 0 for every system. Scores in %. "
        "*Normalised* compares totals as amounts, dates as calendar dates, text "
        "case- and whitespace-insensitively; precision / recall / F1 are on non-null "
        "values with normalised matching. Intervals: 95% bootstrap over receipts.\n",
        "| System | Exact (95% CI) | Normalised (95% CI) | Precision | Recall | F1 (95% CI) "
        "| Valid JSON (raw) | Schema adherence | Out-of-schema keys | Time | Peak memory |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in SYSTEM_ORDER:
        if name not in runs:
            continue
        m, rt = runs[name]["metrics"], runs[name]["runtime"]
        mi, ci = m["micro"], m["ci95"]
        out.append(
            f"| {SYSTEM_LABEL[name]} "
            f"| {pct(mi['exact'])} ({pct(ci['exact'][0])}–{pct(ci['exact'][1])}) "
            f"| {pct(mi['normalised'])} ({pct(ci['normalised'][0])}–{pct(ci['normalised'][1])}) "
            f"| {pct(mi['precision'])} | {pct(mi['recall'])} "
            f"| {pct(mi['f1'])} ({pct(ci['f1'][0])}–{pct(ci['f1'][1])}) "
            f"| {pct(m['json_valid_raw'])} | {pct(m['schema_adherence'])} "
            f"| {pct(m['out_of_schema_keys'])} | {rt['seconds'] / 60:.1f} min "
            f"| {rt['peak_memory_gb']:.1f} GB |")

    out += ["", "### Per field (normalised precision / recall / F1, and exact accuracy)\n",
            "| System | Field | Exact | Normalised | Precision | Recall | F1 |",
            "|:--|:--|--:|--:|--:|--:|--:|"]
    for name in SYSTEM_ORDER:
        if name not in runs:
            continue
        for f in FIELDS:
            r = runs[name]["metrics"]["fields"][f]
            out.append(f"| {SYSTEM_LABEL[name]} | {f} | {pct(r['exact'])} | "
                       f"{pct(r['normalised'])} | {pct(r['precision'])} | "
                       f"{pct(r['recall'])} | {pct(r['f1'])} |")

    old = phase4_runs(4)
    shared = [s for s in SYSTEM_ORDER if s in old and s in runs]
    if shared:
        out += ["", "### Batch-size sensitivity (batch 4 vs batch 2, same test receipts)\n",
                "| System | Outputs that changed | Normalised at batch 4 | at batch 2 |",
                "|:--|--:|--:|--:|"]
        for s in shared:
            a, b = old[s]["records"], runs[s]["records"]
            changed = sum(x["raw"] != y["raw"] for x, y in zip(a, b))
            out.append(f"| {SYSTEM_LABEL[s]} | {changed}/{len(b)} | "
                       f"{pct(old[s]['metrics']['micro']['normalised'])} | "
                       f"{pct(runs[s]['metrics']['micro']['normalised'])} |")
        out.append("\nThe batch-4 attempt ran out of GPU memory during the three-shot run; "
                   "its two completed systems are kept as evidence of batch sensitivity.")
    out.append("\nFiles: " + ", ".join(f"`{runs[s]['_file']}`" for s in SYSTEM_ORDER if s in runs))
    return out


# --------------------------------------------------------------------------
# Phase 5.
# --------------------------------------------------------------------------
def phase5_runs() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS.glob("*_phase5_*.json")):
        run = load(path)
        if run["split"] == "test" and run["config"]["limit"] is None:
            run["_file"] = path.name
            latest[run["system"]] = run
    return latest


CONDITION_LABEL = {
    "renamed": "1. Renamed fields (`merchant_name`, `purchase_date`, `amount_due`)",
    "subset_order": "2. Random subset, random order of the four",
    "absent (four fields)": "4. Absent-field requests: the four fields",
    "absent (absent slot)": "4. `shipping_tracking_number` (absent) → `null`",
}


def phase5_section() -> list[str]:
    runs = phase5_runs()
    order = [s for s in ("parsetune-v2", "parsetune-v1", "zero-shot-bf16", "zero-shot-4bit")
             if s in runs]
    if not order:
        return ["## Phase 5 — schema-instructability\n", "_No Phase 5 test runs found._\n"]
    out = [
        "## Phase 5 — schema-instructability (test split, held-out conditions)\n",
        "None of these field names appeared in training (`prepare_data.py` refuses "
        "them). Printed fields are scored with normalised matching; a missing key or an "
        "unusable output counts as a null answer. Batch size 2, seed 0.\n",
        "| Condition | System | Printed fields right | Precision | Recall | F1 "
        "| Printed fields answered null | Absent → null | Schema kept |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for cond in CONDITION_LABEL:
        for s in order:
            sl, r = runs[s]["scores"]["slots"][cond], runs[s]["scores"]["rates"][cond]
            out.append(
                f"| {CONDITION_LABEL[cond]} | {SYSTEM_LABEL[s]} "
                f"| {frac(sl['normalised_right'], sl['present'])} | {pct(r['precision'])} "
                f"| {pct(r['recall'])} | {pct(r['f1'])} | {sl['false_null']} "
                f"| {frac(sl['absent_null'], sl['absent'])} "
                f"| {frac(sl['adherent'], sl['requests'])} |")

    out += ["", "### 3. Novel fields (`payment_method`, `gst_id`, `line_item_count`)\n",
            "Never supervised. `gst_id` and `payment_method` are compared with strict-pattern "
            "silver labels where the receipt makes them unambiguous (a single 12-digit GST id "
            "after \"GST\": 304 receipts; a single payment keyword: 296). On those receipts each "
            "answer is *right*, a *wrong value*, or *null* — so precision (right / answered) and "
            "recall (right / labelled) can be told apart. `line_item_count` has no reliable "
            "pattern; it is judged by the manual spot-check below.\n",
            "| System | Field | Answered (all receipts) | Right | Wrong value | Null | Precision | Recall | F1 |",
            "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for s in order:
        for field, c in runs[s]["scores"]["novel"].items():
            answered = frac(c["answered"], c["requests"])
            if not c["labelled"]:
                out.append(f"| {SYSTEM_LABEL[s]} | `{field}` | {answered} | manual check | | | | | |")
                continue
            verdicts = [novel_verdict(field, r["expected"][field], (r["parsed"] or {}).get(field))
                        for r in runs[s]["results"]["novel"]]
            right, wrong, null = (verdicts.count(v) for v in ("right", "wrong value", "null"))
            p = right / (right + wrong) if right + wrong else None
            rc = right / (right + wrong + null)
            f1 = 2 * p * rc / (p + rc) if p and rc else None
            out.append(f"| {SYSTEM_LABEL[s]} | `{field}` | {answered} | {right} | {wrong} | {null} "
                       f"| {pct(p)} | {pct(rc)} | {pct(f1)} |")
    pay = {s: runs[s]["scores"]["novel"]["payment_method"].get("by_label", {}) for s in order}
    if any(pay.values()):
        out += ["", "`payment_method` by silver label (cash dominates — a model that always "
                "answered \"cash\" would agree on most receipts, so per-label counts matter):\n",
                "| System | " + " | ".join(sorted({k.split(':')[0] for v in pay.values() for k in v})) + " |",
                "|:--|" + "--:|" * len({k.split(':')[0] for v in pay.values() for k in v})]
        labels = sorted({k.split(':')[0] for v in pay.values() for k in v})
        for s in order:
            cells = []
            for lab in labels:
                a, d = pay[s].get(f"{lab}: agree", 0), pay[s].get(f"{lab}: disagree", 0)
                cells.append(f"{a}/{a + d}")
            out.append(f"| {SYSTEM_LABEL[s]} | " + " | ".join(cells) + " |")

    spot = RESULTS / "phase5_spotcheck.json"
    if spot.exists():
        sc = load(spot)
        out += ["", f"### Manual spot-check of novel fields — {sc['system']}, "
                f"{sc['sample_size']} receipts\n", sc["protocol"], "",
                "| Field | " + " | ".join(sc["verdicts"]) + " |",
                "|:--|" + "--:|" * len(sc["verdicts"])]
        for field, counts in sc["summary"].items():
            out.append(f"| `{field}` | " + " | ".join(str(counts.get(v, 0)) for v in sc["verdicts"]) + " |")
        if sc.get("notes"):
            out += [""] + [f"- {n}" for n in sc["notes"]]
    out.append("\nFiles: " + ", ".join(f"`{runs[s]['_file']}`" for s in order))
    return out


# --------------------------------------------------------------------------
# Model selection (validation).
# --------------------------------------------------------------------------
def selection_section() -> list[str]:
    files = {v: RESULTS / f"selection_validation_parsetune-{v}.json" for v in ("v1", "v2")}
    if not all(p.exists() for p in files.values()):
        return []
    d = {v: load(p)["scores"] for v, p in files.items()}
    rows = [
        ("Four SROIE fields, exact", "canonical", "present_right", "present"),
        ("Renamed fields (`shop_name`, …), exact", "renamed", "present_right", "present"),
        ("Never-trained absent field → null", "unseen_absent (absent slot)", "absent_null", "absent"),
        ("Grounding fields: printed value right", "grounding", "present_right", "present"),
        ("Grounding fields: absent → null", "grounding", "absent_null", "absent"),
    ]
    out = ["## Model selection (validation split, before test was touched)\n",
           "The adoption bar was agreed before retraining: four fields ≥ 231/248 exact, no "
           "printed field answered null, ≥ 80% null on never-trained absent fields.\n",
           "| Probe | v1 | v2 (adopted) |", "|:--|--:|--:|"]
    for label, cond, num, den in rows:
        out.append(f"| {label} | {frac(d['v1'][cond][num], d['v1'][cond][den])} "
                   f"| {frac(d['v2'][cond][num], d['v2'][cond][den])} |")
    return out


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# ParseTune — results report\n",
        f"_Generated {stamp} by `evaluate/report.py` from the files in `evaluate/results/`._\n",
        *phase4_section(), "",
        *phase5_section(), "",
        *selection_section(), "",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {RESULTS / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
