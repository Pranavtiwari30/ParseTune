"""
ParseTune - regression tests for the pieces every number depends on.

    python tests/test_pipeline.py          # plain asserts, no extra dependencies
    python -m pytest tests/                # also works, if pytest is installed

Covers: Phase 4's normalisers and metrics, Phase 3's JSON repair and token
budget, the prompt contract (every stored prompt rebuilds byte for byte), and
the data invariants (no receipt in two splits, answers in schema order,
absent-field examples built only from real text, Phase 5 names never trained).
CPU only - nothing here touches the GPU.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evaluate"), str(ROOT / "data")]

from metrics import mark, matches, summarize  # noqa: E402
from predict import check_template, extract_json, load_template, token_budget  # noqa: E402
from prepare_data import (  # noqa: E402
    FIELD_SPECS, GROUNDING, NEVER_PRESENT, PHASE5_HELD_OUT, VALIDATION_ONLY_ABSENT,
    check_field_pools, grounding_value,
)

SPLITS = {s: json.loads((ROOT / f"data/processed/{s}.json").read_text(encoding="utf-8"))
          for s in ("train", "validation", "test")}


# --------------------------------------------------------------------------
# Phase 4: normalisers and metrics.
# --------------------------------------------------------------------------
def test_normalised_match() -> None:
    cases = [
        ("total", "RM 4.5000", "4.50", True), ("total", "RM20,80", "20.80", True),
        ("total", "1,234.50", "1234.5", True), ("total", "0021151.00", "51.00", False),
        ("total", "45.00", "5.00", False), ("total", "$0.39", "0.39", True),
        ("date", "21-05-17", "21/05/2017", True), ("date", "Date:11/03/18", "11/03/18", True),
        ("date", "2018-04-06", "06/04/2018", True), ("date", "27 May 2018", "27/05/2018", True),
        ("date", "14/May/2017", "14-05-17", True), ("date", "09-22-2018", "22/09/2018", True),
        ("date", "May 27, 2018", "27-05-2018", True), ("date", "06/07/2017", "07/06/2017", False),
        ("date", "15-06-2ized", "15-06-2018", False),
        ("company", "BEMED (SP) SON.", "bemed  (sp) son", True),
        ("company", "MYNEWS RETAIL SB(306295-X)", "MYNEWS RETAIL SB", False),
        ("total", None, None, True), ("total", "4.50", None, False),
    ]
    for field, gold, pred, want in cases:
        assert matches(field, gold, pred, True) is want, (field, gold, pred)


def test_summary_counts() -> None:
    records = [
        {"schema": ["company", "total"], "target": {"company": "A", "total": "RM 4.50"},
         "parsed": {"company": "a", "total": "4.50"}, "valid_raw": True, "repaired": False},
        {"schema": ["company", "total"], "target": {"company": "B", "total": "9.00"},
         "parsed": {"total": "9.00", "company": None, "extra": 1},
         "valid_raw": True, "repaired": False},
    ]
    mark(records)
    s = summarize(records, ["company", "total"])
    assert s["micro"]["exact"] == 0.25 and s["micro"]["normalised"] == 0.75
    assert s["micro"]["precision"] == 1.0 and s["micro"]["recall"] == 0.75
    assert s["schema_adherence"] == 0.5 and s["out_of_schema_keys"] == 0.5


# --------------------------------------------------------------------------
# Phase 3: JSON repair, token budget, prompt contract.
# --------------------------------------------------------------------------
def test_json_repair() -> None:
    cases = {
        '{"company": "A", "total": "4.50"}': ({"company": "A", "total": "4.50"}, False),
        '```json\n{"company": "A"}\n```': ({"company": "A"}, True),
        '{"company": "A"}\nThe company is A.': ({"company": "A"}, True),
        'Here is the JSON: {"company": "A"}': ({"company": "A"}, True),
        '{"company": "A}B", "total": null}': ({"company": "A}B", "total": None}, False),
        '{"company": "SANYU STAT': ({"company": "SANYU STAT"}, True),
        '{"company": "A", ': ({"company": "A"}, True),
        "I cannot find these fields.": (None, True),
        '["A", "B"]': (None, True),
    }
    for text, want in cases.items():
        assert extract_json(text) == want, text


def test_token_budget_covers_every_gold_answer() -> None:
    """Every gold answer fits in at most two-thirds of its schema's budget."""
    model_dir = ROOT / "models" / "phi-3.5-mini-instruct-4bit"
    if not (model_dir / "tokenizer.json").exists():
        print("    (skipped: converted model not present - run train/sft_train.py)")
        return
    from mlx_lm.utils import load_tokenizer
    tokenizer = load_tokenizer(model_dir)
    longest: dict[int, int] = {}
    for rows in SPLITS.values():
        for r in rows:
            n = len(tokenizer.encode(r["completion"][0]["content"], add_special_tokens=False))
            longest[len(r["schema"])] = max(longest.get(len(r["schema"]), 0), n)
    for size, tokens in longest.items():
        assert tokens <= 2 / 3 * token_budget(dict.fromkeys(range(size))), (size, tokens)


def test_every_prompt_rebuilds_byte_for_byte() -> None:
    template = load_template()
    for rows in SPLITS.values():
        check_template(template, rows)          # raises SystemExit on any drift


# --------------------------------------------------------------------------
# Phase 1: data invariants.
# --------------------------------------------------------------------------
def test_no_receipt_in_two_splits() -> None:
    ids = {s: {r["receipt_id"] for r in rows} for s, rows in SPLITS.items()}
    texts = {s: {r["receipt_text"] for r in rows} for s, rows in SPLITS.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not ids[a] & ids[b] and not texts[a] & texts[b], (a, b)


def test_answers_match_targets_in_schema_order() -> None:
    for rows in SPLITS.values():
        for r in rows:
            out = json.loads(r["completion"][0]["content"])
            assert out == r["target"] and list(out) == r["schema"], r["id"]
            if "schema_spec" in r:
                assert list(r["schema_spec"]) == r["schema"], r["id"]


def test_absent_field_examples_use_only_real_text() -> None:
    v2 = [r for r in SPLITS["train"] if r["id"].endswith("-v2")]
    assert v2, "no absent-field examples found"
    for r in v2:
        for field, value in r["target"].items():
            if field in NEVER_PRESENT:
                assert value is None
                assert not re.search(NEVER_PRESENT[field][1], r["receipt_text"], re.I)
            elif field in GROUNDING:
                usable, label = grounding_value(r["receipt_text"], field)
                assert usable and label == value, (r["id"], field)
                assert value is None or value in r["receipt_text"]
            else:
                assert field in FIELD_SPECS and value is not None, (r["id"], field)


def test_held_out_names_never_trained() -> None:
    check_field_pools()
    trained = {f for r in SPLITS["train"] for f in r["schema"]}
    assert not trained & set(PHASE5_HELD_OUT)
    assert not trained & set(VALIDATION_ONLY_ABSENT)
    assert {r["split"] for r in SPLITS["test"]} == {"test"}
    assert all(r["schema"] == list(FIELD_SPECS) for r in SPLITS["test"] + SPLITS["validation"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
