# ParseTune — project memory

The project in one place: what it is, how it is built, what happened in each phase, where every
artifact lives, and what is still open. The *why* behind each choice is in `decisions.md`
(referenced here as D-nn); the full numbers are in `evaluate/results/REPORT.md`.

---

## 1. Scope

**ParseTune** fine-tunes Phi-3.5-mini-instruct (3.8B) to do *schema-instructable* document
extraction: given receipt text and a schema `{field: description}`, return a JSON object with
exactly those keys, in that order, values copied verbatim from the receipt, and `null` for
anything the receipt does not contain.

- **In scope:** SROIE receipts (Malaysian retail, English, 2017–2018), OCR *text* input (not
  images), the four SROIE fields for supervision, schema-following beyond them, honest evaluation
  against baselines, local training on Apple Silicon.
- **Out of scope:** OCR itself, images and layout models, other languages or document types,
  CUDA training (moved off by D-10), publishing to the Hugging Face Hub (prepared, left to the
  user — D-35).

## 2. Architecture

```
SROIE (HF parquet mirror)
   │  data/prepare_data.py — decode BIO tags → values; rebuild printed lines from OCR boxes;
   │  dedup across splits; schema augmentation; absent-field examples (train only)
   ▼
data/processed/{train,validation,test}.json + dataset_info.json   (the prompt contract)
   │  train/sft_train.py — convert base to MLX 4-bit (once); export chat JSONL;
   │  prove the loss mask; QLoRA with mlx-lm; reload + one extraction
   ▼
train/adapters/phi35-sroie-v2/adapters.safetensors   (adopted; v1 kept beside it)
   │  evaluate/predict.py — same prompt (proven byte-for-byte), greedy, 3 stop conditions,
   │  JSON repair flagged, continuous batching
   ▼
evaluate/benchmark.py + metrics.py   → Phase 4 results (test)
evaluate/schema_probe.py             → model selection (validation) and Phase 5 (test)
evaluate/report.py                   → evaluate/results/REPORT.md
notebooks/parsetune.ipynb            → walkthrough
```

| Component | Role | Key facts |
|:--|:--|:--|
| Base model | `microsoft/Phi-3.5-mini-instruct` → MLX 4-bit affine, group 64 (`models/`, 2.0 GB) | converted locally (D-11); bf16 weights stay in the HF cache for baseline 0b |
| Adapter | LoRA r16, scale 2.0, dropout 0.05, all 32 blocks, qkv/o/gate_up/down | 25.17M trainable (0.66%) |
| Training | mlx-lm 0.32.0 driven from Python; batch 2 × 4; lr 2e-4 cosine, 3% warmup; 3 epochs; grad checkpointing | D-12…D-17 |
| Prompt | system prompt + instruction template with the schema JSON; read from `dataset_info.json` | identical at training and inference (D-19) |
| Inference | greedy; stop at end-of-turn, budget 48 + 80/field, or closing brace; batch fixed per run | D-20…D-23 |

## 3. Environment

- MacBook Pro, Apple **M5 Pro** (16-core GPU), **24 GB** unified memory, macOS 26.6.
- Python 3.12 venv at `.venv` (made with `uv`; system Python is 3.9).
- `data/requirements.txt` (datasets) · `train/requirements.txt` (mlx-lm[train]==0.32.0, mlx==0.32.3).
- **One GPU job at a time** — training plus inference together ran out of GPU memory. **Long runs
  under `caffeinate`** — the Mac sleeps otherwise and the run stalls.

## 4. Phases

### Phase 1 — Data · done
- Built before 2026-09-02; revisited 2026-10-08.
- 557 train receipts → **1,671 examples** (1,114 four-field + 557 absent-field mixes), 62
  validation, 347 test (official split, untouched).
- 7 train receipts that duplicated test receipts dropped (D-05).
- Absent-field examples from real receipts only: 495 `null` answers, 251 real grounding values;
  hand audit fixed one systematic flaw (D-07, D-08).
- Invariants checked: completions parse and match targets in schema order, every prompt rebuilds
  byte for byte, no receipt in two splits by id or by text, Phase 5 names never trained on (D-09).

### Phase 2 — Training · done
- **v1** (four-field data): 1,671 iterations, 76 min, val loss 0.626 → 0.019.
  - First attempt died at iteration ~90 (GPU out of memory, inference running alongside) and was
    restarted.
- **v2** (with absent-field data): 2,505 iterations, 2 h 15 min, peak 5.6 GB, val loss
  0.626 → 0.021 (best 0.019).
- **v2 adopted** on validation against a bar agreed in advance: four fields 234/248 (= v1), 0 false
  nulls, never-trained absent fields → `null` 62/62 (v1: 0/62) (D-18).

### Phase 3 — Inference · done
- `predict.py`, used by every evaluation.
- Validation spot check: v2 62/62 valid JSON, 234/248 fields exact; bare 4-bit base 54/62 and
  83/248.
- Found the 4-bit `2ized` year glitch (quantisation, not batching) → baseline 0b (D-24, D-25).

### Phase 4 — Evaluation · done
Test, 347 receipts, batch 2, normalised match (95% CI):

| System | Normalised | Exact | F1 |
|:--|--:|--:|--:|
| **ParseTune v2** | **94.2 (92.9–95.5)** | 93.7 | 94.4 |
| zero-shot bf16 | 68.0 (66.0–70.2) | 55.0 | 68.0 |
| 3-shot 4-bit | 54.3 (51.5–56.8) | 41.4 | 55.2 |
| zero-shot 4-bit | 44.6 (41.7–47.6) | 33.6 | 47.4 |

- Fine-tuning gains +50 points on the same 4-bit weights, of which about 24 are quantisation
  repair. The fair headline is **+26 over the full-precision base**.
- ParseTune: 99.7% valid JSON as generated and the same schema adherence. Its 80 misses: 44 span
  boundaries (often gold noise), 32 different values, 4 from the one invalid output.
- The batch-4 attempt ran out of memory on three-shot (D-28). Its completed runs show ParseTune is
  batch-stable (3/347 outputs changed) where the base is not (80/347).

### Phase 5 — Generalisation · done
Held-out schema conditions on test (names never trained), four systems:

| Condition | v2 | v1 | base bf16 | base 4-bit |
|:--|--:|--:|--:|--:|
| Renamed fields right | **96.1** | 96.1 | 77.8 | 53.3 |
| Subset / order right (schema kept) | **94.1 (100)** | 94.3 (100) | 65.9 (81.8) | 46.6 (71.2) |
| Absent field → `null` | **100** | 0 | 96.5 | 92.8 |
| `gst_id` precision / recall | 100 / 89.8 | 75.0 / 75.0 | 99.0 / 99.0 | 98.7 / 98.7 |
| `payment_method` precision / recall | 83.7 / 74.3 | 82.1 / 82.1 | 99.0 / 99.0 | 99.0 / 98.6 |

- **What holds:** the instruction generalises across *names and shapes* of the trained concepts.
- **What doesn't:** it does not generalise to *new concepts*. There, fine-tuning narrowed the
  model: the 4-bit base equals the bf16 base, so this is the fine-tune and not quantisation.
- **Absent-field data:** restored the `null` answers the four-field fine-tune had erased.
- **Spot-check:** 50 receipts reviewed by the AI assistant, not a human. `gst_id` 42/50 right
  with 0 wrong values; `payment_method` 33/50; `line_item_count` 33/50.
- **Decisions:** v2 kept despite the `payment_method` regression (D-38); the 4-bit base was added
  to Phase 5 (D-39).

### Phase 6 — Packaging · done (publishing left to the user)
- Walkthrough notebook (Apple Silicon), model card and release command, results report, README
  results.
- Hub upload and PEFT conversion deliberately not done (D-35).

## 5. Where things live

| Path | What | In git? |
|:--|:--|:--|
| `data/processed/*.json`, `dataset_info.json` | the dataset and its prompt contract | yes |
| `train/config.yaml` | training configuration (adapter path → v2) | yes |
| `models/phi-3.5-mini-instruct-4bit/` | converted base model, 2.0 GB | no (rebuilt by `sft_train.py`) |
| `train/adapters/phi35-sroie/` | v1 adapter + checkpoints | no (96 MB per file; D-40) |
| `train/adapters/phi35-sroie-v2/` | **v2 adapter (adopted)** + checkpoints | no (the only copy is on this Mac; D-35, D-40) |
| `train/adapters/phi35-sroie*/run_log.json`, `adapter_config.json` | loss curves and every training setting, both runs | yes |
| `train/data/` | exported chat JSONL, rebuilt every run | no |
| `evaluate/results/*_test_*.json` | Phase 4 runs, every prediction kept | yes |
| `evaluate/results/*_phase5_*.json` | Phase 5 runs (v2, v1, bf16 base, 4-bit base) | yes |
| `evaluate/results/phase5_spotcheck.json` | the 150 manual verdicts (AI-reviewed) behind the spot-check | yes |
| `evaluate/results/selection_validation_*.json` | the probes that chose v2 | yes |
| `evaluate/results/REPORT.md`, `comparison_test.md` | generated reports (`python evaluate/report.py`) | yes |
| `notebooks/parsetune.ipynb` | walkthrough, saved with its outputs | yes |
| `MODEL_CARD.md` | adapter card + publishing commands | yes |
| `tests/test_pipeline.py` | 9 regression tests (`python tests/test_pipeline.py`) | yes |
| `evaluate/results/benchmark-batch4-oom.log` | the batch-4 out-of-memory run the README cites (D-28) | yes |
| `evaluate/results/*.log`, `train/adapters/*/console*.log` | the other console logs | no |
| `~/.cache/huggingface/` | bf16 base weights (baseline 0b), SROIE download | no |

## 6. Reproduce

```bash
source .venv/bin/activate
python data/prepare_data.py                        # Phase 1 (Hub → data/processed)
caffeinate -i python train/sft_train.py            # Phase 2 (~2 h 15 min; converts the base once)
python evaluate/schema_probe.py --adapter train/adapters/phi35-sroie-v2       # selection probes
caffeinate -i python -u evaluate/benchmark.py > evaluate/results/benchmark.log 2>&1   # Phase 4
caffeinate -i python -u evaluate/schema_probe.py --conditions phase5 --name parsetune-v2 \
    --adapter train/adapters/phi35-sroie-v2                                     # Phase 5
python evaluate/report.py                          # REPORT.md from the saved runs
```

## 7. Open items and known limitations

- **Address and company boundaries** are the weakest fields (89.3% / 91.9%). Part of the gap is
  gold noise (a registration number at the start of an address), part is OCR line order: the
  rebuilt receipt can print address fragments in a different order from the gold.
- **Totals:** the remaining errors take a subtotal or the pre-rounding amount (Malaysian receipts
  print both `42.88` and the rounded `42.90`), which makes the gold itself debatable.
- **Unescaped quotes:** a value copied from OCR text containing `"` breaks the JSON (1 of 347).
  Escaping inside values would need constrained decoding.
- **Absent fields:** the test probes one field name (`shipping_tracking_number`). Broader absent
  coverage needs more names or a second dataset (D-07 option d).
- **New concepts (the main open problem):** on fields unlike the four it was trained on, the
  adapter is worse than the base model (GST id recall 90% vs 99%, payment method 74% vs 99%). It
  copies nearby text such as "Cashier" where the base answers "Cash". Candidate fixes:
  - mix general instruction / extraction data into training;
  - supervise more field types the grounding way;
  - route new concepts to the base model.
  Any v3 must be chosen on validation and must swap trained fields out of Phase 5's novel set
  (D-38).
- **`payment_method` regression v1 → v2:** recall 82 → 74, reported rather than tuned on test.
- **Single dataset:** generalisation beyond Malaysian retail receipts is untested.
- **Not done on purpose:** Hub release (left to the user), PEFT conversion (needs validation).
  The commit and push are the user's to run (D-36); until the adapter is published, its weights
  exist only on this Mac.

## 8. Lessons recorded

1. Check leakage by *content*, not only by id (D-05).
2. Read the trainer's batching before choosing a batch size (D-13), and size schedules in the unit
   the optimiser steps in (D-14).
3. Measure memory before trading it for speed (D-15), and never share the GPU between a training
   run and an inference run.
4. A base-model baseline can be damaged by quantisation; compare with full precision too (D-25).
5. Fix and record the decoding batch size; base models flip near-ties under batching (D-23, D-28).
6. Agree the adoption bar before seeing the new model's numbers (D-18).
7. Long local runs need `caffeinate`, built into the command, not suggested (D-29).
8. Compare a fine-tune with its own base on concepts *outside* the training set: a narrow
   fine-tune can lose general ability the base had — here, answering `null` (lost by v1, restored
   by v2) and extracting new concepts (still lost) (D-39).
