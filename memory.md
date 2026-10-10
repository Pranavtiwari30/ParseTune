# ParseTune — project memory

The project in one place: what it is, how it is built, what happened in each phase, where every
artifact lives, and what is still open. The *why* behind each choice is in `decisions.md`
(referenced here as D-nn); the full numbers are in `evaluate/results/REPORT.md`; setup and the
headline results are in `README.md`.

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
  CUDA training (moved off by D-10), publishing to the Hugging Face Hub (prepared, not yet
  done — D-35).

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
| Adapter | LoRA r16, scale 2.0, dropout 0.05, all 32 blocks, qkv/o/gate_up/down | 25.17M trainable (0.66% of 3.82B) |
| Training | mlx-lm 0.32.0 driven from Python; batch 2 × 4; lr 2e-4 cosine, 3% warmup; 3 epochs; grad checkpointing; max sequence length 2048 (a cap never reached: the longest example is 1,178 tokens) | D-12…D-17 |
| Prompt | system prompt + instruction template with the schema JSON; read from `dataset_info.json` | identical at training and inference (D-19) |
| Inference | greedy; stop at end-of-turn, budget 48 + 80/field, or closing brace; batch fixed per run | D-20…D-23 |

## 3. Environment

- MacBook Pro, Apple **M5 Pro** (16-core GPU), **24 GB** unified memory, macOS 26.6.
- Python 3.12 venv at `.venv` (made with `uv`; system Python is 3.9). The install commands are in
  the README.
- `data/requirements.txt` (datasets) · `train/requirements.txt` (mlx-lm[train]==0.32.0, mlx==0.32.3).
- Memory: training peaks at 5.6 GB, batch-2 inference at 6–11 GB (the bf16 baseline is the
  highest). The first run downloads Phi-3.5-mini (about 7.6 GB) and keeps the 2.0 GB 4-bit copy.
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
- Invariants checked: completions parse and match targets in schema order (0 inconsistencies over
  2,080 examples), every prompt rebuilds byte for byte, no receipt in two splits by id or by text,
  0 duplicate ids, 0 empty receipt texts, Phase 5 names never trained on (D-09). One receipt
  corpus-wide is missing a field.
- **Source:** parquet mirrors of `darentang/sroie` on the Hugging Face Hub, tried in order. No
  remote code runs (`datasets>=3.0` refuses it anyway). `--source local` reads the official
  release from `data/raw/` instead.
- **Build steps:** BIO/BIOES token tags are decoded into one value per field; when a field has
  several spans (a company name repeated in the footer) the longest wins. OCR words are regrouped
  into printed lines by the vertical midpoints of their boxes. Receipts are split first and
  converted afterwards, so variants of one receipt can never straddle two splits.
- **Format:** `prompt` and `completion` are chat-message lists, not pre-rendered strings. mlx-lm
  applies the Phi-3.5 chat template and masks the prompt out of the loss itself, so hand-formatted
  special tokens would be doubled.
- **Schema augmentation:** train only, 63 distinct four-field schemas. Validation and test stay
  canonical so scores are comparable run to run.
- **Verbatim targets (D-03):** against the official task-3 gold on 100 receipts, verbatim spans
  agree exactly on date 97%, total 94%, company 83%, address 60%, and normalising lowered that.
- **Field pools (D-07):** never-present fields are `flight_number`, `patient_name`,
  `vehicle_plate_number`, `hotel_room_number`, `bank_account_number`, `student_id`; grounding
  fields are `time_of_purchase`, `store_phone`, `store_fax`, `store_email`.

### Phase 2 — Training · done
- **v1** (four-field data): 1,671 iterations (417 optimizer updates), 76 min, peak 5.6 GB. Train
  loss 0.570 → 0.005. Val loss 0.626 → 0.019, reached by iteration 700 and held (one blip to
  0.022 at iteration 1,100).
  - First attempt died at iteration ~90 (GPU out of memory, inference running alongside) and was
    restarted.
- **v2** (with absent-field data): 2,505 iterations (626 optimizer updates), 2 h 15 min, peak
  5.6 GB. Train loss 0.579 → under 0.001. Val loss 0.626 → 0.021 (best 0.019).
- **v2 adopted** on validation against a bar agreed in advance: four fields 234/248 (= v1), 0 false
  nulls, never-trained absent fields → `null` 62/62 (v1: 0/62) (D-18). The grounding rows of the
  selection probes in `REPORT.md` are labelled by the same patterns as training, so they show the
  patterns were learned, not that they are right.
- **Smoke run** (`--limit 20 --iters 5`): end to end in under a minute, about 0.4 iterations/s at
  batch 2, peak 4.7 GB.
- Both runs keep one checkpoint per epoch beside the final adapter and the full loss curve in
  `run_log.json`. On the final check's validation receipt, both reloaded adapters return parseable
  JSON with 3 of 4 fields exact.
- **Exit criteria, met by both runs:** training loss decreases, validation loss does not diverge,
  the smoke run completes end to end, the adapter saves and reloads, and one greedy generation on
  a validation receipt parses as JSON.

### Phase 3 — Inference · done
- `predict.py`, used by every evaluation. Each record notes which stop condition fired. The 62
  validation receipts take about a minute at batch 8, peaking at 9–12 GB.
- Validation spot check: v2 62/62 valid JSON, 234/248 fields exact; bare 4-bit base 54/62 (8
  unusable outputs) and 83/248.
- v1 and v2 score the same there but miss different fields. v2 fixes two totals and a truncated
  company name that v1 got wrong, and gets two receipts' store identity wrong instead (a
  promotional header, and a fuel brand in place of the operating company). About half of either
  adapter's 14 misses are noise in the gold spans: gold `Date:11/03/18` against a predicted
  `11/03/18`, or gold addresses that begin with a fragment of the line above.
- Found the 4-bit `2ized` year glitch, in 21 of 62 validation receipts (quantisation, not
  batching) → baseline 0b (D-24, D-25).

### Phase 4 — Evaluation · done
Test, 347 receipts, batch 2, normalised match (95% CI):

| System | Normalised | Exact | F1 | Question it answers |
|:--|--:|--:|--:|:--|
| **ParseTune v2** | **94.2 (92.9–95.5)** | 93.7 | 94.4 | What did fine-tuning buy? |
| zero-shot bf16 (baseline 0b) | 68.0 (66.0–70.2) | 55.0 | 68.0 | How much of the gap is quantisation damage rather than missing skill? |
| 3-shot 4-bit | 54.3 (51.5–56.8) | 41.4 | 55.2 | How much of the gain is just formatting demonstration? |
| zero-shot 4-bit | 44.6 (41.7–47.6) | 33.6 | 47.4 | What does the base model already do, on the weights the adapter runs on? |

- Fine-tuning gains +50 points on the same 4-bit weights, of which about 24 are quantisation
  repair. The fair headline is **+26 over the full-precision base**.
- Three worked examples only reach 54.3, so the gain is extraction and not formatting. The 4-bit
  base garbles years in 120 of 347 test outputs (26 of them wander until the token budget runs
  out); the bf16 base does it in none.
- ParseTune: 99.7% valid JSON as generated and the same schema adherence. Its 80 misses (of 1,388
  fields: address 37, company 28, total 12, date 3): 44 span boundaries (often gold noise, or
  ambiguous ones such as the brand against the legal entity), 32 different values, 4 from the one
  invalid output. Section 7 has the causes.
- ParseTune's exact and normalised scores differ by half a point. The baselines gain 8–13 points
  from normalisation, because they reformat amounts and dates.
- **Metrics (D-26):** normalised match drops `RM`, `MYR`, `$` and thousands separators and reads a
  decimal comma as a point (`RM20,80` = `20.80`); dates are read day-first (`21-05-17` =
  `21 May 2017`). Precision / recall / F1 are on non-null values, so inventing a value where the
  answer is `null` is penalised separately from missing one. Intervals use 1,000 bootstrap
  resamples over receipts.
- **Three-shot examples (D-27):** `sroie-train-00814-v0`, `sroie-train-00886-v0`,
  `sroie-train-00096-v0`.
- Every run writes a timestamped JSON with each prediction kept, and recomputing the metrics from
  the saved predictions reproduces every number.
- The batch-4 attempt ran out of memory on three-shot (D-28); the log is
  `evaluate/results/benchmark-batch4-oom.log`. Its completed runs show ParseTune is batch-stable
  (3/347 outputs changed) where the base is not (80/347). The benchmark also survived the Mac
  sleeping mid-run, because generation pauses rather than fails (D-29).

### Phase 5 — Generalisation · done
Held-out schema conditions on test (names never trained), four systems. The conditions are
specified in D-31 and their scoring in D-32.

| Condition | v2 | v1 | base bf16 | base 4-bit |
|:--|--:|--:|--:|--:|
| Renamed fields right | **96.1** | 96.1 | 77.8 | 53.3 |
| Subset / order right (schema kept) | **94.1 (100)** | 94.3 (100) | 65.9 (81.8) | 46.6 (71.2) |
| Absent field → `null` | **100** | 0 | 96.5 | 92.8 |
| `gst_id` precision / recall | 100 / 89.8 | 75.0 / 75.0 | 99.0 / 99.0 | 98.7 / 98.7 |
| `payment_method` precision / recall | 83.7 / 74.3 | 82.1 / 82.1 | 99.0 / 99.0 | 99.0 / 98.6 |

- **What holds:** the instruction generalises across *names and shapes* of the trained concepts.
  Given the same descriptions, the base model volunteers keys nobody asked for: asked only for
  `total`, it added `subTotal`, `tax`, `change`, `itemCount` and `qtyCount`.
- **What doesn't:** it does not generalise to *new concepts*. There, fine-tuning narrowed the
  model: the 4-bit base equals the bf16 base, so this is the fine-tune and not quantisation.
- **Absent-field data:** restored the `null` answers the four-field fine-tune had erased, at a
  small cost: 4 printed values (of 1,380) answered `null` when an absent field was in the request.
- **What the conditions measure:** training includes fields beyond SROIE's four, so the novel
  condition measures transfer to new fields rather than a first meeting with any, and the absent
  condition measures whether "absent" generalises to a concept (tracking and shipping) kept out of
  training on purpose.
- **Spot-check:** 50 receipts reviewed by the AI assistant, not a human. `gst_id` 42/50 right
  with 0 wrong values; `payment_method` 33/50; `line_item_count` 33/50.
- **Decisions:** v2 kept despite the `payment_method` regression (D-38); the 4-bit base was added
  to Phase 5 (D-39).

### Phase 6 — Packaging · done (Hub release not yet published)
- **Walkthrough notebook** (Apple Silicon, saved with its outputs): the data, both training runs'
  loss curves, one live extraction with renamed fields and an absent field, and the Phase 4 and 5
  results loaded from the saved runs. The heavy steps appear as commands rather than re-running.
- **Results report:** `python evaluate/report.py` rebuilds `REPORT.md` from the result files, so
  no number is copied by hand.
- Hub upload and PEFT conversion deliberately not done (D-35).
- **Docs:** a short README, this file, `decisions.md` and `REPORT.md`. The model card drafted for
  the Hub release was removed from the repository (D-42).

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
| `tests/test_pipeline.py` | 9 regression tests (`python tests/test_pipeline.py`) | yes |
| `evaluate/results/benchmark-batch4-oom.log` | the batch-4 out-of-memory run cited under Phase 4 (D-28) | yes |
| `evaluate/results/*.log`, `train/adapters/*/console*.log` | the other console logs | no |
| `~/.cache/huggingface/` | bf16 base weights (baseline 0b), SROIE download | no |

## 6. Reproduce

The full pipeline, in order:

```bash
source .venv/bin/activate
python data/prepare_data.py                        # Phase 1 (Hub → data/processed)
caffeinate -i python train/sft_train.py            # Phase 2 (~2 h 15 min; converts the base once)
python evaluate/schema_probe.py --adapter train/adapters/phi35-sroie-v2       # selection probes
caffeinate -i python -u evaluate/benchmark.py > evaluate/results/benchmark.log 2>&1   # Phase 4 (~36 min)
caffeinate -i python -u evaluate/schema_probe.py --conditions phase5 --name parsetune-v2 \
    --adapter train/adapters/phi35-sroie-v2                                     # Phase 5 (~24 min per system)
python evaluate/report.py                          # REPORT.md from the saved runs
```

Other invocations:

```bash
python data/prepare_data.py --limit 20             # smoke test
python data/prepare_data.py --schema-variants 1    # canonical schema only
python data/prepare_data.py --absent-variants 0    # without the absent-field examples
python data/prepare_data.py --source local         # offline, from data/raw/

python train/sft_train.py --limit 20 --iters 5     # smoke test, separate adapter directory
python train/sft_train.py --check-only             # convert + export + mask check, no training

python evaluate/predict.py --split validation --adapter train/adapters/phi35-sroie-v2
python evaluate/predict.py --split test --limit 20 --out base-test.json      # bare base model
python evaluate/predict.py --adapter train/adapters/phi35-sroie-v2 \
    --receipt receipt.txt --schema '{"merchant_name": "name of the shop"}'   # any schema

python evaluate/benchmark.py --compare-only        # rebuild the comparison table from saved runs
# follow a running benchmark from another terminal (Ctrl-C stops the watching, not the run)
tail -f evaluate/results/benchmark.log | grep --line-buffered -E "^===|receipts \(|done in|FAILED|all done"
```

Publishing the adapter to the Hugging Face Hub (not done, D-35):

```bash
huggingface-cli login
huggingface-cli upload <your-username>/parsetune-phi35-sroie-mlx \
    train/adapters/phi35-sroie-v2/adapters.safetensors adapters.safetensors
huggingface-cli upload <your-username>/parsetune-phi35-sroie-mlx \
    train/adapters/phi35-sroie-v2/adapter_config.json adapter_config.json
```

`mlx_lm.fuse --model models/phi-3.5-mini-instruct-4bit --adapter-path train/adapters/phi35-sroie-v2
--save-path parsetune-fused --dequantize` writes fused, de-quantised weights for use outside MLX
(not validated here).

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
- **Quantisation:** the adapter was trained on and is meant for the MLX 4-bit base. The bare 4-bit
  base garbles years and the adapter has learned around it; on a different quantisation its
  behaviour is unverified.
- **Not done on purpose:** Hub release (a manual step, D-35), PEFT conversion (needs validation).
  Until the adapter is published, its weights exist only on the training machine.

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
