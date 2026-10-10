# ParseTune — results report

_Generated 2026-10-08 22:09 UTC by `evaluate/report.py` from the files in `evaluate/results/`._

## Phase 4 — benchmark (test split)

347 receipts × 4 fields; batch size 2, seed 0 for every system. Scores in %. *Normalised* compares totals as amounts, dates as calendar dates, text case- and whitespace-insensitively; precision / recall / F1 are on non-null values with normalised matching. Intervals: 95% bootstrap over receipts.

| System | Exact (95% CI) | Normalised (95% CI) | Precision | Recall | F1 (95% CI) | Valid JSON (raw) | Schema adherence | Out-of-schema keys | Time | Peak memory |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **ParseTune (v2 adapter)** | 93.7 (92.2–95.1) | 94.2 (92.9–95.5) | 94.5 | 94.2 | 94.4 (93.1–95.6) | 99.7 | 99.7 | 0.0 | 8.0 min | 5.9 GB |
| Phi-3.5-mini zero-shot, bf16 | 55.0 (52.8–57.1) | 68.0 (66.0–70.2) | 68.0 | 68.0 | 68.0 (66.0–70.2) | 100.0 | 100.0 | 0.0 | 8.9 min | 10.8 GB |
| Phi-3.5-mini 3-shot, 4-bit | 41.4 (38.7–44.2) | 54.3 (51.5–56.8) | 56.3 | 54.3 | 55.2 (52.7–57.6) | 96.5 | 96.0 | 0.0 | 12.6 min | 10.0 GB |
| Phi-3.5-mini zero-shot, 4-bit | 33.6 (31.1–36.1) | 44.6 (41.7–47.6) | 50.6 | 44.6 | 47.4 (44.9–50.0) | 88.2 | 87.6 | 0.6 | 6.3 min | 5.9 GB |

### Per field (normalised precision / recall / F1, and exact accuracy)

| System | Field | Exact | Normalised | Precision | Recall | F1 |
|:--|:--|--:|--:|--:|--:|--:|
| **ParseTune (v2 adapter)** | company | 91.9 | 91.9 | 92.2 | 91.9 | 92.1 |
| **ParseTune (v2 adapter)** | date | 96.8 | 99.1 | 99.4 | 99.1 | 99.3 |
| **ParseTune (v2 adapter)** | address | 89.3 | 89.3 | 89.6 | 89.3 | 89.5 |
| **ParseTune (v2 adapter)** | total | 96.5 | 96.5 | 96.8 | 96.5 | 96.7 |
| Phi-3.5-mini zero-shot, bf16 | company | 48.1 | 58.2 | 58.2 | 58.2 | 58.2 |
| Phi-3.5-mini zero-shot, bf16 | date | 83.3 | 95.4 | 95.4 | 95.4 | 95.4 |
| Phi-3.5-mini zero-shot, bf16 | address | 15.0 | 35.7 | 35.7 | 35.7 | 35.7 |
| Phi-3.5-mini zero-shot, bf16 | total | 73.8 | 82.7 | 82.7 | 82.7 | 82.7 |
| Phi-3.5-mini 3-shot, 4-bit | company | 37.5 | 50.1 | 51.9 | 50.1 | 51.0 |
| Phi-3.5-mini 3-shot, 4-bit | date | 64.3 | 79.8 | 82.7 | 79.8 | 81.2 |
| Phi-3.5-mini 3-shot, 4-bit | address | 16.7 | 26.2 | 27.3 | 26.2 | 26.8 |
| Phi-3.5-mini 3-shot, 4-bit | total | 47.3 | 60.8 | 63.0 | 60.8 | 61.9 |
| Phi-3.5-mini zero-shot, 4-bit | company | 33.1 | 39.5 | 44.8 | 39.5 | 42.0 |
| Phi-3.5-mini zero-shot, 4-bit | date | 44.7 | 57.3 | 65.0 | 57.3 | 60.9 |
| Phi-3.5-mini zero-shot, 4-bit | address | 5.2 | 20.2 | 22.9 | 20.2 | 21.4 |
| Phi-3.5-mini zero-shot, 4-bit | total | 51.6 | 61.4 | 69.6 | 61.4 | 65.2 |

### Batch-size sensitivity (batch 4 vs batch 2, same test receipts)

| System | Outputs that changed | Normalised at batch 4 | at batch 2 |
|:--|--:|--:|--:|
| **ParseTune (v2 adapter)** | 3/347 | 94.4 | 94.2 |
| Phi-3.5-mini zero-shot, 4-bit | 80/347 | 44.1 | 44.6 |

The batch-4 attempt ran out of GPU memory during the three-shot run; its two completed systems are kept as evidence of batch sensitivity.

Files: `20261008T175318Z_test_parsetune.json`, `20261008T191006Z_test_zero-shot-bf16.json`, `20261008T190112Z_test_three-shot.json`, `20261008T175938Z_test_zero-shot.json`

## Phase 5 — schema-instructability (test split, held-out conditions)

None of these field names appeared in training (`prepare_data.py` refuses them). Printed fields are scored with normalised matching; a missing key or an unusable output counts as a null answer. Batch size 2, seed 0.

| Condition | System | Printed fields right | Precision | Recall | F1 | Printed fields answered null | Absent → null | Schema kept |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 1. Renamed fields (`merchant_name`, `purchase_date`, `amount_due`) | **ParseTune v2** (adopted) | 1000/1041 (96.1%) | 96.1 | 96.1 | 96.1 | 0 | – | 347/347 (100.0%) |
| 1. Renamed fields (`merchant_name`, `purchase_date`, `amount_due`) | ParseTune v1 (before absent-field data) | 1000/1041 (96.1%) | 96.1 | 96.1 | 96.1 | 0 | – | 347/347 (100.0%) |
| 1. Renamed fields (`merchant_name`, `purchase_date`, `amount_due`) | Phi-3.5-mini zero-shot, bf16 | 810/1041 (77.8%) | 77.8 | 77.8 | 77.8 | 0 | – | 347/347 (100.0%) |
| 1. Renamed fields (`merchant_name`, `purchase_date`, `amount_due`) | Phi-3.5-mini zero-shot, 4-bit | 555/1041 (53.3%) | 60.9 | 53.3 | 56.9 | 130 | – | 303/347 (87.3%) |
| 2. Random subset, random order of the four | **ParseTune v2** (adopted) | 809/860 (94.1%) | 94.1 | 94.1 | 94.1 | 0 | – | 347/347 (100.0%) |
| 2. Random subset, random order of the four | ParseTune v1 (before absent-field data) | 811/860 (94.3%) | 94.3 | 94.3 | 94.3 | 0 | – | 347/347 (100.0%) |
| 2. Random subset, random order of the four | Phi-3.5-mini zero-shot, bf16 | 567/860 (65.9%) | 66.5 | 65.9 | 66.2 | 7 | – | 284/347 (81.8%) |
| 2. Random subset, random order of the four | Phi-3.5-mini zero-shot, 4-bit | 401/860 (46.6%) | 52.6 | 46.6 | 49.4 | 98 | – | 247/347 (71.2%) |
| 4. Absent-field requests: the four fields | **ParseTune v2** (adopted) | 1302/1380 (94.3%) | 94.6 | 94.3 | 94.5 | 4 | – | 344/345 (99.7%) |
| 4. Absent-field requests: the four fields | ParseTune v1 (before absent-field data) | 1304/1380 (94.5%) | 94.5 | 94.5 | 94.5 | 0 | – | 345/345 (100.0%) |
| 4. Absent-field requests: the four fields | Phi-3.5-mini zero-shot, bf16 | 931/1380 (67.5%) | 67.5 | 67.5 | 67.5 | 0 | – | 345/345 (100.0%) |
| 4. Absent-field requests: the four fields | Phi-3.5-mini zero-shot, 4-bit | 627/1380 (45.4%) | 51.3 | 45.4 | 48.2 | 157 | – | 300/345 (87.0%) |
| 4. `shipping_tracking_number` (absent) → `null` | **ParseTune v2** (adopted) | – | – | – | – | 0 | 345/345 (100.0%) | 344/345 (99.7%) |
| 4. `shipping_tracking_number` (absent) → `null` | ParseTune v1 (before absent-field data) | – | 0.0 | – | – | 0 | 0/345 (0.0%) | 345/345 (100.0%) |
| 4. `shipping_tracking_number` (absent) → `null` | Phi-3.5-mini zero-shot, bf16 | – | 0.0 | – | – | 0 | 333/345 (96.5%) | 345/345 (100.0%) |
| 4. `shipping_tracking_number` (absent) → `null` | Phi-3.5-mini zero-shot, 4-bit | – | 0.0 | – | – | 0 | 320/345 (92.8%) | 300/345 (87.0%) |

### 3. Novel fields (`payment_method`, `gst_id`, `line_item_count`)

Never supervised. `gst_id` and `payment_method` are compared with strict-pattern silver labels where the receipt makes them unambiguous (a single 12-digit GST id after "GST": 304 receipts; a single payment keyword: 296). On those receipts each answer is *right*, a *wrong value*, or *null* — so precision (right / answered) and recall (right / labelled) can be told apart. `line_item_count` has no reliable pattern; it is judged by the manual spot-check below.

| System | Field | Answered (all receipts) | Right | Wrong value | Null | Precision | Recall | F1 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **ParseTune v2** (adopted) | `payment_method` | 291/347 (83.9%) | 220 | 43 | 33 | 83.7 | 74.3 | 78.7 |
| **ParseTune v2** (adopted) | `gst_id` | 282/347 (81.3%) | 273 | 0 | 31 | 100.0 | 89.8 | 94.6 |
| **ParseTune v2** (adopted) | `line_item_count` | 302/347 (87.0%) | manual check | | | | | |
| ParseTune v1 (before absent-field data) | `payment_method` | 347/347 (100.0%) | 243 | 53 | 0 | 82.1 | 82.1 | 82.1 |
| ParseTune v1 (before absent-field data) | `gst_id` | 347/347 (100.0%) | 228 | 76 | 0 | 75.0 | 75.0 | 75.0 |
| ParseTune v1 (before absent-field data) | `line_item_count` | 347/347 (100.0%) | manual check | | | | | |
| Phi-3.5-mini zero-shot, bf16 | `payment_method` | 336/347 (96.8%) | 293 | 3 | 0 | 99.0 | 99.0 | 99.0 |
| Phi-3.5-mini zero-shot, bf16 | `gst_id` | 333/347 (96.0%) | 301 | 3 | 0 | 99.0 | 99.0 | 99.0 |
| Phi-3.5-mini zero-shot, bf16 | `line_item_count` | 328/347 (94.5%) | manual check | | | | | |
| Phi-3.5-mini zero-shot, 4-bit | `payment_method` | 344/347 (99.1%) | 292 | 3 | 1 | 99.0 | 98.6 | 98.8 |
| Phi-3.5-mini zero-shot, 4-bit | `gst_id` | 334/347 (96.3%) | 300 | 4 | 0 | 98.7 | 98.7 | 98.7 |
| Phi-3.5-mini zero-shot, 4-bit | `line_item_count` | 340/347 (98.0%) | manual check | | | | | |

`payment_method` by silver label (cash dominates — a model that always answered "cash" would agree on most receipts, so per-label counts matter):

| System | cash | credit card | debit | mastercard | visa |
|:--|--:|--:|--:|--:|--:|
| **ParseTune v2** (adopted) | 215/282 | 0/2 | 0/1 | 1/2 | 4/9 |
| ParseTune v1 (before absent-field data) | 232/282 | 1/2 | 1/1 | 0/2 | 9/9 |
| Phi-3.5-mini zero-shot, bf16 | 282/282 | 2/2 | 0/1 | 2/2 | 7/9 |
| Phi-3.5-mini zero-shot, 4-bit | 282/282 | 2/2 | 0/1 | 2/2 | 6/9 |

### Manual spot-check of novel fields — parsetune-v2, 50 receipts

A fixed sample of 50 test receipts (random.Random(2026).sample over the 347) was reviewed by the AI assistant, reading each receipt's full text beside the adapter's answers. Verdicts: *correct*; *defensible* (the right value with stray characters, or a count matching the receipt's own printed item total when that total counts units); *wrong*; *null, but printed* (the field is on the receipt); *null, correctly* (it is not). `line_item_count` counts distinct item lines. Every verdict and note is in `evaluate/results/phase5_spotcheck.json`; the review was by an AI, not a human.

| Field | correct | defensible | wrong | null, but printed | null, correctly |
|:--|--:|--:|--:|--:|--:|
| `payment_method` | 32 | 1 | 10 | 7 | 0 |
| `gst_id` | 39 | 1 | 0 | 8 | 2 |
| `line_item_count` | 29 | 4 | 12 | 5 | 0 |

- payment_method: all 10 wrong answers copy a nearby label containing 'cash' - 'Cashier' (7), a cashier or counter id ('cash1', 'CASH1'), the 'Cash Bill' header - including on receipts paid by VISA or credit card. It is the verbatim-copy habit from training applied to a field it never learned.
- gst_id: never a wrong value. Every miss is a format unlike the usual 'GST ID:' line - 'GST REG #...' printed at the bottom (6 receipts, one chain) or an OCR-garbled label ('GBT', 'GETID').
- line_item_count: right on single-item receipts; on multi-item receipts it often misses by one or two rather than reading the printed 'Item(s)' total, and on supplier delivery notes it answers null.

Files: `20261008T195321Z_phase5_parsetune-v2.json`, `20261008T201902Z_phase5_parsetune-v1.json`, `20261008T205143Z_phase5_zero-shot-bf16.json`, `20261008T211031Z_phase5_zero-shot-4bit.json`

## Model selection (validation split, before test was touched)

The adoption bar was agreed before retraining: four fields ≥ 231/248 exact, no printed field answered null, ≥ 80% null on never-trained absent fields.

| Probe | v1 | v2 (adopted) |
|:--|--:|--:|
| Four SROIE fields, exact | 234/248 (94.4%) | 234/248 (94.4%) |
| Renamed fields (`shop_name`, …), exact | 177/186 (95.2%) | 176/186 (94.6%) |
| Never-trained absent field → null | 0/62 (0.0%) | 62/62 (100.0%) |
| Grounding fields: printed value right | 19/82 (23.2%) | 82/82 (100.0%) |
| Grounding fields: absent → null | 0/131 (0.0%) | 131/131 (100.0%) |

