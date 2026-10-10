---
base_model: microsoft/Phi-3.5-mini-instruct
library_name: mlx
tags:
  - mlx
  - lora
  - qlora
  - information-extraction
  - receipts
  - structured-output
  - json
datasets:
  - SROIE (ICDAR 2019)
language:
  - en
---

# ParseTune — Phi-3.5-mini LoRA adapter for schema-instructable receipt extraction (MLX)

A LoRA adapter for `microsoft/Phi-3.5-mini-instruct` that reads receipt text and returns **exactly
the JSON schema it is handed** — the requested keys, in the requested order, values copied verbatim
from the receipt, and `null` for anything the receipt does not contain. The schema travels in the
prompt as `{field: description}`, so it also answers field names it never trained on.

- **Adapter:** LoRA rank 16 (scale 2.0 ≈ alpha 32), dropout 0.05, on `qkv_proj`, `o_proj`,
  `gate_up_proj`, `down_proj` in all 32 blocks — 25.17M parameters, 101 MB (`adapters.safetensors`).
- **Base it was trained on:** Phi-3.5-mini-instruct converted to MLX 4-bit (affine, group size 64)
  — QLoRA. Load it on the same 4-bit conversion for the results below.
- **Framework:** `mlx-lm` 0.32.0 on Apple Silicon (trained on an M5 Pro, 24 GB, 2 h 15 min).
- **Source:** the ParseTune repository — data build, training, evaluation and every result file.

## Use

```python
import json
from mlx_lm import convert, load, generate

receipt_text = open("receipt.txt").read()          # OCR text, one printed line per line

# once: the 4-bit base the adapter was trained on
convert("microsoft/Phi-3.5-mini-instruct", mlx_path="phi-3.5-mini-instruct-4bit",
        quantize=True, q_bits=4, q_group_size=64)

model, tokenizer = load("phi-3.5-mini-instruct-4bit", adapter_path="parsetune-adapter")
schema = {"merchant_name": "name of the merchant or business issuing the receipt",
          "amount_due": "final total amount paid",
          "loyalty_card_number": "membership or loyalty card number of the customer"}
messages = [
    {"role": "system", "content": "You are a document extraction engine. You read a document and "
     "return only a single JSON object matching the schema you are given. No explanations."},
    {"role": "user", "content": "Extract the following fields from this receipt: " + json.dumps(schema)
     + "\n\nRespond with a single JSON object containing exactly these keys, in this order. "
     "If a field does not appear in the receipt, use null.\n\nReceipt text:\n" + receipt_text},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=48 + 80 * len(schema)))
```

The prompt must match the training template exactly (above; also recorded in
`data/processed/dataset_info.json`). The repository's `evaluate/predict.py` does this for you, with
batching, a stop at the closing brace and a flagged JSON-repair pass.

## Training data

SROIE (ICDAR 2019 key-information extraction): Malaysian retail receipts, English, OCR text with
line layout rebuilt from word boxes. 557 training receipts → 1,671 examples:

- each receipt with SROIE's four fields (company, date, address, total), plus one random subset in
  random order;
- one example per receipt mixing 1–3 real fields with fields it does not all carry: *grounding*
  fields (time, phone, fax, email) labelled by strict patterns only where unambiguous, and
  *never-present* fields (flight number, patient name, …) whose absence is checked. 495 real `null`
  answers; **no generated text anywhere**.

Answers are verbatim OCR spans. The 7 receipts SROIE repeats across its official train and test
splits were removed from training.

## Training procedure

QLoRA with `mlx-lm`: 3 epochs, 2,505 iterations at batch 2 × 4 gradient accumulation, AdamW, peak
learning rate 2e-4 with 3% warmup and cosine decay, gradient checkpointing, loss on the answer only
(prompt masked — verified on every example). Chosen over a v1 adapter (four-field data only) on the
validation split, against a bar fixed before retraining.

## Evaluation — SROIE official test split (347 receipts, never used for any choice)

Normalised match compares totals as amounts, dates as calendar dates, text case-insensitively;
F1 on non-null values; 95% bootstrap intervals over receipts. Batch size 2.

| System | Normalised (95% CI) | Exact | F1 | Valid JSON (raw) |
|:--|--:|--:|--:|--:|
| **This adapter** | **94.2 (92.9–95.5)** | **93.7** | **94.4** | **99.7** |
| Phi-3.5-mini zero-shot, bf16 | 68.0 (66.0–70.2) | 55.0 | 68.0 | 100.0 |
| Phi-3.5-mini 3-shot, 4-bit | 54.3 (51.5–56.8) | 41.4 | 55.2 | 96.5 |
| Phi-3.5-mini zero-shot, 4-bit | 44.6 (41.7–47.6) | 33.6 | 47.4 | 88.2 |

Per field (normalised): company 91.9 · date 99.1 · address 89.3 · total 96.5.

**Schema-following on held-out names (Phase 5, test, none seen in training):**

| Condition | This adapter | Base model, zero-shot (bf16) |
|:--|--:|--:|
| Renamed fields (`merchant_name`, `purchase_date`, `amount_due`) right | **96.1%** | 77.8% |
| Random subset in random order right / requested keys kept | **94.1% / 100%** | 65.9% / 81.8% |
| Absent field (`shipping_tracking_number`) answered `null` | **345/345** | 333/345 |
| Never-taught `gst_id`: precision / recall vs strict-pattern labels | 100% / 89.8% | **99.0% / 99.0%** |
| Never-taught `payment_method`: precision / recall | 83.7% / 74.3% | **99.0% / 99.0%** |

The adapter follows schemas over the concepts it was trained on — under new names, orders and
subsets — far better than the base model, and answers `null` for absent fields. On genuinely new
concepts it is *worse* than the base model it started from (the 4-bit base scores the same ~99%,
so this is the fine-tune, not quantisation): it tends to copy nearby text such as "Cashier" where
the base answers "Cash". For a field unlike receipt company / date / address / total, prefer the
base model or verify the answer.

## Limitations

- **Domain:** Malaysian retail receipts from 2017–2018 in English. Other receipt styles, languages
  and document types are untested.
- **Input is OCR text,** not images; quality depends on the OCR and on the line order rebuilt from
  it (some address errors are fragments in OCR order).
- **Boundaries:** company and address spans are where most errors sit; part of that is noise in
  SROIE's gold labels.
- **Quantisation:** trained on and meant for the MLX 4-bit base. The bare 4-bit base garbles years;
  the adapter has learned around it, but on a different quantisation its behaviour is unverified.
- **Absent fields:** trained to answer `null`, and tested on held-out names, but no model can be
  shown never to invent a value; check critical fields against the receipt.
- **New concepts:** fields unlike the four it was trained on are answered less reliably than by
  the base model (see Phase 5 above); a missing value is also likelier than with the base model.
- **JSON:** a value copied from text containing a `"` can break the JSON (1 of 347 test receipts).

## Licence and data terms

The base model is MIT-licensed. SROIE was released for the ICDAR 2019 Robust Reading Competition;
check its terms before any use beyond research. Choose the adapter's licence accordingly when
publishing.

## Publishing (not done — left to the repository owner)

```bash
huggingface-cli login
huggingface-cli upload <your-username>/parsetune-phi35-sroie-mlx \
    train/adapters/phi35-sroie-v2/adapters.safetensors adapters.safetensors
huggingface-cli upload <your-username>/parsetune-phi35-sroie-mlx \
    train/adapters/phi35-sroie-v2/adapter_config.json adapter_config.json
huggingface-cli upload <your-username>/parsetune-phi35-sroie-mlx MODEL_CARD.md README.md
```

`mlx_lm.fuse --model models/phi-3.5-mini-instruct-4bit --adapter-path train/adapters/phi35-sroie-v2
--save-path parsetune-fused --dequantize` writes fused, de-quantised weights for use outside
MLX (not validated here). A PEFT-format adapter is not provided: it would need validating against
the MLX adapter first.
