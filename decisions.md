# ParseTune — decision log

Every decision that shaped the project, with what was considered and why the choice
fell where it did. Entries are grouped by phase; the D-numbers give the order the
decisions were made in. Where a decision was **changed**, the entry records the
original choice, what changed it, and the replacement — the original reasoning is
kept, not rewritten.

Dates are when the decision was made. Phase 1 was built before 2026-09-02 (the
initial commit); everything from the status check onward happened on 2026-10-08/09,
and the publication entries (D-36, D-40, D-41) on 2026-10-10.

**Status key:** ✅ kept · 🔄 changed (see the entry) · ⏸ deferred

---

## 0. Scope and framing — the original plan (before 2026-09-02)

### D-01 ✅ Learn the *instruction*, not a fixed output shape
- **Decision:** the schema travels in the prompt as `{field: description}`, and the model
  returns exactly those keys, in that order, and nothing else.
- **Why:** most extraction fine-tunes learn one fixed shape. If the schema is part of the input,
  a model trained on receipts can answer field names and orderings it never saw — that claim is
  the whole project, and Phase 5 exists to test it.
- **Consequence:** descriptions, not just names, go in the prompt; they are the bridge to unseen
  field names (D-30).

### D-02 ✅ Phi-3.5-mini-instruct (3.8B) as the base model
- **Why:** small enough to fine-tune on one consumer machine, instruction-tuned already, MIT
  licensed, long context.

### D-03 ✅ SROIE, with verbatim OCR spans as targets
- **Decision:** SROIE (ICDAR 2019, four fields: company, date, address, total) from a
  parquet mirror on the Hugging Face Hub; targets are the exact OCR spans, not normalised values.
- **Why verbatim:** checked against the official gold on 100 receipts, verbatim spans agree
  better than normalised ones (the gold itself is inconsistent), and verbatim spans are the only
  targets a text-only model can copy from its input. Cosmetic differences (`RM 11.40` vs `11.40`)
  are a *scoring* concern, handled in Phase 4's normalised match (D-24).
- **Related:** receipts are rebuilt into printed lines from OCR boxes (layout is signal); schema
  augmentation (random field subsets and orders) on train only; validation carved from train at
  10%; the official test split untouched so results stay comparable.

### D-04 ✅ Six phases
Data → Training → Inference → Evaluation → Generalisation → Packaging, each with a written
deliverable and exit criteria in the README. Kept throughout; the README's status table is the
source of truth for progress.

---

## 1. Data (Phase 1, revisited 2026-10-08)

### D-05 ✅ Drop the 7 receipts that appear in both train and test
- **Found:** during the status check. Seven test receipts have byte-identical OCR text to train
  receipts. The original "0 receipt_id overlap" check could not see them: ids carry their split as
  a prefix. The duplication comes from SROIE's official split itself.
- **Options:** drop from test (breaks comparability with published numbers) · drop from train ·
  ignore (≈2% of test silently seen in training).
- **Decision:** drop the training copies; never touch test. Validation is cleaned against test,
  train against both. Train: 564 → 557 receipts.
- **Verified:** before the change, regenerating with the old code reproduced the committed data
  byte for byte, so the only difference afterwards is the dedup.

### D-06 🔄 Null answers in training: first deferred, then added
- **Original (2026-10-08, status check):** I flagged that the training data held 1 `null` answer
  in 1,128 examples, so Phase 5's absent-field test was set up to fail. The user chose to fix only
  the leakage and the README for that round — reasonable, since the gap was a prediction, not yet
  an observed failure.
- **What changed it:** Phase 3 produced the evidence. Asked for `shipping_tracking_number`, the
  v1 adapter returned a document number off the receipt instead of `null`.
- **Replacement:** add absent-field examples and retrain, decided before touching test (D-07,
  D-18). The user ruled out synthetic data explicitly, because generated text costs real-world
  accuracy — so the recipe had to use real receipts only.

### D-07 ✅ The absent-field recipe: real receipts, verified answers, no generated text
- **Options weighed with the user:** (a) fields that never occur on these receipts, always
  `null` · (b) (a) plus "grounding" fields that are printed on some receipts and not others ·
  (c) (a) plus renamed SROIE fields with gold values · (d) a second human-annotated dataset
  (e.g. WildReceipt).
- **Decision: (b).** Each training receipt gets one extra example mixing 1–3 of its SROIE fields
  with a grounding field and, in a third of cases, a never-present field.
  - *Never-present* (`flight_number`, `patient_name`, …): a keyword scan of all 966 receipts
    found 0–5 hits; any receipt with a hit is skipped for that field.
  - *Grounding* (`time_of_purchase`, `store_phone`, `store_fax`, `store_email`): a strict pattern
    labels the field only when it finds exactly one value (taken verbatim) or proves absence;
    anything in between is skipped, never guessed.
- **Why not (a) alone:** if a field name is always `null` in training, the model learns "unfamiliar
  name → null" and would start nulling fields that *are* printed. Grounding fields carry the same
  name with values on some receipts and nulls on others, so the only way to answer is to read the
  receipt.
- **Why not (c):** it would teach renaming directly and spoil Phase 5's renamed-field test.
- **Why not (d):** the most "real" option, but a new loader, a licence check and a different
  receipt distribution — the biggest scope; kept as future work.
- **Result:** 557 examples, 495 `null` answers (9% of training slots), 251 real values.
  `--absent-variants 0` reproduces the four-field data byte for byte, and the original 1,114
  examples are unchanged (the new pass uses its own RNG and appends).

### D-08 🔄 Three corrections to the grounding labels, found before training
1. **Newline run-on (changed before first use):** phone and fax patterns first used `\s`, which
   matches line breaks, so a number could absorb digits from the next printed line. Patterns now
   match spaces and tabs only.
2. **Email precision (changed twice):** the first scan showed `SR@6.00` (tax notation) matching a
   loose email pattern → the top-level domain must be alphabetic. The hand audit then found
   `IVAN KONG12@HOTMAIL.COM`, an address split by OCR, labelled as the fragment
   `KONG12@HOTMAIL.COM` → an address must now start its line or follow a label's colon; anything
   else is skipped.
3. **Balance (changed after the first generation):** the first build asked for `store_email` with
   a value 3 times and with `null` 148 times — nearly "the name predicts the answer", the shortcut
   D-07 exists to avoid. The pick now prefers the (field, has-value) pairing used least so far:
   email went to 17 values / 90 nulls; every field appears both ways.
- **Audit result (final):** 60 sampled values and 40 sampled nulls checked against their receipt
  lines — one systematic flaw (item 2), fixed; 0 wrong nulls.

### D-09 ✅ Field-pool hygiene, enforced in code
Phase 5's held-out names *and concepts* (tracking/shipping/courier, payment, GST, line items)
are refused by an assertion in `prepare_data.py`; a third pool of absent fields
(`passport_number`, …) is reserved for choosing between adapters on validation and never trained
on; all pools must be disjoint or the data build stops.

---

## 2. Training (Phase 2)

### D-10 🔄 Training stack: CUDA QLoRA + PEFT + TRL → MLX on the MacBook
- **Original plan (README):** QLoRA with bitsandbytes NF4, PEFT and TRL's `SFTTrainer`, on a
  16 GB CUDA GPU (Kaggle T4/P100), batch 1 × 8, `paged_adamw_8bit`, bf16 compute.
- **First amendment (status check):** T4 and P100 have no bf16, so the plan needed an fp16
  fallback.
- **What changed it:** the user decided to train locally on a MacBook Pro M5 Pro with 24 GB of
  unified memory. bitsandbytes' 4-bit path, its 8-bit/paged optimizers and bf16-on-CUDA do not
  carry over.
- **Options:** PyTorch on the Apple GPU (MPS) + PEFT/TRL, 16-bit LoRA — *my recommendation*,
  because it kept the README's stack and a standard PEFT adapter · MLX (`mlx-lm`) — Apple's own
  framework: fastest on M5, true 4-bit QLoRA, prompt masking built in, but an MLX-format adapter.
- **Decision (user's): MLX.** Accepted trade-offs: the README's stack was rewritten, and a
  PEFT/Kaggle release needs a conversion step (deferred, D-35).
- **Outcome:** 76 min for v1, 2 h 15 min for v2, peak 5.6 GB — the choice was sound.

### D-11 ✅ Convert the base model ourselves
Microsoft's checkpoint, converted once to MLX 4-bit (affine, group size 64) rather than pulling a
community upload: pins provenance and the quantisation settings to `train/config.yaml`.

### D-12 ✅ LoRA configuration
Rank 16; `scale` 2.0 (mlx-lm multiplies the LoRA output directly, so 2.0 = the README's alpha 32 /
rank 16); dropout 0.05; all 32 blocks (`num_layers: -1` — mlx-lm's default adapts only the last
16); targets `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` (Phi-3's fused projections — the
usual `q_proj`/`k_proj` names silently match nothing). Verified: exactly 25.17M trainable
parameters, the expected count.

### D-13 🔄 Batch size: 1 × 8 → 8 → 2 × 4
- **Original:** 1 × 8 (CUDA memory); my first MLX draft used batch 8 (unified memory allows it).
- **What changed it:** reading mlx-lm's `iterate_batches` — it sorts by length and silently skips
  the tail that cannot fill a batch, always the *longest* examples. Batch 8 would never train on 2
  examples and never score 6 of the 62 validation examples.
- **Decision:** 2 × 4 accumulation (effective batch 8). 1,114 and 62 are both even → nothing
  skipped. With v2's 1,671 examples (odd), one example — the longest — sits out each epoch; its
  receipt still trains through its other variants. Documented rather than distorted.

### D-14 ✅ Learning-rate schedule sized in optimizer updates
mlx-lm counts `iters` in micro-batches but steps the schedule once per optimizer update. Sizing
warmup and cosine decay in iterations would have ended the decay a quarter of the way in. Peak
2e-4, 3% warmup, cosine to 0, 3 epochs, AdamW without weight decay.

### D-15 ✅ Gradient checkpointing stays on — measured, not assumed
With 4.7 GB peak there seemed to be room to turn it off. Measured: off is ~30% faster but peaks at
16.9 GB on 780-token batches; the longest examples are 1,178 tokens, which would approach the
24 GB ceiling. Kept on.

### D-16 ✅ Prove the loss mask on every example
- Export writes only `{"messages": …}`: mlx-lm guesses the format from the keys, and records still
  carrying `prompt`/`completion` lists would be read as the plain-string format.
- Before training, every example goes through mlx-lm's own dataset code and the span past the mask
  offset must decode to exactly the JSON answer (+ end-of-turn markers). 1,733 / 1,733 pass.
- mlx-lm 0.32's loss also counts one padding target after `<|endoftext|>` (an off-by-one in its
  mask). Documented, not patched: it sits past the `<|end|>` where generation stops, so it never
  reaches the output, and patching meant re-implementing mlx-lm's trainer.

### D-17 ✅ Drive mlx-lm from Python, pin it exactly
`mlx_lm.lora`'s entry point replaces any training callback, and the loss curve is what the exit
criteria are judged on — so `sft_train.py` calls `train_model` directly with a logger. That uses
mlx-lm internals, hence the exact pin (`mlx-lm[train]==0.32.0`, `mlx==0.32.3`). Python 3.12 via
`uv`: mlx-lm needs ≥ 3.11 and macOS ships 3.9.

### D-18 ✅ Choose between adapters on validation, against a bar agreed in advance
- **Bar (set with the user before retraining):** the four fields stay within noise (≥ 231/248
  exact vs v1's 234), no printed field answered `null`, ≥ 80% `null` on never-trained absent fields.
- **Result:** v2 234/248, 0 false nulls, 62/62 nulls (v1: 0/62) → **v2 adopted**. v1 kept at
  `train/adapters/phi35-sroie` for comparison and rollback.
- **Why this way:** the test set must not be used to pick a model; a bar fixed before seeing v2's
  numbers cannot be bent to fit them.

---

## 3. Inference (Phase 3)

### D-19 ✅ One prompt contract, proven
`predict.py` reads the template from `dataset_info.json` instead of restating it, and before
predicting on a split rebuilds every stored prompt from its parts — all 2,080 match byte for byte.
Inference cannot drift from training.

### D-20 ✅ Greedy decoding with three stop conditions, on `BatchGenerator`
End-of-turn token, a per-schema token budget, or the first JSON object closing — so a chatty
model cannot spend its budget on prose after the answer. mlx-lm's `batch_generate` hard-codes its
stop tokens, so the brace stop needed the lower-level `BatchGenerator` (continuous batching).

### D-21 🔄 Token budget: 32 + 64 per field → 48 + 80 per field
- **Original:** 32 + 64 × fields, assumed generous.
- **What changed it:** measuring the corpus — a lone long address needs 78 tokens, against a
  1-field budget of 96. Too tight to call generous.
- **Replacement:** 48 + 80 × fields; the longest gold answer of any schema size now uses at most
  ~60% of its budget.

### D-22 ✅ JSON repair is flagged, never hidden
Take the first top-level `{…}` (drops fences and prose), close an object the budget cut off — and
record `repaired` beside the raw text, so raw validity is reportable. In practice ParseTune needed
no repair; its single invalid output (an unescaped quote copied from OCR) was beyond repair.

### D-23 ✅ Fix the batch size for evaluation, and record it
Batched and single-sequence decoding are not bit-identical; near-tied greedy choices flip. On
validation the bare base model changed 5 of 16 outputs between batch 8 and 1, the adapter 0 of 16.
So every evaluation run records its batch size and all systems in a comparison share one.

### D-24 ✅ The `2ized` glitch is quantisation, not batching
The 4-bit base writes years as `2ized` (`15-06-2ized`). Ruled out batching (single-sequence
generation reproduces it) and confirmed with the bf16 base, which copies the same dates exactly.
This finding produced baseline 0b (D-25).

---

## 4. Evaluation (Phase 4)

### D-25 🔄 Baselines: three rows → four
- **Original:** zero-shot (4-bit), 3-shot (4-bit), ParseTune.
- **What changed it:** D-24. Comparing ParseTune only with the 4-bit base would credit
  fine-tuning for undoing quantisation damage.
- **Added:** 0b, zero-shot on the full-precision (bf16) base. On test it showed the honest
  headline: ParseTune 94.2% vs bf16 base 68.0% (+26 points), not just +50 over the damaged 4-bit base.

### D-26 ✅ Metrics, defined before the test run
Exact and normalised match (totals as amounts to the cent incl. decimal commas; dates as calendar
dates read day-first; text case- and whitespace-insensitive), precision / recall / F1 on non-null
values, raw JSON validity, schema adherence (exact keys, exact order), out-of-schema keys, and 95%
bootstrap intervals over receipts. Normalisers were unit-tested on 19 real formats first.

### D-27 ✅ Three-shot examples
Three canonical training receipts no longer than the median length, seed 0, ids recorded — keeps
prompts bounded and the baseline reproducible.

### D-28 🔄 Benchmark batch size: 4 → 2, and failure isolation
- **Original:** batch 4 for every system.
- **What changed it:** the three-shot run hit a Metal out-of-memory error at 30%. Its prompts are
  ~3,000 tokens and continuous batching holds up to twice the batch in memory while the next
  prompts load; the 8-receipt smoke test had no long enough receipts to show it.
- **Replacement:** batch 2 for *all* systems (keeping them comparable, D-23) and a full re-run. A
  failing system is now logged and skipped so the others finish, and the comparison table only
  mixes runs of one batch size.
- **Kept as evidence:** the two batch-4 runs that completed. They measured batch sensitivity on
  the full test set: ParseTune changed 3 of 347 outputs (94.4 → 94.2), the bare base 80 of 347.

### D-29 🔄 Long runs go under `caffeinate`
- **Original:** I *suggested* `caffeinate` to the user and left it there.
- **What changed it:** the Mac idled into sleep mid-benchmark; the three-shot run crawled forward
  only during brief maintenance wakes. (Pausing does not change results — MLX resumes — but the run
  stalled.)
- **Replacement:** I attached `caffeinate` to the running benchmark, and every long command since
  (and in the README) starts under it.

### D-30 ✅ Test touched once per system in the final configuration
Model selection used validation only (D-18). Each system ran on test once at batch 2; the
superseded batch-4 runs are kept and disclosed, not hidden.

---

## 5. Generalisation (Phase 5)

### D-31 ✅ The four held-out conditions, as the README specified them
1. **Renamed:** `merchant_name`, `purchase_date`, `amount_due`, each with its SROIE field's own
   description — new keys, same semantics; the description is the only bridge (D-01). (Validation
   had already probed renamed keys with *paraphrased* descriptions, 176/186.)
2. **Subset and order:** a random subset of the four in a random (never canonical) order — the
   sanity control.
3. **Novel:** `payment_method`, `gst_id`, `line_item_count` — printed, never supervised.
4. **Absent:** the four + `shipping_tracking_number`; the 2 test receipts that mention tracking or
   couriers are excluded, so `null` is verifiably right for the other 345.

### D-32 ✅ How the novel fields are scored
- `gst_id` and `payment_method`: compared with strict-pattern **silver** labels where the receipt
  makes them unambiguous (exactly one 12-digit id after "GST": 304 receipts; exactly one payment
  keyword: 295 receipts). Silver labels approximate truth; disagreements were sampled by hand.
- `payment_method` is reported per label: 281 of the 295 are cash, so a model that always said
  "cash" would look excellent overall.
- `line_item_count` has no reliable pattern, so it is judged by the README's manual spot-check
  on a 50-receipt sample. **The spot-check was done by the AI assistant (Claude) reading each
  receipt, not by a human** — the protocol and every verdict are saved in
  `evaluate/results/phase5_spotcheck.json` so it can be audited or redone.

### D-33 🔄 Systems in Phase 5: v2, v1 and the bf16 base
v2 is the system under test; v1 shows on test what the absent-field data changed; the bf16 base
shows how much schema-following the base model already had. *Changed by D-39: the 4-bit base was
added once the bf16 base beat both adapters on the novel fields.*

### D-38 ✅ Keep v2 despite its `payment_method` regression on test
- **Found in Phase 5 (test):** the absent-field data made v2 more precise and less eager on
  fields it was never taught. `gst_id`: v1 returned 76 wrong ids, v2 none (precision 75% → 100%,
  recall 75% → 90%). But `payment_method`: recall fell from 82% to 74% (v2 answers `null` on 33
  receipts that print a method) and it copies the word "Cashier" more often (55 answers vs 39);
  on the 9 VISA receipts v1 was right on all 9, v2 on 4.
- **Options:** switch back to v1 · keep v2 and report the regression · retrain to fix it.
- **Decision:** keep v2, report the regression plainly. The adapter was chosen on validation
  against a bar fixed before retraining (D-18); changing the choice because of a *test* result
  would select on the test set and void the honest headline. The four fields, renamed fields and
  absent fields — the project's claims — are equal or better under v2.
- **Next, if pursued:** treat `payment_method` like a grounding field (labelled where a payment
  line is unambiguous) in a v3, chosen again on validation only — and swap it out of Phase 5's
  novel set for another never-trained field, or the novel-field test stops being one.

---

### D-39 🔄 Phase 5 systems: three → four (the 4-bit base added)
- **Original (D-33):** v2, v1 and the bf16 base.
- **What changed it:** on the novel fields the bf16 base *beat both adapters* (payment method
  99% vs 74%, GST id 99% vs 90% agreement). Two explanations compete: fine-tuning narrowed the
  model (it copies text — `Cashier` — where the base answers the concept — `Cash`), or the
  adapters are handicapped by running on 4-bit weights. The same logic that added baseline 0b in
  Phase 4 (D-25) applies in reverse.
- **Added:** the 4-bit base, zero-shot — the same weights the adapters run on — so the
  novel-field gap splits into a quantisation part and a fine-tuning part.
- **Outcome:** the 4-bit base scores as high as the bf16 one on the novel fields (GST id 98.7%,
  payment method 98.6% recall), so the gap is the fine-tune, not quantisation. The project's claim
  is narrowed accordingly in the README: the instruction generalises across names and shapes of
  the trained concepts, not to new concepts.

## 6. Packaging (Phase 6)

### D-34 🔄 The end-to-end notebook targets Apple Silicon, not Kaggle
- **Original:** a Kaggle T4/P100 notebook.
- **Changed by D-10:** MLX training does not run on Kaggle's CUDA GPUs. The notebook walks the
  pipeline on the Mac and reads the saved results.

### D-35 ⏸ Hub release prepared, not published; PEFT conversion deferred
- **Hub:** the model card and the upload command are ready (`MODEL_CARD.md`). Publishing is a
  public action on the user's own Hugging Face account, so it is left to the user.
- **PEFT conversion:** deferred. It would need PyTorch, PEFT and a validation run to prove the
  converted adapter behaves like the MLX one; an unvalidated conversion could silently degrade.
  Apple Silicon users load the MLX adapter directly; `mlx_lm.fuse` can export fused weights.

### D-36 🔄 Nothing committed or pushed → prepared for one commit, pushed by the user
- **Original:** the user never asked for a commit. All work sits in the working tree for review.
- **What changed it (2026-10-10):** the user asked to push the results, and chose to run the git
  commands themselves.
- **Replacement:** the working tree is prepared for a single commit covering Phases 2–6 (D-40,
  D-41); the user stages, commits and pushes it.

### D-37 ✅ Turn the one-off checks into regression tests
Every check run by hand along the way — the 19 normaliser cases, the JSON-repair cases, the
token budget against every gold answer, prompts rebuilding byte for byte, no receipt in two
splits, absent-field answers coming only from real text, Phase 5 names never trained — is now in
`tests/test_pipeline.py` (plain asserts, CPU only, runs without pytest). The numbers rest on these
pieces; a change that breaks one should fail loudly, not shift a score silently.

### D-40 🔄 What goes in git, and what stays on the Mac (2026-10-10)
- **In git:** all code and tests; `data/processed/`; every results JSON, `REPORT.md` and the
  comparison tables; the notebook with its outputs; the docs. About 12 MB, largest file 4.4 MB.
- **Local only:** the converted base model (2.0 GB, rebuilt by `sft_train.py`); the adapter
  weights and checkpoints (ten files of 96 MB — each just under GitHub's 100 MB file limit, and
  history would carry them for good); `train/data/` (rebuilt every run); the smoke-run adapter;
  the virtual environment.
- **Original:** everything under `train/adapters/` and every console log was ignored.
- **What changed it:** deciding what to publish showed two gaps. The loss curves the notebook
  plots and the full training settings lived only beside the weights, reproducible otherwise by
  retraining for ~2 h; and the README cites `benchmark-batch4-oom.log` as kept beside the batch-4
  runs (D-28) while the ignore rules dropped it.
- **Replacement:** `run_log.json` and `adapter_config.json` for v1 and v2 (~70 KB) and that one
  log are tracked. The other console logs stay local — the results JSON already hold what they
  show.
- **Consequence:** until the Hub release (D-35) the adopted adapter's weights exist only on this
  machine. A clone can read every result but must retrain to run the model.

### D-41 ✅ Recorded paths are relative to the repository (2026-10-10)
- **Found before the first push:** the absolute project path — and with it the macOS user name —
  was recorded in eight results files (`config.model`, `config.adapter`), `dataset_info.json`
  (`raw_dir`, `out_dir`), both adapter configs, the batch-4 log and one notebook output. None of it
  was in pushed history yet.
- **Decision:** strip the project prefix, so the files read `models/phi-3.5-mini-instruct-4bit`
  and `train/adapters/phi35-sroie-v2`. Done as a byte replacement of that one prefix rather than
  a JSON rewrite, so every prediction, metric and timing is exactly as the runs wrote it. The
  notebook's first cell now prints the project folder's name instead of its path; its saved output
  line was edited to match, and the notebook was not re-executed.
- **Checked:** 33 lines changed across 13 files — 32 recorded paths and that one print statement;
  the nine regression tests pass; `report.py` regenerates `REPORT.md` identically apart from its
  timestamp; the v2 adapter loads with its edited config (mlx-lm reads only the LoRA settings from
  it) and extracts a validation receipt correctly.
- **Going forward:** `benchmark.py`, `schema_probe.py`, `prepare_data.py` and `sft_train.py` record
  paths relative to the repository. A Hub id, or a path outside the repository, is recorded as
  given.
