# ParseTune decision log

Every decision that shaped the project, in the order it was made: what was chosen, what else was
considered, and why. A decision that was later changed keeps its original choice and reasoning,
followed by what changed it and what replaced it. Entries not marked *changed* or *deferred* still
stand.

Phase 1 and the original plan (D-01 to D-04) predate the initial commit of 2026-09-02. D-05 to
D-39 were made on 2026-10-08 and 2026-10-09. D-40 to D-42 were made on 2026-10-10, when D-36 was
also revised.

## Index

| ID | Decision | Status |
|:--|:--|:--|
| D-01 | Learn the instruction, not a fixed output shape | Kept |
| D-02 | Phi-3.5-mini-instruct (3.8B) as the base model | Kept |
| D-03 | SROIE, with verbatim OCR spans as targets | Kept |
| D-04 | Six phases | Kept |
| D-05 | Drop the 7 receipts that appear in both train and test | Kept |
| D-06 | Null answers in training: deferred, then added | Changed |
| D-07 | Absent-field recipe: real receipts, verified answers, no generated text | Kept |
| D-08 | Three corrections to the grounding labels, found before training | Changed |
| D-09 | Field-pool hygiene, enforced in code | Kept |
| D-10 | Training stack: CUDA QLoRA with PEFT and TRL, replaced by MLX on a MacBook | Changed |
| D-11 | Convert the base model locally | Kept |
| D-12 | LoRA configuration | Kept |
| D-13 | Batch size: 1 × 8, then 8, then 2 × 4 | Changed |
| D-14 | Learning-rate schedule sized in optimizer updates | Kept |
| D-15 | Gradient checkpointing stays on | Kept |
| D-16 | Prove the loss mask on every example | Kept |
| D-17 | Drive mlx-lm from Python and pin it exactly | Kept |
| D-18 | Choose between adapters on validation, against a bar fixed in advance | Kept |
| D-19 | One prompt contract, proven | Kept |
| D-20 | Greedy decoding with three stop conditions | Kept |
| D-21 | Token budget: 32 + 64 per field, raised to 48 + 80 per field | Changed |
| D-22 | JSON repair is flagged, never hidden | Kept |
| D-23 | Fix the batch size for evaluation and record it | Kept |
| D-24 | The `2ized` glitch is quantisation, not batching | Kept |
| D-25 | Baselines: three rows, then four | Changed |
| D-26 | Metrics defined before the test run | Kept |
| D-27 | Three-shot examples | Kept |
| D-28 | Benchmark batch size: 4, reduced to 2, with failure isolation | Changed |
| D-29 | Long runs go under `caffeinate` | Changed |
| D-30 | Test touched once per system in the final configuration | Kept |
| D-31 | The four held-out conditions | Kept |
| D-32 | How the novel fields are scored | Kept |
| D-33 | Phase 5 systems: v2, v1 and the bf16 base | Changed |
| D-34 | The walkthrough notebook targets Apple Silicon, not Kaggle | Changed |
| D-35 | Hub release prepared but not published; PEFT conversion deferred | Deferred |
| D-36 | Nothing committed during the work; published as one commit | Changed |
| D-37 | Turn the one-off checks into regression tests | Kept |
| D-38 | Keep v2 despite its `payment_method` regression on test | Kept |
| D-39 | Phase 5 systems: three, then four, with the 4-bit base added | Changed |
| D-40 | What goes in git and what stays local | Changed |
| D-41 | Recorded paths are relative to the repository | Kept |
| D-42 | One place per fact: model card removed, README shortened | Kept |

## Scope and framing (D-01 to D-04)

### D-01 Learn the instruction, not a fixed output shape

- **Decision:** the schema travels in the prompt as `{field: description}`, and the model returns
  exactly those keys, in that order, and nothing else.
- **Why:** most extraction fine-tunes learn one fixed shape. If the schema is part of the input,
  a model trained on receipts can answer field names and orderings it never saw. That claim is
  the whole project, and Phase 5 exists to test it.
- **Note:** descriptions, not just names, go in the prompt. They are the bridge to unseen field
  names (D-31).

### D-02 Phi-3.5-mini-instruct (3.8B) as the base model

- **Decision:** `microsoft/Phi-3.5-mini-instruct` is the base model.
- **Why:** small enough to fine-tune on one consumer machine, already instruction-tuned, MIT
  licensed, long context.

### D-03 SROIE, with verbatim OCR spans as targets

- **Decision:** SROIE (ICDAR 2019; four fields: company, date, address, total) from a parquet
  mirror on the Hugging Face Hub. Targets are the exact OCR spans, not normalised values.
- **Why:** checked against the official gold on 100 receipts, verbatim spans agree better than
  normalised ones (the gold itself is inconsistent), and verbatim spans are the only targets a
  text-only model can copy from its input. Cosmetic differences (`RM 11.40` vs `11.40`) are a
  scoring concern, handled by Phase 4's normalised match (D-26).
- **Note:** receipts are rebuilt into printed lines from OCR boxes, because layout is signal.
  Schema augmentation (random field subsets and orders) is applied to train only. Validation is
  carved from train at 10%, and the official test split is untouched so results stay comparable.

### D-04 Six phases

- **Decision:** Data, Training, Inference, Evaluation, Generalisation, Packaging, each with a
  written deliverable and exit criteria in the README.
- **Note:** kept throughout. Progress was tracked in the README's status table; since D-42 the
  per-phase write-ups live in `memory.md`.

## Data (D-05 to D-09)

### D-05 Drop the 7 receipts that appear in both train and test

- **Context:** found during the status check. Seven test receipts have byte-identical OCR text to
  train receipts. The original "0 receipt_id overlap" check could not see them, because ids carry
  their split as a prefix. The duplication comes from SROIE's official split itself.
- **Options:** drop them from test (breaks comparability with published numbers); drop them from
  train; ignore them (about 2% of test silently seen in training).
- **Decision:** drop the training copies and never touch test. Validation is cleaned against
  test, train against both. Train went from 564 to 557 receipts.
- **Check:** before the change, regenerating with the old code reproduced the committed data byte
  for byte, so the only difference afterwards is the dedup.

### D-06 Null answers in training: deferred, then added (changed)

- **Before:** the status check on 2026-10-08 flagged that the training data held 1 `null` answer
  in 1,128 examples, so Phase 5's absent-field test was set up to fail. Only the leakage and the
  README were fixed in that round, since the gap was a prediction and not yet an observed failure.
- **What changed:** Phase 3 produced the evidence. Asked for `shipping_tracking_number`, the v1
  adapter returned a document number off the receipt instead of `null`.
- **After:** absent-field examples were added and the model retrained, decided before touching
  test (D-07, D-18). Synthetic data was ruled out because generated text costs real-world
  accuracy, so the recipe had to use real receipts only.

### D-07 Absent-field recipe: real receipts, verified answers, no generated text

- **Options:** (a) fields that never occur on these receipts, always `null`; (b) option (a) plus
  "grounding" fields that are printed on some receipts and not others; (c) option (a) plus renamed
  SROIE fields with gold values; (d) a second human-annotated dataset such as WildReceipt.
- **Decision:** option (b). Each training receipt gets one extra example mixing 1 to 3 of its
  SROIE fields with a grounding field and, in a third of cases, a never-present field.
  - Never-present fields (`flight_number`, `patient_name`, ...): a keyword scan of all 966
    receipts found 0 to 5 hits. Any receipt with a hit is skipped for that field.
  - Grounding fields (`time_of_purchase`, `store_phone`, `store_fax`, `store_email`): a strict
    pattern labels the field only when it finds exactly one value (taken verbatim) or proves
    absence. Anything in between is skipped, never guessed.
- **Why:**
  - Not (a) alone: if a field name is always `null` in training, the model learns that an
    unfamiliar name means `null` and would start nulling fields that are printed. Grounding fields
    carry the same name with values on some receipts and nulls on others, so the only way to
    answer is to read the receipt.
  - Not (c): it would teach renaming directly and spoil Phase 5's renamed-field test.
  - Not (d): the most realistic option, but it needs a new loader, a licence check and a different
    receipt distribution. It had the biggest scope and is kept as future work.
- **Result:** 557 examples, 495 `null` answers (9% of training slots), 251 real values.
  `--absent-variants 0` reproduces the four-field data byte for byte, and the original 1,114
  examples are unchanged (the new pass uses its own RNG and appends).

### D-08 Three corrections to the grounding labels, found before training (changed)

- **What changed:** three flaws in the first grounding patterns, each fixed before training.
  1. Newline run-on (fixed before first use). Phone and fax patterns first used `\s`, which
     matches line breaks, so a number could absorb digits from the next printed line. Patterns
     now match spaces and tabs only.
  2. Email precision (changed twice). The first scan showed `SR@6.00` (tax notation) matching a
     loose email pattern, so the top-level domain must now be alphabetic. The hand audit then
     found `IVAN KONG12@HOTMAIL.COM`, an address split by OCR, labelled as the fragment
     `KONG12@HOTMAIL.COM`. An address must now start its line or follow a label's colon. Anything
     else is skipped.
  3. Balance (changed after the first generation). The first build asked for `store_email` with a
     value 3 times and with `null` 148 times, which is close to the name predicting the answer,
     the shortcut D-07 exists to avoid. The pick now prefers the (field, has-value) pairing used
     least so far. Email went to 17 values and 90 nulls, and every field appears both ways.
- **Check:** in the final audit, 60 sampled values and 40 sampled nulls were checked against their
  receipt lines. It found one systematic flaw (item 2), since fixed, and 0 wrong nulls.

### D-09 Field-pool hygiene, enforced in code

- **Decision:** Phase 5's held-out names and concepts (tracking, shipping, courier, payment, GST,
  line items) are refused by an assertion in `prepare_data.py`. A third pool of absent fields
  (`passport_number`, ...) is reserved for choosing between adapters on validation and is never
  trained on. All pools must be disjoint or the data build stops.

## Training (D-10 to D-18)

### D-10 Training stack: CUDA QLoRA with PEFT and TRL, replaced by MLX on a MacBook (changed)

- **Before:** the README planned QLoRA with bitsandbytes NF4, PEFT and TRL's `SFTTrainer` on a
  16 GB CUDA GPU (Kaggle T4/P100), batch 1 × 8, `paged_adamw_8bit`, bf16 compute. The status check
  added a first amendment: T4 and P100 have no bf16, so the plan needed an fp16 fallback.
- **What changed:** training moved to a local MacBook Pro M5 Pro with 24 GB of unified memory.
  bitsandbytes' 4-bit path, its 8-bit and paged optimizers, and bf16-on-CUDA do not carry over.
- **Options:** PyTorch on the Apple GPU (MPS) with PEFT/TRL and 16-bit LoRA, which was the first
  recommendation because it kept the README's stack and a standard PEFT adapter; or MLX
  (`mlx-lm`), Apple's own framework: fastest on M5, true 4-bit QLoRA, prompt masking built in, but
  an MLX-format adapter.
- **After:** MLX. The accepted trade-offs are that the README's stack was rewritten and that a
  PEFT/Kaggle release needs a conversion step (deferred, D-35).
- **Result:** 76 min for v1, 2 h 15 min for v2, peak 5.6 GB. The choice was sound.

### D-11 Convert the base model locally

- **Decision:** Microsoft's checkpoint is converted once to MLX 4-bit (affine, group size 64)
  rather than pulling a community upload.
- **Why:** it pins provenance and the quantisation settings to `train/config.yaml`.

### D-12 LoRA configuration

- **Decision:** rank 16, `scale` 2.0, dropout 0.05, all 32 blocks (`num_layers: -1`), targets
  `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`.
- **Why:** mlx-lm multiplies the LoRA output by `scale` directly, so 2.0 equals the README's
  alpha 32 / rank 16. mlx-lm's default adapts only the last 16 blocks. The targets are Phi-3's
  fused projections; the usual `q_proj`/`k_proj` names silently match nothing.
- **Check:** exactly 25.17M trainable parameters, the expected count.

### D-13 Batch size: 1 × 8, then 8, then 2 × 4 (changed)

- **Before:** 1 × 8, sized for CUDA memory. The first MLX draft used batch 8, which unified memory
  allows.
- **What changed:** reading mlx-lm's `iterate_batches`. It sorts by length and silently skips the
  tail that cannot fill a batch, which is always the longest examples. Batch 8 would never train
  on 2 examples and never score 6 of the 62 validation examples.
- **After:** 2 × 4 accumulation (effective batch 8). 1,114 and 62 are both even, so nothing is
  skipped. With v2's 1,671 examples (odd), one example, the longest, sits out each epoch; its
  receipt still trains through its other variants. Documented rather than distorted.

### D-14 Learning-rate schedule sized in optimizer updates

- **Decision:** peak 2e-4, 3% warmup, cosine to 0, 3 epochs, AdamW without weight decay, with
  warmup and decay sized in optimizer updates.
- **Why:** mlx-lm counts `iters` in micro-batches but steps the schedule once per optimizer
  update. Sizing warmup and cosine decay in iterations would have ended the decay a quarter of the
  way in.

### D-15 Gradient checkpointing stays on

- **Decision:** keep gradient checkpointing on. This was measured, not assumed.
- **Why:** with a 4.7 GB peak there seemed to be room to turn it off. Measured, off is about 30%
  faster but peaks at 16.9 GB on 780-token batches. The longest examples are 1,178 tokens, which
  would approach the 24 GB ceiling.

### D-16 Prove the loss mask on every example

- **Decision:**
  - The export writes only `{"messages": ...}`. mlx-lm guesses the format from the keys, and
    records still carrying `prompt`/`completion` lists would be read as the plain-string format.
  - Before training, every example goes through mlx-lm's own dataset code, and the span past the
    mask offset must decode to exactly the JSON answer plus end-of-turn markers.
- **Check:** 1,733 of 1,733 examples pass.
- **Note:** mlx-lm 0.32's loss also counts one padding target after `<|endoftext|>` (an off-by-one
  in its mask). This is documented, not patched. It sits past the `<|end|>` where generation
  stops, so it never reaches the output, and patching meant re-implementing mlx-lm's trainer.

### D-17 Drive mlx-lm from Python and pin it exactly

- **Decision:** `sft_train.py` calls mlx-lm's `train_model` directly with a logger, with exact
  pins (`mlx-lm[train]==0.32.0`, `mlx==0.32.3`), on Python 3.12 installed with `uv`.
- **Why:** the `mlx_lm.lora` entry point replaces any training callback, and the loss curve is
  what the exit criteria are judged on. Calling `train_model` uses mlx-lm internals, hence the
  exact pin. mlx-lm needs Python 3.11 or later and macOS ships 3.9.

### D-18 Choose between adapters on validation, against a bar fixed in advance

- **Decision:** the bar, set before retraining: the four fields stay within noise (at least
  231/248 exact, against v1's 234), no printed field is answered `null`, and at least 80% `null`
  on never-trained absent fields.
- **Why:** the test set must not be used to pick a model, and a bar fixed before seeing v2's
  numbers cannot be bent to fit them.
- **Result:** v2 scored 234/248, 0 false nulls and 62/62 nulls (v1: 0/62), so v2 was adopted. v1
  is kept at `train/adapters/phi35-sroie` for comparison and rollback.

## Inference (D-19 to D-24)

### D-19 One prompt contract, proven

- **Decision:** `predict.py` reads the template from `dataset_info.json` instead of restating it,
  and before predicting on a split it rebuilds every stored prompt from its parts.
- **Why:** inference cannot drift from training.
- **Check:** all 2,080 prompts match byte for byte.

### D-20 Greedy decoding with three stop conditions

- **Decision:** greedy decoding that stops at the end-of-turn token, a per-schema token budget, or
  the first JSON object closing, built on mlx-lm's `BatchGenerator` (continuous batching).
- **Why:** a chatty model cannot spend its budget on prose after the answer. mlx-lm's
  `batch_generate` hard-codes its stop tokens, so the brace stop needed the lower-level
  `BatchGenerator`.

### D-21 Token budget: 32 + 64 per field, raised to 48 + 80 per field (changed)

- **Before:** 32 + 64 × fields, assumed generous.
- **What changed:** measuring the corpus. A lone long address needs 78 tokens, against a 1-field
  budget of 96. Too tight to call generous.
- **After:** 48 + 80 × fields. The longest gold answer of any schema size now uses at most about
  60% of its budget.

### D-22 JSON repair is flagged, never hidden

- **Decision:** take the first top-level `{...}` (which drops fences and prose), close an object
  the budget cut off, and record `repaired` beside the raw text.
- **Why:** raw validity stays reportable.
- **Result:** in practice ParseTune needed no repair. Its single invalid output (an unescaped
  quote copied from OCR) was beyond repair.

### D-23 Fix the batch size for evaluation and record it

- **Decision:** every evaluation run records its batch size, and all systems in a comparison share
  one.
- **Why:** batched and single-sequence decoding are not bit-identical, and near-tied greedy
  choices flip. On validation the bare base model changed 5 of 16 outputs between batch 8 and 1,
  the adapter 0 of 16.

### D-24 The `2ized` glitch is quantisation, not batching

- **Context:** the 4-bit base writes years as `2ized` (`15-06-2ized`).
- **Check:** batching was ruled out, because single-sequence generation reproduces it. The bf16
  base confirmed the cause: it copies the same dates exactly.
- **Result:** this finding produced baseline 0b (D-25).

## Evaluation (D-25 to D-30)

### D-25 Baselines: three rows, then four (changed)

- **Before:** zero-shot (4-bit), 3-shot (4-bit), ParseTune.
- **What changed:** D-24. Comparing ParseTune only with the 4-bit base would credit fine-tuning
  for undoing quantisation damage.
- **After:** baseline 0b was added: zero-shot on the full-precision (bf16) base.
- **Result:** on test it showed the honest headline: ParseTune 94.2% vs the bf16 base's 68.0%
  (+26 points), not just +50 over the damaged 4-bit base.

### D-26 Metrics defined before the test run

- **Decision:** exact and normalised match (totals as amounts to the cent, including decimal
  commas; dates as calendar dates read day-first; text case- and whitespace-insensitive),
  precision / recall / F1 on non-null values, raw JSON validity, schema adherence (exact keys,
  exact order), out-of-schema keys, and 95% bootstrap intervals over receipts.
- **Check:** the normalisers were unit-tested on 19 real formats first.

### D-27 Three-shot examples

- **Decision:** three canonical training receipts no longer than the median length, seed 0, ids
  recorded.
- **Why:** it keeps prompts bounded and the baseline reproducible.

### D-28 Benchmark batch size: 4, reduced to 2, with failure isolation (changed)

- **Before:** batch 4 for every system.
- **What changed:** the three-shot run hit a Metal out-of-memory error at 30%. Its prompts are
  about 3,000 tokens, and continuous batching holds up to twice the batch in memory while the next
  prompts load. The 8-receipt smoke test had no receipts long enough to show it.
- **After:** batch 2 for all systems (keeping them comparable, D-23) and a full re-run. A failing
  system is now logged and skipped so the others finish, and the comparison table only mixes runs
  of one batch size.
- **Note:** the two batch-4 runs that completed are kept as evidence. They measured batch
  sensitivity on the full test set: ParseTune changed 3 of 347 outputs (94.4 to 94.2), the bare
  base 80 of 347.

### D-29 Long runs go under `caffeinate` (changed)

- **Before:** `caffeinate` was suggested for long runs but not enforced.
- **What changed:** the Mac idled into sleep mid-benchmark, and the three-shot run crawled forward
  only during brief maintenance wakes. Pausing does not change results, because MLX resumes, but
  the run stalled.
- **After:** `caffeinate` was attached to the running benchmark, and every long command since,
  including those in the README, starts under it.

### D-30 Test touched once per system in the final configuration

- **Decision:** model selection used validation only (D-18). Each system ran on test once at
  batch 2. The superseded batch-4 runs are kept and disclosed, not hidden.

## Generalisation (D-31 to D-33)

### D-31 The four held-out conditions

- **Decision:** the four conditions the README specified.
  1. Renamed: `merchant_name`, `purchase_date`, `amount_due`, each with its SROIE field's own
     description. New keys, same semantics; the description is the only bridge (D-01). Validation
     had already probed renamed keys with paraphrased descriptions (176/186).
  2. Subset and order: a random subset of the four in a random, never canonical, order. This is
     the sanity control.
  3. Novel: `payment_method`, `gst_id`, `line_item_count`. Printed, never supervised.
  4. Absent: the four plus `shipping_tracking_number`. The 2 test receipts that mention tracking
     or couriers are excluded, so `null` is verifiably right for the other 345.

### D-32 How the novel fields are scored

- **Decision:**
  - `gst_id` and `payment_method` are compared with strict-pattern silver labels where the receipt
    makes them unambiguous (exactly one 12-digit id after "GST": 304 receipts; exactly one payment
    keyword: 296 receipts). Silver labels approximate truth; disagreements were sampled by hand.
  - `payment_method` is reported per label. 282 of the 296 are cash, so a model that always said
    "cash" would look excellent overall.
  - `line_item_count` has no reliable pattern, so it is judged by the README's manual spot-check
    on a 50-receipt sample.
- **Note:** the spot-check was done by an AI assistant (Claude) reading each receipt, not by a
  human. The protocol and every verdict are saved in `evaluate/results/phase5_spotcheck.json` so
  it can be audited or redone.

### D-33 Phase 5 systems: v2, v1 and the bf16 base (changed)

- **Before:** three systems. v2 is the system under test, v1 shows on test what the absent-field
  data changed, and the bf16 base shows how much schema-following the base model already had.
- **What changed:** the bf16 base beat both adapters on the novel fields.
- **After:** the 4-bit base was added (D-39).

## Packaging (D-34 to D-37)

### D-34 The walkthrough notebook targets Apple Silicon, not Kaggle (changed)

- **Before:** a Kaggle T4/P100 notebook.
- **What changed:** D-10. MLX training does not run on Kaggle's CUDA GPUs.
- **After:** the notebook walks the pipeline on the Mac and reads the saved results.

### D-35 Hub release prepared but not published; PEFT conversion deferred (deferred)

- **Decision:** the Hub release is prepared but not published, and the PEFT conversion is
  deferred.
- **Why:** a model card and the upload commands were written (`MODEL_CARD.md`), but publishing is
  a public action on a personal Hugging Face account, so it is left as a manual step. A PEFT
  conversion would need PyTorch, PEFT and a validation run to prove the converted adapter behaves
  like the MLX one; an unvalidated conversion could silently degrade.
- **Note:** Apple Silicon users load the MLX adapter directly, and `mlx_lm.fuse` can export fused
  weights. The model card was later removed from the repository (D-42); the upload commands are
  kept in `memory.md`, section 6.

### D-36 Nothing committed during the work; published as one commit (changed)

- **Before:** nothing was committed or pushed while Phases 2 to 6 were in progress. All work sat
  in the working tree for review.
- **What changed:** on 2026-10-10 the results were ready to publish.
- **After:** Phases 2 to 6 were pushed as a single commit, after settling what belongs in git
  (D-40) and making the recorded paths relative (D-41).

### D-37 Turn the one-off checks into regression tests

- **Decision:** every check run by hand along the way is now in `tests/test_pipeline.py` (plain
  asserts, CPU only, runs without pytest): the 19 normaliser cases, the JSON-repair cases, the
  token budget against every gold answer, prompts rebuilding byte for byte, no receipt in two
  splits, absent-field answers coming only from real text, and Phase 5 names never trained.
- **Why:** the numbers rest on these pieces. A change that breaks one should fail loudly, not
  shift a score silently.

## Generalisation, after the test results (D-38 to D-39)

### D-38 Keep v2 despite its `payment_method` regression on test

- **Context:** Phase 5 on test showed that the absent-field data made v2 more precise and less
  eager on fields it was never taught. On `gst_id`, v1 returned 76 wrong ids and v2 none (precision
  75% to 100%, recall 75% to 90%). But `payment_method` recall fell from 82% to 74% (v2 answers
  `null` on 33 receipts that print a method), and v2 copies the word "Cashier" more often (55
  answers vs 39). On the 9 VISA receipts v1 was right on all 9, v2 on 4.
- **Options:** switch back to v1; keep v2 and report the regression; retrain to fix it.
- **Decision:** keep v2 and report the regression plainly.
- **Why:** the adapter was chosen on validation against a bar fixed before retraining (D-18).
  Changing the choice because of a test result would select on the test set and void the honest
  headline. The four fields, renamed fields and absent fields, which are the project's claims, are
  equal or better under v2.
- **Note:** if pursued, the next step is to treat `payment_method` like a grounding field
  (labelled where a payment line is unambiguous) in a v3, chosen again on validation only, and to
  swap it out of Phase 5's novel set for another never-trained field. Otherwise the novel-field
  test stops being one.

### D-39 Phase 5 systems: three, then four, with the 4-bit base added (changed)

- **Before:** v2, v1 and the bf16 base (D-33).
- **What changed:** on the novel fields the bf16 base beat both adapters (payment method 99% vs
  74%, GST id 99% vs 90% agreement). Two explanations compete: fine-tuning narrowed the model (it
  copies text, `Cashier`, where the base answers the concept, `Cash`), or the adapters are
  handicapped by running on 4-bit weights. The same logic that added baseline 0b in Phase 4 (D-25)
  applies in reverse.
- **After:** the 4-bit base, zero-shot, was added. It uses the same weights the adapters run on,
  so the novel-field gap splits into a quantisation part and a fine-tuning part.
- **Result:** the 4-bit base scores as high as the bf16 one on the novel fields (GST id 98.7%,
  payment method 98.6% recall), so the gap is the fine-tune, not quantisation. The project's claim
  is narrowed accordingly in the README: the instruction generalises across names and shapes of
  the trained concepts, not to new concepts.

## Publication (D-40 to D-42)

### D-40 What goes in git and what stays local (changed)

- **Before:** everything under `train/adapters/` and every console log was ignored.
- **What changed:** deciding what to publish showed two gaps. The loss curves the notebook plots
  and the full training settings lived only beside the weights, reproducible otherwise by
  retraining for about 2 h. And the README cited `benchmark-batch4-oom.log` as kept beside the
  batch-4 runs (D-28) while the ignore rules dropped it.
- **After:**
  - In git: all code and tests; `data/processed/`; every results JSON, `REPORT.md` and the
    comparison tables; the notebook with its outputs; the docs. About 12 MB, largest file 4.4 MB.
    `run_log.json` and `adapter_config.json` for v1 and v2 (about 70 KB) and that one log are now
    tracked.
  - Local only: the converted base model (2.0 GB, rebuilt by `sft_train.py`); the adapter weights
    and checkpoints (ten files of 96 MB, each just under GitHub's 100 MB file limit, and history
    would carry them for good); `train/data/` (rebuilt every run); the smoke-run adapter; the
    virtual environment; the other console logs, because the results JSON already hold what they
    show.
- **Note:** until the Hub release (D-35) the adopted adapter's weights exist only on the training
  machine. A clone can read every result but must retrain to run the model.

### D-41 Recorded paths are relative to the repository

- **Context:** before the first push, the absolute project path, including the local user name,
  was found recorded in eight results files (`config.model`, `config.adapter`),
  `dataset_info.json` (`raw_dir`, `out_dir`), both adapter configs, the batch-4 log and one
  notebook output. None of it was in pushed history yet.
- **Decision:** strip the project prefix, so the files read `models/phi-3.5-mini-instruct-4bit`
  and `train/adapters/phi35-sroie-v2`. This was a byte replacement of that one prefix rather than
  a JSON rewrite, so every prediction, metric and timing is exactly as the runs wrote it. The
  notebook's first cell now prints the project folder's name instead of its path. Its saved output
  line was edited to match, and the notebook was not re-executed.
- **Check:** 33 lines changed across 13 files (32 recorded paths and that one print statement).
  The nine regression tests pass, `report.py` regenerates `REPORT.md` identically apart from its
  timestamp, and the v2 adapter loads with its edited config (mlx-lm reads only the LoRA settings
  from it) and extracts a validation receipt correctly.
- **Note:** from here on `benchmark.py`, `schema_probe.py`, `prepare_data.py` and `sft_train.py`
  record paths relative to the repository. A Hub id, or a path outside the repository, is recorded
  as given.

### D-42 One place per fact: model card removed, README shortened

- **Context:** after the first push the same material sat in three files: the README (a long
  write-up of every phase), `MODEL_CARD.md` and `memory.md`.
- **Decision:** remove `MODEL_CARD.md` and cut the README down to an entry point: what the project
  is, the headline results, setup, usage, limitations, and links to everything else. `memory.md`
  holds the project detail, this file the reasons, and `evaluate/results/REPORT.md` the full
  numbers.
- **Why:** a fact kept in one place cannot drift out of step with its copies. The README may
  repeat the headline because it is the first file a visitor opens, but it stays short.
- **Note:** the per-phase detail that existed only in the README moved into `memory.md`. With the
  model card gone, a card for the Hub page has to be written at release time (D-35).
