# Component 2 — training + re-extraction + C1 train/test split (handoff, 25 Nov 2026)

Send this folder to your friend. Unzip it and **keep the folder structure exactly as it is** — `common_io.py` and `build_cutoff_views.py` compute paths relative to this root.

Root after unzipping = the folder containing `component2_entity_relation_extraction/` and `data_collection/`.

## What he is doing (and what he is NOT doing)

He will:
1. fine-tune SciBERT on SciERC (training),
2. run the trained model over the enlarged corpus (re-extraction),
3. canonicalize entities and build the cutoff-local **train/test** views (the C1 leak fix).

He will **not** touch Components 3–6, the dashboard, or the paper.

"Re-extraction" means running inference with the trained model over all 2,800 NLP + 1,200 COVID papers to produce new `NLP_entities.json`, `NLP_relations.json`, `NLP_extracted.json` (and COVID equivalents). Training alone only produces weights; re-extraction produces the data the rest of the pipeline uses. The submitted results used the old 1,400/840 corpus.

## 1. Environment (Python 3.12)

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`requirements.txt` pins the versions that are known to work here: `transformers==4.55.2`, `datasets==2.19.2`, `huggingface-hub==0.36.0`, `pyarrow==17.0.0`, `torch>=2.0.0`, `rapidfuzz`, `scikit-learn`, `seqeval`, `spacy`. Use a CUDA build of torch if he has an NVIDIA GPU. First run downloads base SciBERT (~440 MB) from Hugging Face; the SciERC train/dev/test JSONs are already included, so nothing else needs downloading.

## 2. One edit for a GPU machine

Open `component2_entity_relation_extraction/scripts/extract_entities_relations_scibert.py`, find `TrainingArguments(...)` around line 233 and **remove** this line (it was added for the Apple M1 in the original project):

```python
use_cpu=True,      # MPS runs out of memory on this machine (Apple M1)
```

Keep the other fixes already in the file:
- `os.environ.setdefault("USE_TF", "0")` and `report_to="none"` — avoid TensorFlow/TensorBoard hangs;
- `use_safetensors=False` on the base model load — the SciBERT repo currently has only `pytorch_model.bin`.

`extract_relations_scibert.py` has the same safetensors fix and needs no edit.

## 3. Commands (all from the unzipped root)

```bash
cd component2_entity_relation_extraction/scripts

# 1) Fine-tune SciBERT NER on SciERC (weights -> checkpoints/scibert_scierc/ner/)
python extract_entities_relations_scibert.py --train

# 2) Sanity check on a small slice before the full run
python extract_entities_relations_scibert.py --run --domain NLP --year 2018

# 3) FULL RE-EXTRACTION on the enlarged corpus (2,800 NLP + 1,200 COVID)
python extract_entities_relations_scibert.py --run --domain both

# 4) Canonicalize entities over the new full corpus (in place)
python normalize_entities.py

# 5) C1 LEAK FIX — cutoff-local train/test views (no future data in train)
python build_cutoff_views.py --force

# 6) Held-out entity F1 + validation reports
python eval_scierc_f1.py
python validate_extraction.py
```

Step 5 produces, for each cutoff (NLP 2021/2022/2023, COVID 2020/2021/2022):
- `preprocessing/output/{DOMAIN}_{cutoff}_train/` — only papers with `year <= cutoff`; canonicalization runs on these only;
- `preprocessing/output/{DOMAIN}_{cutoff}_test/` — papers with `year > cutoff` (holdout), mapped onto train canonical IDs, never used to create IDs.

Each view includes a `manifest.json` with leak-check flags (`no_future_paper_in_train`, `train_test_paper_ids_disjoint`, `test_canonical_ids_subset_of_train`, `crosswalk_complete`) and source SHA-256s. If an assertion fails, the script stops — do not "fix" it by editing the assertions.

`run_scibert_pipeline.py` exists but includes an internet connectivity check and a slice-quality gate; the manual commands above are simpler and more reliable.

## 4. What to send back (keep folder structure)

- `component2_entity_relation_extraction/output/NLP_entities.json`
- `component2_entity_relation_extraction/output/NLP_relations.json`
- `component2_entity_relation_extraction/output/NLP_extracted.json`
- the same three for `COVID`
- `component2_entity_relation_extraction/output/*_entity_normalization_map.json`
- `component2_entity_relation_extraction/output/NLP_stats.json`, `COVID_stats.json`
- `component2_entity_relation_extraction/output/*_extraction_method.json`, `*_relation_method.json`
- `component2_entity_relation_extraction/scripts/checkpoints/scibert_scierc/scierc_test_f1.json`
- the whole `preprocessing/output/` folder (train + test views, manifests, crosswalks)
- `component2_entity_relation_extraction/reports/*.md`
- the stdout log of steps 1–6 (so run parameters are recorded)

Optional but useful: the fine-tuned `checkpoints/scibert_scierc/ner/model.safetensors` (~440 MB) if transfer is easy.

## 5. Sanity checks before sending

- `len(json.load(NLP_extracted.json))` ≈ **2,800**; `COVID_extracted.json` ≈ **1,200**.
- Entity and relation counts should be noticeably larger than the old run (NLP 28,463 entities / 22,690 relations; COVID 11,918 / 8,081).
- `Other`-class entity share should stay well below 55%; typed-relation share above 5%.
- All 12 cutoff manifests must show the four leak checks as `true`.
- Report the new `scierc_test_f1.json` value — the old submitted numbers were entity F1 0.597 / relation F1 0.689 and will change because the model is re-trained.

## 6. Do NOT do these

- Do not run `01_fetch_citations.py` or anything in Components 3–6 yet.
- Do not edit `common_io.py` — its paths assume this folder layout.
- Do not overwrite the included SciERC train/dev/test JSONs.
- Do not run `normalize_entities.py` with a different `--threshold` unless asked; τ=85 is the paper's setting (75/95 are for a later sensitivity test).

## 7. Verify the transfer

`CHECKSUMS.sha256` lists the SHA-256 of every file in this bundle. On the receiving machine:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

All lines should print `OK`. If not, the transfer was corrupted — re-send the bundle.
