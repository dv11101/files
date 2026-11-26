# COMPONENT 2 REPORT — Scientific Entity & Relation Extraction


## NLP domain

**Entity extraction method:** **Tier 2 production** — SciBERT (`allenai/scibert_scivocab_uncased`) fine-tuned on the SciERC corpus for entity recognition. Relation extraction still uses the Tier-1 dependency-pattern layer (not a trained relation classifier) — see Known Limitations below.

**Relation extraction method:** Tier-1 dependency-pattern heuristic (not a trained classifier) — see Known Limitations below.

- Papers in sampled corpus: **2800**
- Papers processed OK: **2800**
- Papers skipped (empty/invalid abstract): **0**
- Papers errored: **0**
- Total entities extracted: **59862**
- Total relations extracted: **26868**
- Distinct surface forms → canonical entities (post-normalization): **29099 → 24195**
- Papers with zero entities: **0**
- Papers with zero relations: **0**

**Entity type distribution:**

| Type | Count | % |
|---|---|---|
| Method | 20955 | 35.0% |
| Other | 20884 | 34.9% |
| Task | 9943 | 16.6% |
| Material | 6083 | 10.2% |
| Metric | 1997 | 3.3% |

**Relation type distribution:**

| Relation | Count | % |
|---|---|---|
| ENTITY_ASSOCIATED_WITH_ENTITY | 25293 | 94.1% |
| METHOD_IMPROVES_TASK | 959 | 3.6% |
| USED_FOR | 427 | 1.6% |
| METHOD_EVALUATED_BY | 107 | 0.4% |
| METHOD_APPLIED_TO | 82 | 0.3% |

**Per-year statistics:**

| Year | Papers | Entities | Relations | Avg Ent/Paper | Avg Rel/Paper |
|---|---|---|---|---|---|
| 2018 | 400 | 7654 | 3459 | 19.1 | 8.6 |
| 2019 | 400 | 7928 | 3554 | 19.8 | 8.9 |
| 2020 | 400 | 8314 | 3722 | 20.8 | 9.3 |
| 2021 | 400 | 8632 | 3900 | 21.6 | 9.8 |
| 2022 | 400 | 8720 | 3922 | 21.8 | 9.8 |
| 2023 | 400 | 9073 | 4059 | 22.7 | 10.1 |
| 2024 | 400 | 9541 | 4252 | 23.9 | 10.6 |

**Sample zero-entity paper IDs:** []

**Sample zero-relation paper IDs:** []

**Errored papers:** none — 0 crashes across the full run.

### NLP — Example extracted papers

**Paper:** Incorporating Latent Meanings of Morphological Compositions to Enhance Word Embeddings  
*(paper_id=P18-1.114, year=2018, domain=NLP)*

Entities:
- Latent Meanings of Morphological Compositions — **Other** (conf 0.8981)
- Word Embeddings — **Task** (conf 0.6892)
- word embedding approaches — **Method** (conf 0.985)
- semantic information — **Other** (conf 0.9447)
- word level — **Other** (conf 0.729)
- morphemes — **Other** (conf 0.8171)
- morphology-based models — **Method** (conf 0.9858)
- morphemes — **Other** (conf 0.9515)
- word embeddings — **Task** (conf 0.4672)
- latent meanings of morphemes — **Other** (conf 0.8811)

Relations:
- Latent Meanings of Morphological Compositions → **ENTITY_ASSOCIATED_WITH_ENTITY** → Word Embeddings  (conf 0.3)
- word embedding approaches → **ENTITY_ASSOCIATED_WITH_ENTITY** → semantic information  (conf 0.3)
- semantic information → **ENTITY_ASSOCIATED_WITH_ENTITY** → word level  (conf 0.3)
- morphology-based models → **ENTITY_ASSOCIATED_WITH_ENTITY** → morphemes  (conf 0.3)
- morphemes → **ENTITY_ASSOCIATED_WITH_ENTITY** → word embeddings  (conf 0.3)
- latent meanings of morphological compositions of words → **ENTITY_ASSOCIATED_WITH_ENTITY** → word embeddings  (conf 0.3)
- Latent Meaning Models → **ENTITY_ASSOCIATED_WITH_ENTITY** → LMMs  (conf 0.3)
- LMMs → **ENTITY_ASSOCIATED_WITH_ENTITY** → LMM-  (conf 0.3)

---

**Paper:** BERT-EMD: Many-to-Many Layer Mapping for BERT Compression with Earth Mover’s Distance  
*(paper_id=2020.emnlp-main.242, year=2020, domain=NLP)*

Entities:
- BERT-EMD — **Method** (conf 0.9749)
- Many-to-Many Layer Mapping — **Method** (conf 0.8186)
- BERT Compression — **Task** (conf 0.8201)
- Earth Mover’s Distance — **Other** (conf 0.6896)
- Pre-trained language models — **Method** (conf 0.8697)
- BERT — **Method** (conf 0.9472)
- natural language processing (NLP) tasks — **Task** (conf 0.9458)
- storage — **Metric** (conf 0.6574)
- computational — **Metric** (conf 0.5047)
- pre-trained language models — **Method** (conf 0.8217)

Relations:
- BERT-EMD → **ENTITY_ASSOCIATED_WITH_ENTITY** → Many-to-Many Layer Mapping  (conf 0.3)
- Many-to-Many Layer Mapping → **ENTITY_ASSOCIATED_WITH_ENTITY** → BERT Compression  (conf 0.3)
- Pre-trained language models → **ENTITY_ASSOCIATED_WITH_ENTITY** → BERT  (conf 0.3)
- BERT → **ENTITY_ASSOCIATED_WITH_ENTITY** → natural language processing (NLP) tasks  (conf 0.3)
- storage → **ENTITY_ASSOCIATED_WITH_ENTITY** → computational  (conf 0.3)
- computational → **ENTITY_ASSOCIATED_WITH_ENTITY** → pre-trained language models  (conf 0.3)
- BERT distillation method → **ENTITY_ASSOCIATED_WITH_ENTITY** → many-to-many layer mapping  (conf 0.3)
- model → **ENTITY_ASSOCIATED_WITH_ENTITY** → teacher layers  (conf 0.3)

---

**Paper:** Rethinking Multi-Modal Alignment in Multi-Choice VideoQA from Feature and Sample Perspectives  
*(paper_id=2022.emnlp-main.561, year=2022, domain=NLP)*

Entities:
- Multi-Modal Alignment — **Task** (conf 0.8897)
- Multi-Choice VideoQA — **Task** (conf 0.8626)
- Feature and Sample Perspectives — **Other** (conf 0.8215)
- causal and temporal event relations in videos — **Other** (conf 0.6959)
- Video Question Answering — **Task** (conf 0.9282)
- VideoQA — **Task** (conf 0.8552)
- semantic gap — **Other** (conf 0.904)
- language — **Other** (conf 0.3731)
- architectures — **Other** (conf 0.6905)
- frame- or object-level visual representations — **Method** (conf 0.8991)

Relations:
- Multi-Modal Alignment → **ENTITY_ASSOCIATED_WITH_ENTITY** → Multi-Choice VideoQA  (conf 0.3)
- Multi-Choice VideoQA → **ENTITY_ASSOCIATED_WITH_ENTITY** → Feature and Sample Perspectives  (conf 0.3)
- causal and temporal event relations in videos → **ENTITY_ASSOCIATED_WITH_ENTITY** → Video Question Answering  (conf 0.3)
- Video Question Answering → **ENTITY_ASSOCIATED_WITH_ENTITY** → VideoQA  (conf 0.3)
- semantic gap → **ENTITY_ASSOCIATED_WITH_ENTITY** → language  (conf 0.3)
- architectures → **ENTITY_ASSOCIATED_WITH_ENTITY** → frame- or object-level visual representations  (conf 0.3)
- multi-modal alignment problem → **ENTITY_ASSOCIATED_WITH_ENTITY** → VideoQA  (conf 0.3)
- VideoQA → **ENTITY_ASSOCIATED_WITH_ENTITY** → feature and sample perspectives  (conf 0.3)

---


## COVID domain

**Entity extraction method:** **Tier 2 production** — SciBERT (`allenai/scibert_scivocab_uncased`) fine-tuned on the SciERC corpus for entity recognition. Relation extraction still uses the Tier-1 dependency-pattern layer (not a trained relation classifier) — see Known Limitations below.

**Relation extraction method:** Tier-1 dependency-pattern heuristic (not a trained classifier) — see Known Limitations below.

- Papers in sampled corpus: **1200**
- Papers processed OK: **1200**
- Papers skipped (empty/invalid abstract): **0**
- Papers errored: **0**
- Total entities extracted: **18602**
- Total relations extracted: **7711**
- Distinct surface forms → canonical entities (post-normalization): **9779 → 8688**
- Papers with zero entities: **6**
- Papers with zero relations: **46**

**Entity type distribution:**

| Type | Count | % |
|---|---|---|
| Other | 6932 | 37.3% |
| Method | 5950 | 32.0% |
| Material | 2990 | 16.1% |
| Task | 2343 | 12.6% |
| Metric | 387 | 2.1% |

**Relation type distribution:**

| Relation | Count | % |
|---|---|---|
| ENTITY_ASSOCIATED_WITH_ENTITY | 7466 | 96.8% |
| USED_FOR | 118 | 1.5% |
| METHOD_IMPROVES_TASK | 60 | 0.8% |
| METHOD_APPLIED_TO | 50 | 0.6% |
| METHOD_EVALUATED_BY | 17 | 0.2% |

**Per-year statistics:**

| Year | Papers | Entities | Relations | Avg Ent/Paper | Avg Rel/Paper |
|---|---|---|---|---|---|
| 2019 | 200 | 2720 | 1113 | 13.6 | 5.6 |
| 2020 | 200 | 2651 | 1091 | 13.3 | 5.5 |
| 2021 | 200 | 2941 | 1230 | 14.7 | 6.2 |
| 2022 | 200 | 3344 | 1386 | 16.7 | 6.9 |
| 2023 | 200 | 3374 | 1397 | 16.9 | 7.0 |
| 2024 | 200 | 3572 | 1494 | 17.9 | 7.5 |

**Sample zero-entity paper IDs:** ['openalex_W2917867348', 'openalex_W2914022334', 'openalex_W2772116241', 'openalex_W3015180407', 'openalex_W2997210103', 'gen_b15adec091ba']

**Sample zero-relation paper IDs:** ['openalex_W2909090086', 'openalex_W2917867348', 'openalex_W2942506922', 'openalex_W3012233527', 'openalex_W2944465352', 'openalex_W2890196916', 'openalex_W2914022334', 'openalex_W2948006344', 'openalex_W2772116241', 'openalex_W4403595011', 'openalex_W2976746499', 'openalex_W2948162576', 'openalex_W2912761948', 'openalex_W3015180407', 'openalex_W3158523910']

**Errored papers:** none — 0 crashes across the full run.

### COVID — Example extracted papers

**Paper:** Optimized Score Transformation for Consistent Fair Classification  
*(paper_id=openalex_W3208316297, year=2019, domain=COVID)*

Entities:
- Optimized Score Transformation — **Method** (conf 0.7799)
- Consistent Fair Classification — **Task** (conf 0.9128)
- fair probabilistic binary classification — **Task** (conf 0.9201)
- predicted — **Other** (conf 0.7059)
- fairness constraints — **Other** (conf 0.9103)
- conditional means of scores — **Other** (conf 0.728)
- cross — **Other** (conf 0.6112)
- formulation — **Other** (conf 0.9159)
- pre-processing extension — **Method** (conf 0.7896)
- classification algorithm — **Method** (conf 0.928)

Relations:
- Optimized Score Transformation → **ENTITY_ASSOCIATED_WITH_ENTITY** → Consistent Fair Classification  (conf 0.3)
- fair probabilistic binary classification → **ENTITY_ASSOCIATED_WITH_ENTITY** → predicted  (conf 0.3)
- fairness constraints → **ENTITY_ASSOCIATED_WITH_ENTITY** → conditional means of scores  (conf 0.3)
- conditional means of scores → **ENTITY_ASSOCIATED_WITH_ENTITY** → cross  (conf 0.3)
- formulation → **ENTITY_ASSOCIATED_WITH_ENTITY** → pre-processing extension  (conf 0.3)
- pre-processing extension → **ENTITY_ASSOCIATED_WITH_ENTITY** → classification algorithm  (conf 0.3)
- transformed scores → **ENTITY_ASSOCIATED_WITH_ENTITY** → convex  (conf 0.3)
- convex → **ENTITY_ASSOCIATED_WITH_ENTITY** → transformation parameters  (conf 0.3)

---

**Paper:** A novel augmented deep transfer learning for classification of COVID-19 and other thoracic diseases from X-rays  
*(paper_id=openalex_W3159381686, year=2021, domain=COVID)*

Entities:
- augmented deep transfer learning — **Method** (conf 0.9685)
- classification of COVID-19 and other thoracic diseases — **Task** (conf 0.6591)
- X — **Other** (conf 0.4479)
- Deep learning — **Method** (conf 0.9672)
- natural imaging tasks — **Task** (conf 0.9332)
- medical images — **Material** (conf 0.9009)
- annotated training data — **Material** (conf 0.7593)
- Transfer learning — **Method** (conf 0.9723)
- medical imaging tasks — **Task** (conf 0.9314)
- natural images — **Material** (conf 0.8092)

Relations:
- augmented deep transfer learning → **ENTITY_ASSOCIATED_WITH_ENTITY** → classification of COVID-19 and other thoracic diseases  (conf 0.3)
- classification of COVID-19 and other thoracic diseases → **ENTITY_ASSOCIATED_WITH_ENTITY** → X  (conf 0.3)
- Deep learning → **ENTITY_ASSOCIATED_WITH_ENTITY** → natural imaging tasks  (conf 0.3)
- medical images → **ENTITY_ASSOCIATED_WITH_ENTITY** → annotated training data  (conf 0.3)
- Transfer learning → **METHOD_APPLIED_TO** → medical imaging tasks  (conf 0.6)
- natural images → **ENTITY_ASSOCIATED_WITH_ENTITY** → medical images  (conf 0.3)
- medical images → **ENTITY_ASSOCIATED_WITH_ENTITY** → transfer  (conf 0.3)
- annotated data → **ENTITY_ASSOCIATED_WITH_ENTITY** → medical imaging tasks  (conf 0.3)

---

**Paper:** A Review on Electronic Health Record Text-Mining for Biomedical Name Entity Recognition in Healthcare Domain  
*(paper_id=openalex_W4367310581, year=2023, domain=COVID)*

Entities:
- Electronic Health Record Text-Mining — **Task** (conf 0.7285)
- Biomedical Name Entity Recognition — **Task** (conf 0.959)
- Healthcare Domain — **Material** (conf 0.4684)
- Biomedical-named entity recognition ( — **Method** (conf 0.6745)
- bNER) — **Method** (conf 0.7751)
- biomedical informatics — **Task** (conf 0.8439)
- It — **Other** (conf 0.794)
- biomedical entities — **Other** (conf 0.5761)
- electronic health records — **Material** (conf 0.7398)
- bNER — **Method** (conf 0.906)

Relations:
- Electronic Health Record Text-Mining → **ENTITY_ASSOCIATED_WITH_ENTITY** → Biomedical Name Entity Recognition  (conf 0.3)
- Biomedical Name Entity Recognition → **ENTITY_ASSOCIATED_WITH_ENTITY** → Healthcare Domain  (conf 0.3)
- Biomedical-named entity recognition ( → **ENTITY_ASSOCIATED_WITH_ENTITY** → bNER)  (conf 0.3)
- bNER) → **ENTITY_ASSOCIATED_WITH_ENTITY** → biomedical informatics  (conf 0.3)
- It → **ENTITY_ASSOCIATED_WITH_ENTITY** → biomedical entities  (conf 0.3)
- biomedical entities → **ENTITY_ASSOCIATED_WITH_ENTITY** → electronic health records  (conf 0.3)
- bNER → **ENTITY_ASSOCIATED_WITH_ENTITY** → computational methods  (conf 0.3)
- computational methods → **ENTITY_ASSOCIATED_WITH_ENTITY** → Information Technology  (conf 0.3)

---


## Known limitations

- Entities: SciBERT fine-tuned on SciERC (learned, not lexicon-based). See `eval_scierc_f1.py` output for held-out entity F1.

- Relations: STILL the Tier-1 dependency-pattern heuristic, not a trained classifier — same-sentence co-occurrence dominates (`ENTITY_ASSOCIATED_WITH_ENTITY`), which is why typed relations are a small minority. Run `extract_relations_scibert.py --train --run` to replace this with a trained relation classifier.

- No coreference resolution, no external KB linking (e.g. UMLS).


## What Component 3 should consume

- **Nodes**: one node per distinct `canonical_id` in `{NLP,COVID}_entities.json`.
- **Edges**: `{NLP,COVID}_relations.json`, remapped to `canonical_id` via the entity lookup.
- **Temporal snapshot key**: `year` on every entity/relation record.
- **Provenance**: `paper_id` + `domain` preserved everywhere.
- **Surface-form provenance**: `{NLP,COVID}_entity_normalization_map.json`.
