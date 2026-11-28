# COMPONENT 2 — CROSS-MODEL COMPARISON (SciBERT vs RoBERTa vs PubMedBERT)


## SciBERT  (`allenai/scibert_scivocab_uncased`)

- SciERC held-out TEST entity F1: **0.6001** (P=0.5936, R=0.6067, n=551 sentences)
- SciERC held-out TEST relation F1: **0.6757** (P=0.6630, R=0.6889, n=3248 pairs)

**NLP domain:**
- Papers: 2800  |  Entities: 59862  |  Relations: 49563
- Typed-relation share (non-generic): **62.9%**
- 'Other' entity-type share: **34.9%**
- Entity types: Method=20955, Other=20884, Task=9943, Material=6083, Metric=1997
- Relation types: USED_FOR=21262, ENTITY_ASSOCIATED_WITH_ENTITY=18376, METHOD_EVALUATED_BY=5161, METHOD_APPLIED_TO=4764

**COVID domain:**
- Papers: 1200  |  Entities: 18602  |  Relations: 14295
- Typed-relation share (non-generic): **39.9%**
- 'Other' entity-type share: **37.3%**
- Entity types: Other=6932, Method=5950, Material=2990, Task=2343, Metric=387
- Relation types: ENTITY_ASSOCIATED_WITH_ENTITY=8591, USED_FOR=3931, METHOD_APPLIED_TO=1307, METHOD_EVALUATED_BY=466

## RoBERTa  (`roberta-base`)

- SciERC held-out TEST entity F1: **0.5913** (P=0.5649, R=0.6203, n=551 sentences)
- SciERC held-out TEST relation F1: **0.6653** (P=0.6544, R=0.6766, n=3248 pairs)

**NLP domain:**
- Papers: 2800  |  Entities: 74703  |  Relations: 76174
- Typed-relation share (non-generic): **66.3%**
- 'Other' entity-type share: **37.2%**
- Entity types: Other=27807, Method=20933, Task=16339, Material=6826, Metric=2798
- Relation types: USED_FOR=35076, ENTITY_ASSOCIATED_WITH_ENTITY=25672, METHOD_EVALUATED_BY=8173, METHOD_APPLIED_TO=7253

**COVID domain:**
- Papers: 1200  |  Entities: 28461  |  Relations: 26112
- Typed-relation share (non-generic): **45.6%**
- 'Other' entity-type share: **40.5%**
- Entity types: Other=11528, Task=6886, Method=5917, Material=3039, Metric=1091
- Relation types: ENTITY_ASSOCIATED_WITH_ENTITY=14195, USED_FOR=8275, METHOD_APPLIED_TO=2425, METHOD_EVALUATED_BY=1217

## PubMedBERT  (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)

- SciERC held-out TEST entity F1: **0.6007** (P=0.5727, R=0.6316, n=551 sentences)
- SciERC held-out TEST relation F1: **0.6603** (P=0.6870, R=0.6355, n=3248 pairs)

**NLP domain:**
- Papers: 2800  |  Entities: 66865  |  Relations: 56841
- Typed-relation share (non-generic): **59.7%**
- 'Other' entity-type share: **36.8%**
- Entity types: Other=24582, Method=19577, Task=13689, Material=6271, Metric=2746
- Relation types: ENTITY_ASSOCIATED_WITH_ENTITY=22900, USED_FOR=22091, METHOD_EVALUATED_BY=6563, METHOD_APPLIED_TO=5287

**COVID domain:**
- Papers: 1200  |  Entities: 24075  |  Relations: 20519
- Typed-relation share (non-generic): **34.7%**
- 'Other' entity-type share: **31.7%**
- Entity types: Other=7629, Task=5838, Method=5269, Material=4387, Metric=952
- Relation types: ENTITY_ASSOCIATED_WITH_ENTITY=13401, USED_FOR=4495, METHOD_APPLIED_TO=1948, METHOD_EVALUATED_BY=675