"""
Component 2 — C1: cutoff-local preprocessing with train/test (holdout) split.

For every (domain, cutoff) this script writes two views:

  preprocessing/output/{DOMAIN}_{cutoff}_train/
      {DOMAIN}_entities.json                papers with year <= cutoff
      {DOMAIN}_relations.json
      {DOMAIN}_extracted.json
      {DOMAIN}_entity_normalization_map.json   normalized_form -> canonical_id
      {DOMAIN}_canonical_crosswalk.json        canonical_id -> {surface_form, type, n_mentions}
      manifest.json

  preprocessing/output/{DOMAIN}_{cutoff}_test/
      {DOMAIN}_entities.json                papers with year > cutoff (holdout only)
      {DOMAIN}_relations.json
      {DOMAIN}_extracted.json
      {DOMAIN}_entity_normalization_map.json   train-derived map (for validation only)
      manifest.json

Leak-freedom rules (all asserted at runtime):
  1. Canonicalization runs on TRAIN entities ONLY. No paper with year > cutoff
     contributes a surface form, a canonical ID, or a cluster anchor.
  2. TEST entities keep their extracted surface forms but are mapped onto the
     TRAIN-derived canonical IDs by normalized_form. A mention unseen in train
     gets canonical_id=None and mapped_in_train=false; it is never used to
     create a new canonical node.
  3. Train paper IDs and test paper IDs are disjoint and together cover the
     source corpus.
  4. A crosswalk is written from TRAIN data only, so concept pairs can be
     compared across cutoffs by a stable surface-form label.

This is the C1 fix from REVIEWER_CHANGES_CODEWISE.md. The pipeline's gap list
at cutoff c must consume the *_train view; the *_test view is for the
retrospective validation hit check and must never enter canonicalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_io import read_json, write_json
from normalize_entities import cluster_domain, FUZZY_THRESHOLD

PROJECT_ROOT = SCRIPT_DIR.parents[1]          # .../sdp
DEFAULT_SOURCE = PROJECT_ROOT / "component2_entity_relation_extraction" / "output"
DEFAULT_OUT = PROJECT_ROOT / "preprocessing" / "output"

DOMAINS = ("NLP", "COVID")
CUTOFFS = {"NLP": (2021, 2022, 2023), "COVID": (2020, 2021, 2022)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_by_year(records: list[dict], cutoff: int) -> tuple[list[dict], list[dict]]:
    train = [r for r in records if int(r.get("year", 0)) <= cutoff]
    test = [r for r in records if int(r.get("year", 0)) > cutoff]
    return train, test


def build_crosswalk(entities: list[dict]) -> dict[str, dict[str, Any]]:
    """Train-only canonical_id -> most frequent surface form + type + count."""
    surfaces: dict[str, Counter] = defaultdict(Counter)
    types: dict[str, Counter] = defaultdict(Counter)
    for e in entities:
        cid = e.get("canonical_id")
        if not cid:
            continue
        surfaces[cid][str(e.get("surface_form", ""))] += 1
        types[cid][str(e.get("type", "Other"))] += 1
    out: dict[str, dict[str, Any]] = {}
    for cid, cnt in surfaces.items():
        out[cid] = {
            "surface_form": cnt.most_common(1)[0][0],
            "type": types[cid].most_common(1)[0][0],
            "n_train_mentions": sum(cnt.values()),
            "n_train_surface_forms": len(cnt),
        }
    return out


def map_test_entities(entities: list[dict], train_map: dict[str, str]) -> tuple[list[dict], int]:
    """Attach TRAIN canonical IDs to holdout mentions; never create new IDs here."""
    mapped = 0
    out = []
    for e in entities:
        rec = dict(e)
        cid = train_map.get(str(e.get("normalized_form", "")))
        rec["canonical_id"] = cid
        rec["mapped_in_train"] = cid is not None
        if cid is not None:
            mapped += 1
        out.append(rec)
    return out, mapped


def one_view(domain: str, cutoff: int, source: Path, out_root: Path,
             threshold: float, force: bool) -> dict[str, Any]:
    t0 = time.time()
    train_dir = out_root / f"{domain}_{cutoff}_train"
    test_dir = out_root / f"{domain}_{cutoff}_test"
    if (train_dir.exists() or test_dir.exists()) and not force:
        print(f"  [SKIP] {domain} {cutoff}: views exist (use --force to rebuild)")
        return {"skipped": True, "domain": domain, "cutoff": cutoff}

    entities = read_json(source / f"{domain}_entities.json")
    extracted = read_json(source / f"{domain}_extracted.json")
    relations = read_json(source / f"{domain}_relations.json")
    src_hashes = {
        name: sha256(source / f"{domain}_{name}.json")
        for name in ("entities", "extracted", "relations")
    }

    tr_e, te_e = split_by_year(entities, cutoff)
    tr_x, te_x = split_by_year(extracted, cutoff)
    tr_r, te_r = split_by_year(relations, cutoff)

    # --- leak-free canonicalization: TRAIN ONLY ---------------------------
    tr_e, train_map = cluster_domain(tr_e, threshold=threshold)
    crosswalk = build_crosswalk(tr_e)
    eid_to_canon = {e["entity_id"]: e.get("canonical_id") for e in tr_e}

    for paper in tr_x:
        for e in paper.get("entities", []):
            e["canonical_id"] = eid_to_canon.get(e["entity_id"])

    # --- holdout: map onto TRAIN IDs, never cluster -----------------------
    te_e, n_e_mapped = map_test_entities(te_e, train_map)
    for paper in te_x:
        mapped, n = map_test_entities(paper.get("entities", []), train_map)
        paper["entities"] = mapped
        for rel in paper.get("relations", []):
            src = eid_to_canon.get(rel.get("source_entity_id"))
            tgt = eid_to_canon.get(rel.get("target_entity_id"))
            rel["train_canonical_source"] = src
            rel["train_canonical_target"] = tgt

    # --- assertions (fail loudly on any leak) ----------------------------
    assert all(int(r.get("year", 0)) <= cutoff for r in tr_e), "train entity from future year"
    assert all(int(p.get("year", 0)) <= cutoff for p in tr_x), "train paper from future year"
    assert all(int(r.get("year", 0)) <= cutoff for r in tr_r), "train relation from future year"
    assert all(int(r.get("year", 0)) > cutoff for r in te_e), "test entity in train years"
    assert all(int(p.get("year", 0)) > cutoff for p in te_x), "test paper in train years"
    assert all(int(r.get("year", 0)) > cutoff for r in te_r), "test relation in train years"
    train_ids = {p["paper_id"] for p in tr_x}
    test_ids = {p["paper_id"] for p in te_x}
    assert train_ids.isdisjoint(test_ids), "paper appears in both train and test"
    assert train_ids | test_ids == {p["paper_id"] for p in extracted}, "paper lost in split"
    train_canons = set(train_map.values())
    assert all(e["canonical_id"] in train_canons for e in te_e if e["canonical_id"]), "test got unseen canonical ID"
    assert set(crosswalk) == train_canons, "crosswalk incomplete"

    # --- write -----------------------------------------------------------
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    write_json(train_dir / f"{domain}_entities.json", tr_e)
    write_json(train_dir / f"{domain}_relations.json", tr_r)
    write_json(train_dir / f"{domain}_extracted.json", tr_x)
    write_json(train_dir / f"{domain}_entity_normalization_map.json", train_map)
    write_json(train_dir / f"{domain}_canonical_crosswalk.json", crosswalk)

    write_json(test_dir / f"{domain}_entities.json", te_e)
    write_json(test_dir / f"{domain}_relations.json", te_r)
    write_json(test_dir / f"{domain}_extracted.json", te_x)
    write_json(test_dir / f"{domain}_entity_normalization_map.json", train_map)

    manifest = {
        "domain": domain,
        "cutoff_year": cutoff,
        "threshold": threshold,
        "train_rule": "year <= cutoff",
        "test_rule": "year > cutoff",
        "source_dir": str(source),
        "source_sha256": src_hashes,
        "rebuild_after_reextraction": True,
        "note": "Source C2 outputs may predate the enlarged corpus. Re-run this "
                "script after the friend returns new extraction outputs.",
        "counts": {
            "source_entities": len(entities), "source_papers": len(extracted),
            "source_relations": len(relations),
            "train_entities": len(tr_e), "train_papers": len(tr_x), "train_relations": len(tr_r),
            "test_entities": len(te_e), "test_papers": len(te_x), "test_relations": len(te_r),
            "train_surface_forms": len(train_map), "train_canonical_entities": len(train_canons),
            "test_entities_mapped_to_train": n_e_mapped,
            "test_entities_unmapped": len(te_e) - n_e_mapped,
        },
        "year_ranges": {
            "train": [min((r["year"] for r in tr_x), default=None),
                      max((r["year"] for r in tr_x), default=None)],
            "test": [min((r["year"] for r in te_x), default=None),
                     max((r["year"] for r in te_x), default=None)],
        },
        "leak_checks": {
            "no_future_paper_in_train": True,
            "train_test_paper_ids_disjoint": True,
            "test_canonical_ids_subset_of_train": True,
            "crosswalk_complete": True,
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    write_json(train_dir / "manifest.json", manifest)
    write_json(test_dir / "manifest.json", manifest)
    print(f"  [{domain} {cutoff}] train={len(tr_x)} papers / {len(tr_e)} entities -> {train_dir.name}; "
          f"test={len(te_x)} papers / {len(te_e)} entities -> {test_dir.name}; "
          f"mapped={n_e_mapped}/{len(te_e)}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Build C1 cutoff-local train/test views.")
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE,
                   help="directory with canonical C2 *_entities.json / *_extracted.json / *_relations.json")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help="root for the generated *_train/ and *_test/ views")
    p.add_argument("--threshold", type=float, default=FUZZY_THRESHOLD,
                   help="RapidFuzz token-sort threshold (default: 85.0)")
    p.add_argument("--domain", choices=DOMAINS, default=None)
    p.add_argument("--cutoff", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    domains = [args.domain] if args.domain else list(DOMAINS)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"C1 cutoff views | source={args.source_dir} | out={out_root} | tau={args.threshold}")
    print("=" * 70)
    for domain in domains:
        cutoffs = [args.cutoff] if args.cutoff is not None else CUTOFFS[domain]
        for cutoff in cutoffs:
            one_view(domain, cutoff, args.source_dir, out_root, args.threshold, args.force)
    print("\nDone. Every cutoff has a *_train/ view (leak-free) and a *_test/ holdout view.")


if __name__ == "__main__":
    main()
