"""
Component 2 — SciERC held-out TEST-set relation F1 evaluation.

Companion to eval_scierc_f1.py (entity side). Loads each trained relation
classifier from checkpoints/{model}_scierc/relation/ and computes
precision/recall/F1 against SciERC's held-out test split (test.json),
excluding 'no_relation' from the micro-F1 average.

Usage:
    python eval_scierc_relation_f1.py --model roberta
    python eval_scierc_relation_f1.py --model pubmedbert
    python eval_scierc_relation_f1.py --model all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_relations_scibert import (
    NO_RELATION, REL_LABELS, load_scierc_relations, MAX_LEN,
)
from extract_entities_relations_multimodel import (
    load_tokenizer, build_multimodel_relation_class,
)

DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "checkpoints"

MODEL_REGISTRY = {
    "scibert":    "allenai/scibert_scivocab_uncased",
    "roberta":    "roberta-base",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}


def _load_model_for_eval(model_key: str, checkpoint_root: Path, force_cpu: bool = False):
    import torch

    rel_dir = checkpoint_root / f"{model_key}_scierc" / "relation"
    model_pt = rel_dir / "model.pt"
    labels_path = rel_dir / "labels.json"
    if not model_pt.exists():
        raise SystemExit(
            f"No trained relation model at {model_pt}. "
            f"Run extract_entities_relations_multimodel.py --model {model_key} "
            f"--step train-relation first."
        )
    if not labels_path.exists():
        raise SystemExit(f"Missing labels.json at {labels_path}.")

    labels = json.loads(labels_path.read_text())
    if labels != REL_LABELS:
        print(f"  WARNING: {labels_path} labels do not match REL_LABELS; using REL_LABELS.")
        labels = REL_LABELS

    model_name = MODEL_REGISTRY[model_key]
    tokenizer = load_tokenizer(rel_dir if (rel_dir / "tokenizer_config.json").exists() else model_name, model_key)
    ModelClass = build_multimodel_relation_class()
    model = ModelClass(model_name, len(labels))
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_pt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer, labels, device


def evaluate_one_model(model_key: str, checkpoint_root: Path, output_dir: Path,
                       scierc_test_path: Path, batch_size: int = 16, force_cpu: bool = False) -> dict:
    import torch
    from sklearn.metrics import precision_score, recall_score, f1_score

    print(f"\n=== [{model_key}] SciERC relation TEST eval ===")
    model, tokenizer, labels, device = _load_model_for_eval(model_key, checkpoint_root, force_cpu=force_cpu)
    label2id = {l: i for i, l in enumerate(labels)}

    test_examples = load_scierc_relations(scierc_test_path)
    print(f"[{model_key}] SciERC test sentences (with >=1 pair): {len(test_examples)}")

    rows = []
    n_pos = 0
    for ex in test_examples:
        for i, j, label in ex["pairs"]:
            rows.append((ex["tokens"], ex["entities"][i], ex["entities"][j], label))
            if label != NO_RELATION:
                n_pos += 1
    n_pairs = len(rows)
    print(f"[{model_key}] test pairs (pos+neg): {n_pairs}  (positives: {n_pos})")

    if n_pairs == 0:
        raise SystemExit(f"[{model_key}] No pairs found in test split — aborting.")

    y_true, y_pred = [], []
    with torch.no_grad():
        for start in range(0, n_pairs, batch_size):
            batch = rows[start:start + batch_size]
            tokens_batch = [r[0] for r in batch]
            enc = tokenizer(tokens_batch, is_split_into_words=True,
                            truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            word_ids_batch = [enc.word_ids(batch_index=i) for i in range(len(batch))]
            e1s = torch.tensor([r[1][0] for r in batch]).to(device)
            e1e = torch.tensor([r[1][1] for r in batch]).to(device)
            e2s = torch.tensor([r[2][0] for r in batch]).to(device)
            e2e = torch.tensor([r[2][1] for r in batch]).to(device)
            gold = torch.tensor([label2id[r[3]] for r in batch]).to(device)

            logits = model(enc["input_ids"], enc["attention_mask"],
                           e1s, e1e, e2s, e2e, word_ids_batch)
            pred_ids = logits.argmax(-1).cpu().tolist()
            y_pred.extend(pred_ids)
            y_true.extend(gold.cpu().tolist())

            if (start // batch_size) % 20 == 0:
                done = min(start + batch_size, n_pairs)
                print(f"  ... {done}/{n_pairs} pairs scored", flush=True)

    typed_label_ids = [label2id[l] for l in labels if l != NO_RELATION]
    p = precision_score(y_true, y_pred, labels=typed_label_ids, average="micro", zero_division=0)
    r = recall_score(y_true, y_pred, labels=typed_label_ids, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=typed_label_ids, average="micro", zero_division=0)

    report = {
        "model_key": model_key,
        "model": MODEL_REGISTRY[model_key],
        "task": "relation_classification",
        "split": "test",
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "n_test_sentences": len(test_examples),
        "n_test_pairs": n_pairs,
        "n_pos_pairs": n_pos,
        "label_set": [l for l in labels if l != NO_RELATION],
    }
    print(f"[{model_key}] SciERC TEST relation F1: "
          f"P={report['precision']:.4f}  R={report['recall']:.4f}  F1={report['f1']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_key}_scierc" / "scierc_relation_test_f1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[{model_key}] Saved -> {out_path}")
    return report


def _resolve_scierc_test_path(checkpoint_root: Path) -> Path:
    scierc_dir = checkpoint_root / "scierc_data"
    candidate = scierc_dir / "processed_data" / "json" / "test.json"
    if candidate.exists():
        return candidate
    found = list(scierc_dir.rglob("test.json"))
    if not found:
        raise SystemExit(
            f"No SciERC test.json under {scierc_dir}. Run "
            f"extract_entities_relations_multimodel.py --model <any> --step train-ner first."
        )
    return found[0]


def main():
    ap = argparse.ArgumentParser(
        description="SciERC held-out TEST-set relation F1 for scibert, roberta, pubmedbert."
    )
    ap.add_argument("--model", choices=list(MODEL_REGISTRY) + ["all"], default="all")
    ap.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--scierc-test", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    scierc_test = args.scierc_test or _resolve_scierc_test_path(args.checkpoint_root)
    print(f"Using SciERC test split: {scierc_test}")

    targets = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    results = {}
    for mk in targets:
        try:
            results[mk] = evaluate_one_model(
                mk, args.checkpoint_root, args.output_dir, scierc_test,
                batch_size=args.batch_size, force_cpu=args.cpu,
            )
        except SystemExit as e:
            print(f"  [SKIP] [{mk}]: {e}")
            results[mk] = {"model_key": mk, "status": "skipped", "reason": str(e)}

    summary_path = args.output_dir / "scierc_relation_test_f1_all.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written -> {summary_path}")


if __name__ == "__main__":
    main()
