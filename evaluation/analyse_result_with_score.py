#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des résultats à partir d'un CSV (par défaut evaluation/dataset/result/results.csv).
Calcule: total, TP, TN, FP, FN, précision, rappel, accuracy et F1-score.
Compatible avec le CSV produit par evaluation/main.py et evaluation/evaluate_cnn_dataset.py.
"""

import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_RESULTS_CSV = Path(__file__).parent / "dataset" / "result" / "results.csv"


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def compute_confusion_from_csv(csv_path: Path) -> Tuple[int, int, int, int, int]:
    """Retourne (total, tp, tn, fp, fn) à partir du CSV de résultats.
    Lit les colonnes: label_blurry (0/1) et predicted_is_blurry (0/1).
    Ignore les lignes invalides ou incomplètes.
    """
    total = tp = tn = fp = fn = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = _safe_int(row.get("label_blurry"))
            yhat = _safe_int(row.get("predicted_is_blurry"))
            if y is None or yhat is None:
                continue
            total += 1
            if y == 1 and yhat == 1:
                tp += 1
            elif y == 0 and yhat == 0:
                tn += 1
            elif y == 0 and yhat == 1:
                fp += 1
            elif y == 1 and yhat == 0:
                fn += 1

    return total, tp, tn, fp, fn


def _fmt(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:.4f}"


def main():
    csv_path = DEFAULT_RESULTS_CSV
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1]).resolve()

    if not csv_path.exists():
        print(f"Fichier de résultats introuvable: {csv_path}")
        print("Utilisation: poetry run python evaluation/analyse_result_with_score.py [chemin/vers/results.csv]")
        sys.exit(1)

    total, tp, tn, fp, fn = compute_confusion_from_csv(csv_path)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    accuracy = ((tp + tn) / total) if total > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision is not None and recall is not None and (precision + recall) > 0) else None

    print("Résumé des performances (CSV)")
    print("----------------------------")
    print(f"Fichier: {csv_path}")
    print(f"Total échantillons: {total}")
    print(f"True Positifs (TP): {tp}")
    print(f"True Négatifs (TN): {tn}")
    print(f"False Positifs (FP): {fp}")
    print(f"False Négatifs (FN): {fn}")
    print()
    print(f"Précision (precision) = TP / (TP + FP): {_fmt(precision)}")
    print(f"Rappel (recall) = TP / (TP + FN):      {_fmt(recall)}")
    print(f"Exactitude (accuracy) = (TP+TN)/Total: {_fmt(accuracy)}")
    print(f"F1-score = 2*P*R/(P+R):               {_fmt(f1)}")


if __name__ == "__main__":
    main()

