#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid-search d'un seuil sur la variance du Laplacien pour maximiser la performance.

Lecture: evaluation/dataset/result/results.csv (par défaut) avec colonnes:
- laplacian_variance (float)
- label_blurry (0/1)

Règle de décision: prédire flou (1) si laplacian_variance < threshold, sinon net (0).
Par défaut, optimise l'accuracy, mais peut optimiser f1/precision/recall via --metric.
"""

import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

DEFAULT_RESULTS_CSV = Path(__file__).parent / "dataset" / "result" / "results.csv"


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() == "none":
            return None
        x = float(s)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _metrics(pairs: List[Tuple[float, int]], thr: float) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    for var, y in pairs:
        yhat = 1 if var < thr else 0
        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 0:
            tn += 1
        elif y == 0 and yhat == 1:
            fp += 1
        elif y == 1 and yhat == 0:
            fn += 1
    total = tp + tn + fp + fn
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    accuracy = ((tp + tn) / total) if total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "threshold": thr,
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def _format_metrics(res: Dict[str, float]) -> str:
    return (
        f"thr={res['threshold']:.6f} | acc={res['accuracy']:.4f} "
        f"P={res['precision']:.4f} R={res['recall']:.4f} F1={res['f1']:.4f} "
        f"(tp={res['tp']} tn={res['tn']} fp={res['fp']} fn={res['fn']})"
    )


def _candidate_thresholds(values: List[float]) -> List[float]:
    """Milieux entre valeurs uniques triées, plus bornes extrêmes.
    Ainsi, la prédiction reste constante entre deux seuils adjacents.
    """
    xs = sorted(set(values))
    if not xs:
        return []
    cands: List[float] = []
    # Valeur avant la plus petite (tout net)
    cands.append(xs[0] - 1e-12)
    # Milieux entre adjacentes
    for a, b in zip(xs, xs[1:]):
        mid = (a + b) / 2.0
        cands.append(mid)
    # Valeur après la plus grande (tout flou)
    cands.append(xs[-1] + 1e-12)
    return cands


def load_pairs(csv_path: Path) -> List[Tuple[float, int]]:
    pairs: List[Tuple[float, int]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = _safe_int(row.get("label_blurry"))
            var = _safe_float(row.get("laplacian_variance"))
            if y is None or var is None:
                continue
            # Filtre conservateur (variance non négative attendue)
            if var < 0:
                continue
            pairs.append((var, y))
    return pairs


def pick_best_by(metric: str, results: List[Dict[str, float]]) -> Dict[str, float]:
    key = metric.lower()
    if key not in {"accuracy", "f1", "precision", "recall"}:
        key = "accuracy"
    # max par métrique, tie-breaker: seuil médian (plus petit thr pour stabilité)
    best = None
    for r in results:
        if best is None or r[key] > best[key] or (r[key] == best[key] and r["threshold"] < best["threshold"]):
            best = r
    return best or {}


def main():
    # Args simples: [csv_path] [metric]
    csv_path = DEFAULT_RESULTS_CSV
    metric = "accuracy"
    if len(sys.argv) >= 2:
        csv_path = Path(sys.argv[1]).resolve()
    if len(sys.argv) >= 3:
        metric = sys.argv[2].lower()

    if not csv_path.exists():
        print(f"Fichier CSV introuvable: {csv_path}")
        print("Usage: poetry run python evaluation/grid_search_laplace_threshold.py [csv_path] [metric=accuracy|f1|precision|recall]")
        sys.exit(1)

    pairs = load_pairs(csv_path)
    if not pairs:
        print("Aucun échantillon valide (laplacian_variance manquant ou labels manquants).")
        sys.exit(2)

    values = [v for v, _ in pairs]
    cands = _candidate_thresholds(values)
    if not cands:
        print("Impossible de générer des seuils candidats.")
        sys.exit(3)

    results = [_metrics(pairs, thr) for thr in cands]
    best = pick_best_by(metric, results)

    print("Grid-search Laplacian variance threshold")
    print("----------------------------------------")
    print(f"Fichier: {csv_path}")
    print(f"Optimisation: {metric}")
    print(f"Nb échantillons: {len(pairs)} | Nb seuils testés: {len(cands)}")
    print()
    print("Top 5 (selon métrique):")
    top = sorted(results, key=lambda r: (-r.get(metric, 0.0), r["threshold"]))[:5]
    for r in top:
        print("  ", _format_metrics(r))
    print()
    print("Meilleur seuil:")
    print("  ", _format_metrics(best))
    print()
    print("Note: décision utilisée = (variance < seuil) => flou (1).")


if __name__ == "__main__":
    main()

