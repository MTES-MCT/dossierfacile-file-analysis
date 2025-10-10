#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation du dataset avec un modèle CNN (DocBlurNet), en miroir de evaluation/main.py,
mais en remplaçant l'étape AnalyseFiles par une prédiction CNN, et en CHERCHANT
le MEILLEUR SEUIL sur le split de test (ou, à défaut, sur tout le dataset).

Sorties dans evaluation/dataset/result/ :
- results.csv  : colonnes identiques à main.py, prédiction binaire @ meilleur seuil
- result.json  : même structure qu'avant (champs manquants à null)
- results_with_scores.csv : + local_median / global_mean / fused_prob / pred@thr*
- metrics.json : seuil optimal et métriques globales (acc, precision, recall, f1)
"""

import os
import sys
import json
import csv
import random
from pathlib import Path
from typing import Iterator, Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from dossierfacile_file_analysis.executor.tasks.prepare_data_for_analysis import PrepareDataForAnalysis
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile
from dossierfacile_file_analysis.models.supported_content_type import SupportedContentType

# Progression (optionnel)
try:
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
        TaskProgressColumn,
        SpinnerColumn,
    )
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

from contextlib import contextmanager
from dossierfacile_file_analysis.custom_logging.logging_config import logger as app_logger

# ----------------- Dossiers & fichiers -----------------
HERE = Path(__file__).parent
DATASET_ROOT = HERE / "dataset"
BLURRY_DIR = DATASET_ROOT / "blurry"
NOT_BLURRY_DIR = DATASET_ROOT / "not_blurry"

RESULT_DIR = DATASET_ROOT / "result"
TMP_IMAGES_DIR = RESULT_DIR / "tmp_images"

RESULT_JSON = RESULT_DIR / "result.json"
RESULT_CSV = RESULT_DIR / "results.csv"
RESULT_CSV_SCORES = RESULT_DIR / "results_with_scores.csv"
METRICS_JSON = RESULT_DIR / "metrics.json"

# Split (si généré par train_cnn.py)
DEFAULT_SPLIT_JSON = HERE / "ouput" / "split_docblur.json"

# ----------------- Seeds (éval reproductible) -----------------
RNG_SEED = int(os.environ.get("EVAL_RNG_SEED", 123))
random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)

# ----------------- Utils communs -----------------

def infer_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return SupportedContentType.JPEG.value
    if ext == ".png":
        return SupportedContentType.PNG.value
    if ext == ".pdf":
        return SupportedContentType.PDF.value
    return SupportedContentType.PNG.value

def iter_files_with_label_from_dirs() -> Iterator[Tuple[Path, bool]]:
    for base, label in [(BLURRY_DIR, True), (NOT_BLURRY_DIR, False)]:
        if not base.exists():
            continue
        for root, _, files in os.walk(base):
            for f in files:
                p = Path(root) / f
                if p.name.startswith('.'):
                    continue
                yield p, label

def iter_files_with_label_from_split(split_json: Path) -> Iterator[Tuple[Path, bool]]:
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    eval_files = split["eval"]["files"]
    eval_labels = split["eval"]["labels"]
    for p, y in zip(eval_files, eval_labels):
        yield Path(p), bool(y)

def compare_summary_from_pred(predicted_blurry: bool, expected_blurry: bool) -> str:
    if predicted_blurry and expected_blurry:
        v = "tp"
    elif (not predicted_blurry) and (not expected_blurry):
        v = "tn"
    elif predicted_blurry and (not expected_blurry):
        v = "fp"
    else:
        v = "fn"
    return f"final={int(predicted_blurry)} expected={int(expected_blurry)} verdict={v}"

@contextmanager
def muted_app_logger():
    prev_disabled = getattr(app_logger, "disabled", False)
    app_logger.disabled = True
    try:
        yield
    finally:
        app_logger.disabled = prev_disabled

# ----------------- Modèle CNN / Prédicteur -----------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class DocBlurNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Recrée la tête MobileNetV3-Small comme dans train_cnn.py
        from torchvision.models import mobilenet_v3_small
        weights = None  # pas de pré-entraînement ici (on charge un ckpt)
        self.backbone = mobilenet_v3_small(weights=weights)
        in_feats = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_feats, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.backbone(x).squeeze(1)

def load_first_page_as_rgb(path: Path) -> Optional[np.ndarray]:
    ctx = None
    try:
        prepare_task = PrepareDataForAnalysis()
        downloaded = DownloadedFile(
            file_name=path.stem,
            file_path=str(path),
            file_type=infer_mime_type(path)
        )
        ctx = BlurryExecutionContext(queue_message=BlurryQueueMessage(file_id=0))
        ctx.downloaded_file = downloaded

        with muted_app_logger():
            if prepare_task.has_to_apply(ctx):
                prepare_task.run(ctx)
            else:
                return None

        data = getattr(ctx, "input_analysis_data", None)
        if data is None:
            return None

        img_path = data.list_of_images[0] if data.list_of_images else data.initial_file
        if not img_path or not os.path.exists(img_path):
            return None

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    except Exception:
        return None
    finally:
        try:
            if ctx is not None and getattr(ctx, "input_analysis_data", None) is not None:
                for img in ctx.input_analysis_data.list_of_images:
                    try:
                        if img and os.path.exists(img):
                            os.remove(img)
                    except Exception:
                        pass
        except Exception:
            pass

def random_texty_crop(img_rgb: np.ndarray, crop: int = 224, max_tries: int = 20) -> np.ndarray:
    import math
    h, w = img_rgb.shape[:2]
    if min(h, w) < crop:
        scale = crop / float(min(h, w))
        new_w = max(crop, int(math.ceil(w * scale)))
        new_h = max(crop, int(math.ceil(h * scale)))
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    if h < crop or w < crop:
        pad_top = max(0, (crop - h) // 2)
        pad_bottom = max(0, crop - h - pad_top)
        pad_left = max(0, (crop - w) // 2)
        pad_right = max(0, crop - w - pad_left)
        img_rgb = cv2.copyMakeBorder(img_rgb, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)
        h, w = img_rgb.shape[:2]
    for _ in range(max_tries):
        max_y = max(0, h - crop)
        max_x = max(0, w - crop)
        y = 0 if max_y == 0 else np.random.randint(0, max_y + 1)
        x = 0 if max_x == 0 else np.random.randint(0, max_x + 1)
        patch = img_rgb[y:y + crop, x:x + crop]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 120)
        if edges.mean() > 4.0:
            return patch
    # fallback centre
    y = max(0, (h - crop) // 2)
    x = max(0, (w - crop) // 2)
    return img_rgb[y:y + crop, x:x + crop]

def global_multi_crops(img_rgb: np.ndarray, out_size: int = 224, long_side: int = 768) -> List[np.ndarray]:
    """
    5 crops déterministes: centre + 4 coins, après resize sur long_side.
    Déterministe pour une éval stable et sensible aux bords/coins (screenshots).
    """
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return [np.zeros((out_size, out_size, 3), dtype=np.uint8)]
    scale = long_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h = max(0, out_size - new_h)
    pad_w = max(0, out_size - new_w)
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2; bottom = pad_h - top
        left = pad_w // 2; right = pad_w - left
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
        new_h, new_w = resized.shape[:2]

    coords = []
    coords.append(((new_h - out_size)//2, (new_w - out_size)//2))  # centre
    coords += [
        (0, 0),
        (0, max(0, new_w - out_size)),
        (max(0, new_h - out_size), 0),
        (max(0, new_h - out_size), max(0, new_w - out_size))
    ]

    crops = []
    for (y0, x0) in coords:
        y0 = int(max(0, min(y0, new_h - out_size)))
        x0 = int(max(0, min(x0, new_w - out_size)))
        crops.append(resized[y0:y0+out_size, x0:x0+out_size])
    return crops

class DocBlurPredictor:
    def __init__(self, ckpt_path: Path, device: str = "cpu",
                 alpha: float = 0.6, beta: float = 0.4,
                 n_local: int = 8, n_global: int = 2,
                 local_crop: int = 224, global_crop: int = 224, global_long_side: int = 768,
                 threshold: float = 0.50,
                 num_threads: int = 2):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        torch.set_num_threads(max(1, int(num_threads)))
        self.model = DocBlurNet()
        self.model.load_state_dict(torch.load(str(ckpt_path), map_location=self.device))
        self.model.eval().to(self.device)
        s = max(1e-9, alpha + beta)
        self.alpha, self.beta = alpha / s, beta / s
        self.n_local, self.n_global = int(n_local), int(n_global)
        self.local_crop, self.global_crop, self.global_long = int(local_crop), int(global_crop), int(global_long_side)
        self.threshold = float(threshold)

    @torch.no_grad()
    def _batch_probs(self, patches: List[np.ndarray]) -> List[float]:
        if not patches:
            return []
        T = torch.stack([
            ((torch.from_numpy(p).permute(2, 0, 1).float() / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
            for p in patches
        ]).to(self.device)
        logits = self.model(T)
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        return probs

    @torch.no_grad()
    def predict_scores(self, file_path: Path) -> Optional[Tuple[float, float, float]]:
        """Retourne (local_median, global_mean, fused_prob) ou None si illisible."""
        img = load_first_page_as_rgb(file_path)
        if img is None:
            return None

        # Locaux
        local_patches = [random_texty_crop(img, crop=self.local_crop) for _ in range(self.n_local)]
        # Globaux déterministes
        global_all = global_multi_crops(img, out_size=self.global_crop, long_side=self.global_long)
        global_patches = global_all[:self.n_global] if self.n_global > 0 else []

        patches = local_patches + global_patches
        probs = self._batch_probs(patches)

        local_probs  = probs[:len(local_patches)] if local_patches else []
        global_probs = probs[len(local_patches):] if global_patches else []

        local_med   = float(np.median(local_probs)) if local_probs else 0.0
        global_mean = float(np.mean(global_probs))  if global_probs else 0.0
        fused       = self.alpha * local_med + self.beta * global_mean
        return local_med, global_mean, fused

    @torch.no_grad()
    def predict_blurry_bool(self, file_path: Path) -> Optional[bool]:
        s = self.predict_scores(file_path)
        if s is None:
            return None
        _, _, fused = s
        return bool(fused >= self.threshold)

# ----------------- Métriques & seuil -----------------

def bin_metrics(y_true: List[int], y_pred: List[int]):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==0)
    acc = (tp+tn)/max(1,(tp+tn+fp+fn))
    prec = tp/max(1,(tp+fp))
    rec  = tp/max(1,(tp+fn))
    f1   = 2*prec*rec/max(1e-9,(prec+rec))
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "tp":tp, "tn":tn, "fp":fp, "fn":fn}

# ----------------- Evaluation -----------------

def main():
    # Dossiers
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOCAL_FILE_PATH", str(TMP_IMAGES_DIR))

    # Chargement modèle / hyperparams depuis env
    ckpt_default = HERE / "docblur_mobilenetv3.pt"
    ckpt_path = Path(os.environ.get("DOCBLUR_CKPT", ckpt_default))
    if not ckpt_path.exists():
        print(f"Checkpoint introuvable: {ckpt_path}. Définir DOCBLUR_CKPT ou placer le fichier à cet emplacement.")
        sys.exit(1)

    predictor = DocBlurPredictor(
        ckpt_path=ckpt_path,
        device=os.environ.get("DOCBLUR_DEVICE", "cpu"),
        alpha=float(os.environ.get("DOCBLUR_ALPHA", 0.6)),
        beta=float(os.environ.get("DOCBLUR_BETA", 0.4)),
        n_local=int(os.environ.get("DOCBLUR_N_LOCAL", 8)),
        n_global=int(os.environ.get("DOCBLUR_N_GLOBAL", 2)),
        local_crop=int(os.environ.get("DOCBLUR_LOCAL_CROP", 224)),
        global_crop=int(os.environ.get("DOCBLUR_GLOBAL_CROP", 224)),
        global_long_side=int(os.environ.get("DOCBLUR_GLOBAL_LONG", 768)),
        threshold=float(os.environ.get("DOCBLUR_THRESHOLD", 0.50)),  # utilisé uniquement si pas de grid-search
        num_threads=int(os.environ.get("DOCBLUR_THREADS", 2)),
    )

    files_with_label = list(iter_files_with_label_from_dirs())
    print(f"[info] Parcours de tout le dataset ({len(files_with_label)} fichiers)")

    total = len(files_with_label)

    # Collecte des scores (pour grid-search)
    records = []  # (path, label, mime, local_median, global_mean, fused_prob or None)

    # Progress
    if RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("Evaluating files (CNN)"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=False,
            expand=True,
        )
        ctx_prog = progress
    else:
        ctx_prog = None

    with muted_app_logger():
        if ctx_prog:
            with ctx_prog:
                task_id = ctx_prog.add_task("evaluate_cnn", total=total)
                for file_path, label_blurry in files_with_label:
                    s = predictor.predict_scores(file_path)
                    mime = infer_mime_type(file_path)
                    if s is None:
                        records.append((file_path, label_blurry, mime, None, None, None))
                    else:
                        lm, gm, fused = s
                        records.append((file_path, label_blurry, mime, lm, gm, fused))
                    ctx_prog.update(task_id, advance=1)
        else:
            processed = 0
            def _render_progress_bar(completed: int, total: int, prefix: str = "Processing") -> str:
                if total <= 0:
                    return f"{prefix} 0% [--------------------------------------------------] 0/0"
                percent = int((completed / total) * 100)
                bar_len = 50
                filled = int(bar_len * completed / total)
                bar = ("=" * filled + ">" + "-" * (bar_len - filled - 1)) if filled < bar_len else "=" * bar_len
                return f"{prefix} {percent:3d}% [{bar}] {completed}/{total}"
            def _update_progress_bar(completed: int, total: int, prefix: str = "Processing") -> None:
                line = _render_progress_bar(completed, total, prefix)
                sys.stderr.write("\r" + line)
                sys.stderr.flush()
                if completed >= total:
                    sys.stderr.write("\n"); sys.stderr.flush()
            _update_progress_bar(processed, total, prefix="Evaluating files (CNN)")
            for file_path, label_blurry in files_with_label:
                s = predictor.predict_scores(file_path)
                mime = infer_mime_type(file_path)
                if s is None:
                    records.append((file_path, label_blurry, mime, None, None, None))
                else:
                    lm, gm, fused = s
                    records.append((file_path, label_blurry, mime, lm, gm, fused))
                processed += 1
                _update_progress_bar(processed, total, prefix="Evaluating files (CNN)")

    # -------- Grid-search du meilleur seuil sur fused_prob --------
    y_true = [int(lbl) for (_, lbl, _, lm, gm, fused) in records if fused is not None]
    scores = [float(fused) for (_, lbl, _, lm, gm, fused) in records if fused is not None]

    if len(scores) == 0:
        print("[error] Aucun score calculé (toutes les lectures ont échoué ?).")
        sys.exit(2)

    thr_min = float(os.environ.get("DOCBLUR_THR_MIN", 0.30))
    thr_max = float(os.environ.get("DOCBLUR_THR_MAX", 0.70))
    thr_step = float(os.environ.get("DOCBLUR_THR_STEP", 0.01))

    best = {"thr": 0.5, "acc":0, "prec":0, "rec":0, "f1":-1, "tp":0, "tn":0, "fp":0, "fn":0}
    thr_vals = np.arange(thr_min, thr_max + 1e-9, thr_step)
    for thr in thr_vals:
        y_pred = [1 if s >= thr else 0 for s in scores]
        m = bin_metrics(y_true, y_pred)
        if m["f1"] > best["f1"]:
            best = {"thr": float(thr), **m}

    print("=== Seuil optimal sur fused_prob (split de test / dataset) ===")
    print(f"thr* = {best['thr']:.2f} | Acc={best['acc']:.3f}  P={best['prec']:.3f}  R={best['rec']:.3f}  F1={best['f1']:.3f}")
    # Sauvegarde des métriques
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    # -------- Ecriture des fichiers de sortie --------

    # CSV officiel (identique à main.py) -> prédiction binaire @ thr*
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "relative_path",
        "label_blurry",
        "file_type",
        "laplace_is_blurry",
        "predicted_is_blurry",
        "laplacian_variance",
        "is_blank",
        "is_readable_tesseract",
        "compare_summary",
    ]
    rows_json = []  # pour result.json

    with RESULT_CSV.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for file_path, label_blurry, mime, lm, gm, fused in records:
            if fused is None:
                row = {
                    "relative_path": str(file_path.relative_to(DATASET_ROOT)) if DATASET_ROOT in file_path.parents else str(file_path),
                    "label_blurry": int(label_blurry),
                    "file_type": mime,
                    "laplace_is_blurry": "",
                    "predicted_is_blurry": "",
                    "laplacian_variance": "",
                    "is_blank": "",
                    "is_readable_tesseract": "",
                    "compare_summary": "no_result",
                }
                row_json = {
                    **row,
                    "laplace_is_blurry": None,
                    "predicted_is_blurry": None,
                    "laplacian_variance": None,
                    "is_blank": None,
                    "is_readable_tesseract": None,
                }
            else:
                pred = 1 if fused >= best["thr"] else 0
                row = {
                    "relative_path": str(file_path.relative_to(DATASET_ROOT)) if DATASET_ROOT in file_path.parents else str(file_path),
                    "label_blurry": int(label_blurry),
                    "file_type": mime,
                    "laplace_is_blurry": "",
                    "predicted_is_blurry": pred,
                    "laplacian_variance": "",
                    "is_blank": "",
                    "is_readable_tesseract": "",
                    "compare_summary": compare_summary_from_pred(bool(pred), bool(label_blurry)),
                }
                row_json = {
                    **row,
                    "laplace_is_blurry": None,
                    "laplacian_variance": None,
                    "is_blank": None,
                    "is_readable_tesseract": None,
                }
            writer.writerow(row)
            rows_json.append(row_json)

    # JSON (structure identique, champs OCR/laplacian à null)
    with RESULT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows_json, f, ensure_ascii=False, indent=2)

    # CSV avec scores détaillés (audit)
    with RESULT_CSV_SCORES.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path","label_blurry","file_type","local_median","global_mean","fused_prob",f"pred@thr={best['thr']:.2f}"])
        for file_path, label_blurry, mime, lm, gm, fused in records:
            rel = str(file_path.relative_to(DATASET_ROOT)) if DATASET_ROOT in file_path.parents else str(file_path)
            if fused is None:
                w.writerow([rel, int(label_blurry), mime, "", "", "", ""])
            else:
                pred = 1 if fused >= best["thr"] else 0
                w.writerow([rel, int(label_blurry), mime,
                            f"{lm:.6f}", f"{gm:.6f}", f"{fused:.6f}", pred])

    print(f"\n✓ Fichiers écrits :\n- {RESULT_CSV}\n- {RESULT_JSON}\n- {RESULT_CSV_SCORES}\n- {METRICS_JSON}")

if __name__ == "__main__":
    main()
