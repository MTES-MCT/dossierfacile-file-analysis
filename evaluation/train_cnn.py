#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement d'un petit CNN (MobileNetV3-Small) pour détecter si une page
doit être rejetée (classe positive = documents flous OU documents non-scannés
ex: photo/screenshot de document), et acceptée sinon (scans nets).

Pipeline :
1) Scan récursif des fichiers dans blurry/ et not_blurry/ (PNG/JPG/PDF)
2) Split déterministe par classe :
   - jusqu'à --train_per_class (par classe) pour le train,
   - le reste pour l’évaluation (on vise >= --eval_min_per_class)
   -> Split sauvegardé dans out_dir/split_docblur.json
3) Entraînement :
   - Dataset multi-échelle :
       * 50% (par défaut) de PATCHS LOCAUX "texty" 224x224 (apprend la netteté)
       * 50% de CROPS GLOBAUX (vue page réduite) 224x224 (apprend "photo/screenshot")
     Légères augmentations photométriques SANS floutage
   - MobileNetV3-Small pré-entraîné (ImageNet) + BCEWithLogitsLoss
4) Validation à la fin de chaque époque :
   - Pour chaque image : N_local patches + N_global crops
   - Agrégation : proba_fusion = alpha*median(local) + beta*mean(global)
   - Décision @ 0.5 sur la proba_fusion
   - Sauvegarde du meilleur checkpoint selon F1
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import math
import os
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm

# --- Intégration avec ta pipeline d’analyse existante ---
from dossierfacile_file_analysis.executor.tasks.prepare_data_for_analysis import PrepareDataForAnalysis
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile
from dossierfacile_file_analysis.models.supported_content_type import SupportedContentType
from dossierfacile_file_analysis.custom_logging.logging_config import logger as app_logger

# Répertoire temporaire pour les images générées par PrepareDataForAnalysis
TMP_TRAIN_IMAGES_DIR = Path(__file__).parent / "tmp_train_images"
TMP_TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LOCAL_FILE_PATH", str(TMP_TRAIN_IMAGES_DIR))

@contextmanager
def muted_app_logger():
    prev_disabled = getattr(app_logger, "disabled", False)
    app_logger.disabled = True
    try:
        yield
    finally:
        app_logger.disabled = prev_disabled

def infer_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return SupportedContentType.JPEG.value
    if ext == ".png":
        return SupportedContentType.PNG.value
    if ext == ".pdf":
        return SupportedContentType.PDF.value
    return SupportedContentType.PNG.value

# ---------- Config de base ----------
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".pdf"}
RNG_SEED = 123

# ---------- Utils I/O ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return sorted(files)

def load_first_page_as_rgb(path: Path) -> Optional[np.ndarray]:
    """
    Charge une image RGB (H,W,3, uint8) via la pipeline PrepareDataForAnalysis
    (PDF -> images temporaires ; images restent telles quelles).
    Nettoie les images temporaires générées, sans toucher au fichier source.
    """
    ctx = None
    try:
        prepare_task = PrepareDataForAnalysis()
        downloaded = DownloadedFile(
            file_name=path.stem,
            file_path=str(path),
            file_type=infer_mime_type(path),
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

        # Si PDF: première image générée ; sinon image initiale
        img_path = data.list_of_images[0] if data.list_of_images else data.initial_file
        if not img_path or not os.path.exists(img_path):
            return None

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    except Exception as e:
        print(f"[warn] Échec de lecture via PrepareDataForAnalysis: {path} ({e})")
        return None

    finally:
        # Nettoyage des images temporaires
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

# ---------- Crops locaux & globaux ----------

def random_texty_crop(img_rgb: np.ndarray, crop: int = 224, max_tries: int = 20) -> np.ndarray:
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
        img_rgb = cv2.copyMakeBorder(img_rgb, pad_top, pad_bottom, pad_left, pad_right,
                                     borderType=cv2.BORDER_REPLICATE)
        h, w = img_rgb.shape[:2]

    for _ in range(max_tries):
        max_y = max(0, h - crop)
        max_x = max(0, w - crop)
        y = 0 if max_y == 0 else random.randint(0, max_y)
        x = 0 if max_x == 0 else random.randint(0, max_x)
        patch = img_rgb[y:y + crop, x:x + crop]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 120)
        if edges.mean() > 4.0:
            return patch
    # fallback : centre
    y = max(0, (h - crop) // 2)
    x = max(0, (w - crop) // 2)
    return img_rgb[y:y + crop, x:x + crop]

def global_resized_crop(img_rgb: np.ndarray, out_size: int = 224, long_side: int = 768) -> np.ndarray:
    """
    Vue globale : réduit la page (long côté -> long_side), puis prélève un crop
    aléatoire couvrant 60–100% de la zone, redimensionné en out_size x out_size.
    Permet de capter bords d'écran, UI, arrière-plans, perspective, etc.
    """
    h, w = img_rgb.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    H, W = img_rgb.shape[:2]
    min_scale = 0.60
    crop_h = int(np.random.uniform(min_scale, 1.0) * H)
    crop_w = int(np.random.uniform(min_scale, 1.0) * W)
    y = 0 if H == crop_h else np.random.randint(0, H - crop_h + 1)
    x = 0 if W == crop_w else np.random.randint(0, W - crop_w + 1)
    crop = img_rgb[y:y + crop_h, x:x + crop_w]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop

# ---------- Dataset ----------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class DocBlurTrainDataset(Dataset):
    """
    Dataset TRAIN multi-échelle :
      - avec probabilité global_prob -> crop GLOBAL
      - sinon -> patch LOCAL "texty"
    Augmentations légères, pas de flou artificiel.
    """
    def __init__(self, files: List[Path], labels: List[int],
                 global_prob: float = 0.5,
                 local_crop: int = 224,
                 global_crop: int = 224,
                 global_long_side: int = 768):
        assert len(files) == len(labels)
        self.files = files
        self.labels = labels
        self.global_prob = float(global_prob)
        self.local_crop = int(local_crop)
        self.global_crop = int(global_crop)
        self.global_long = int(global_long_side)

    def __len__(self):
        return len(self.files)

    def _augment_keep_sharp(self, patch: np.ndarray) -> np.ndarray:
        # Légères variations de luminosité/contraste + compression JPEG
        if random.random() < 0.3:
            alpha = 1.0 + random.uniform(-0.08, 0.08)  # contraste
            beta = random.uniform(-10, 10)            # luminosité
            patch = cv2.convertScaleAbs(patch, alpha=alpha, beta=beta)
        if random.random() < 0.2:
            q = random.randint(60, 95)
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
                                  [int(cv2.IMWRITE_JPEG_QUALITY), q])
            patch = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return patch

    def __getitem__(self, idx: int):
        path = self.files[idx]
        label = self.labels[idx]
        img = load_first_page_as_rgb(path)
        if img is None:
            # fallback déterministe
            alt_idx = (idx + 1) % len(self.files)
            return self.__getitem__(alt_idx)

        use_global = (random.random() < self.global_prob)
        if use_global:
            patch = global_resized_crop(img, out_size=self.global_crop, long_side=self.global_long)
        else:
            patch = random_texty_crop(img, crop=self.local_crop)

        patch = self._augment_keep_sharp(patch)

        t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        y = torch.tensor(label, dtype=torch.float32)
        return t, y

class DocBlurValList:
    """Liste des chemins/labels pour l'évaluation image-par-image."""
    def __init__(self, files: List[Path], labels: List[int]):
        self.files = files
        self.labels = labels
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return self.files[idx], self.labels[idx]

# ---------- Modèle ----------

class DocBlurNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        in_feats = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_feats, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # binaire: rejeter (1) / accepter (0)
        )
    def forward(self, x):
        return self.backbone(x).squeeze(1)  # logits

# ---------- Validation helpers (fusion local+global) ----------

@torch.no_grad()
def predict_image_reject_prob(model: nn.Module, path: Path, device: str = "cpu",
                              n_local: int = 8, n_global: int = 2,
                              alpha: float = 0.6, beta: float = 0.4,
                              local_crop: int = 224, global_crop: int = 224,
                              global_long_side: int = 768) -> Optional[float]:
    """
    Retourne la proba FUSIONNÉE d'être 'à rejeter' (classe positive) :
      fused = alpha * median(local_probs) + beta * mean(global_probs)
    None si l'image est illisible.
    """
    s = alpha + beta
    if s <= 0:  # normalisation de sécurité
        alpha, beta = 0.5, 0.5
    else:
        alpha, beta = alpha / s, beta / s

    img = load_first_page_as_rgb(path)
    if img is None:
        return None

    local_probs = []
    for _ in range(max(0, n_local)):
        p = random_texty_crop(img, crop=local_crop)
        t = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        prob = torch.sigmoid(model(t.unsqueeze(0).to(device))).item()
        local_probs.append(prob)

    global_probs = []
    for _ in range(max(0, n_global)):
        g = global_resized_crop(img, out_size=global_crop, long_side=global_long_side)
        t = torch.from_numpy(g).permute(2, 0, 1).float() / 255.0
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        prob = torch.sigmoid(model(t.unsqueeze(0).to(device))).item()
        global_probs.append(prob)

    if local_probs:
        local_med = float(np.median(local_probs))
    else:
        local_med = 0.0
    if global_probs:
        global_mean = float(np.mean(global_probs))
    else:
        global_mean = 0.0

    fused = alpha * local_med + beta * global_mean
    return float(fused)

def bin_metrics(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float, float]:
    assert len(y_true) == len(y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-9, (prec + rec))
    return acc, prec, rec, f1

# ---------- Split ----------

def make_split(
    blurry_dir: Path,
    not_blurry_dir: Path,
    train_per_class: int = 200,
    eval_min_per_class: int = 300
):
    random.seed(RNG_SEED)
    blurry = list_files(blurry_dir)
    notb = list_files(not_blurry_dir)

    def can_load(p: Path) -> bool:
        # On garde PDF + images : la pipeline interne s’occupe du rendu image.
        return p.suffix.lower() in ALLOWED_EXTS

    blurry = [p for p in blurry if can_load(p)]
    notb   = [p for p in notb   if can_load(p)]
    random.shuffle(blurry)
    random.shuffle(notb)

    def split_class(files: List[Path]) -> Tuple[List[Path], List[Path]]:
        if len(files) <= (train_per_class + eval_min_per_class):
            cut = int(0.8 * len(files))
            return files[:cut], files[cut:]
        else:
            train = files[:train_per_class]
            eval_ = files[train_per_class:]
            return train, eval_

    train_b, eval_b = split_class(blurry)
    train_n, eval_n = split_class(notb)

    X_train = train_b + train_n
    y_train = [1] * len(train_b) + [0] * len(train_n)

    X_eval  = eval_b + eval_n
    y_eval  = [1] * len(eval_b) + [0] * len(eval_n)

    tr = list(zip(X_train, y_train))
    ev = list(zip(X_eval,  y_eval))
    random.shuffle(tr)
    random.shuffle(ev)
    X_train, y_train = zip(*tr) if tr else ([], [])
    X_eval,  y_eval  = zip(*ev) if ev else ([], [])

    return list(X_train), list(y_train), list(X_eval), list(y_eval)

# ---------- Train loop ----------

def train(
    data_root: Path,
    out_dir: Path,
    train_per_class: int = 200,
    eval_min_per_class: int = 300,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 2e-3,
    device: str = "cuda",
    # params multi-échelle & fusion
    global_prob: float = 0.5,
    local_crop: int = 224,
    global_crop: int = 224,
    global_long_side: int = 768,
    n_local_val: int = 8,
    n_global_val: int = 2,
    alpha: float = 0.6,
    beta: float = 0.4,
    selection_threshold: float = 0.5,
):
    ensure_dir(out_dir)
    ckpt_path = out_dir / "docblur_mobilenetv3.pt"
    split_path = out_dir / "split_docblur.json"

    blurry_dir = data_root / "blurry"
    not_blurry_dir = data_root / "not_blurry"
    assert blurry_dir.is_dir(), f"Manque le dossier: {blurry_dir}"
    assert not_blurry_dir.is_dir(), f"Manque le dossier: {not_blurry_dir}"

    X_train, y_train, X_eval, y_eval = make_split(
        blurry_dir, not_blurry_dir, train_per_class, eval_min_per_class
    )

    split_obj = {
        "train": {"files": [str(p) for p in X_train], "labels": y_train},
        "eval":  {"files": [str(p) for p in X_eval],  "labels": y_eval},
        "params": {
            "train_per_class": train_per_class,
            "eval_min_per_class": eval_min_per_class,
            "seed": RNG_SEED,
            "global_prob": global_prob,
            "local_crop": local_crop,
            "global_crop": global_crop,
            "global_long_side": global_long_side,
            "alpha": alpha,
            "beta": beta,
            "n_local_val": n_local_val,
            "n_global_val": n_global_val,
            "selection_threshold": selection_threshold
        }
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, indent=2, ensure_ascii=False)
    print(f"[info] Split sauvegardé → {split_path}")
    print(f"[info] Train: {len(X_train)}  | Eval: {len(X_eval)}")

    # Datasets / Loaders
    train_ds = DocBlurTrainDataset(
        [Path(p) for p in split_obj["train"]["files"]],
        split_obj["train"]["labels"],
        global_prob=global_prob,
        local_crop=local_crop,
        global_crop=global_crop,
        global_long_side=global_long_side
    )
    val_list = DocBlurValList([Path(p) for p in split_obj["eval"]["files"]],
                              split_obj["eval"]["labels"])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)

    # Modèle
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = DocBlurNet(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_dl, desc=f"[Train] Epoch {epoch}/{epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)

        train_loss = running / max(1, len(train_ds))

        # Validation (F1 @ selection_threshold) sur la proba fusionnée
        model.eval()
        y_true, y_pred = [], []
        for path, label in tqdm(zip(val_list.files, val_list.labels), total=len(val_list), desc="[Val]"):
            fused = predict_image_reject_prob(
                model, path, device=device,
                n_local=n_local_val, n_global=n_global_val,
                alpha=alpha, beta=beta,
                local_crop=local_crop, global_crop=global_crop,
                global_long_side=global_long_side
            )
            if fused is None:
                continue
            pred = 1 if fused >= selection_threshold else 0
            y_true.append(label)
            y_pred.append(pred)

        acc, prec, rec, f1 = bin_metrics(y_true, y_pred)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} | VAL  acc={acc:.3f} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Meilleur checkpoint → {ckpt_path}  (F1={f1:.3f})")

    print("[done] Entraînement terminé.")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="evaluation/dataset", type=str,
                    help="Racine du dataset (contient blurry/ et not_blurry/)")
    ap.add_argument("--out_dir", default="evaluation/ouput", type=str,
                    help="Dossier de sortie (ckpt + split JSON)")
    ap.add_argument("--train_per_class", default=300, type=int)
    ap.add_argument("--eval_min_per_class", default=300, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--lr", default=2e-3, type=float)
    ap.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])

    # multi-échelle & fusion
    ap.add_argument("--global_prob", default=0.5, type=float)
    ap.add_argument("--local_crop", default=224, type=int)
    ap.add_argument("--global_crop", default=224, type=int)
    ap.add_argument("--global_long_side", default=768, type=int)
    ap.add_argument("--n_local_val", default=8, type=int)
    ap.add_argument("--n_global_val", default=2, type=int)
    ap.add_argument("--alpha", default=0.6, type=float)
    ap.add_argument("--beta", default=0.4, type=float)
    ap.add_argument("--selection_threshold", default=0.5, type=float,
                    help="Seuil de décision utilisé pour sauvegarder le meilleur checkpoint")

    args = ap.parse_args()

    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    train(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        train_per_class=args.train_per_class,
        eval_min_per_class=args.eval_min_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        global_prob=args.global_prob,
        local_crop=args.local_crop,
        global_crop=args.global_crop,
        global_long_side=args.global_long_side,
        n_local_val=args.n_local_val,
        n_global_val=args.n_global_val,
        alpha=args.alpha,
        beta=args.beta,
        selection_threshold=args.selection_threshold
    )

if __name__ == "__main__":
    main()
