import csv
import os
import sys
from pathlib import Path
from typing import Iterator, Tuple
import shutil

from dossierfacile_file_analysis.executor.tasks.prepare_data_for_analysis import PrepareDataForAnalysis
from dossierfacile_file_analysis.executor.tasks.analyse_files import AnalyseFiles
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile
from dossierfacile_file_analysis.models.supported_content_type import SupportedContentType


# Support progression (optionnel)
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
from concurrent.futures import ThreadPoolExecutor, as_completed


DATASET_ROOT = Path(__file__).parent / "dataset"
BLURRY_DIR = DATASET_ROOT / "blurry"
NOT_BLURRY_DIR = DATASET_ROOT / "not_blurry"
RESULT_DIR = DATASET_ROOT / "result"
TMP_IMAGES_DIR = RESULT_DIR / "tmp_images"
RESULT_CSV = RESULT_DIR / "results.csv"
# Dossiers de tri des erreurs de classification
TP_NP_DIR = RESULT_DIR / "TP_NP"
FP_DIR = TP_NP_DIR / "FP"
FN_DIR = TP_NP_DIR / "FN"


def infer_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return SupportedContentType.JPEG.value
    if ext == ".png":
        return SupportedContentType.PNG.value
    if ext == ".pdf":
        return SupportedContentType.PDF.value
    # fallback: try common image types
    return SupportedContentType.PNG.value if ext else SupportedContentType.PNG.value


def iter_files_with_label() -> Iterator[Tuple[Path, bool]]:
    for base, label in [(BLURRY_DIR, True), (NOT_BLURRY_DIR, False)]:
        if not base.exists():
            continue
        for root, _, files in os.walk(base):
            for f in files:
                p = Path(root) / f
                # skip hidden files
                if p.name.startswith('.'):
                    continue
                yield p, label


def decide_final_blurry(is_blurry) -> bool:
    """Decision rule (updated):
    - If Laplace says blurry (True) => final is blurry.
    - Else, if the document is not readable (is_readable=False) => final is blurry.
    - Else => final is not blurry.
    Equivalent: laplace_is_blurry or (not is_readable)
    """
    return is_blurry


# Barre de progression (fallback si Rich indisponible)
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
        sys.stderr.write("\n")
        sys.stderr.flush()


@contextmanager
def muted_app_logger():
    """Désactive temporairement le logger applicatif pour éviter d'interférer avec la barre de progression."""
    prev_disabled = getattr(app_logger, "disabled", False)
    app_logger.disabled = True
    try:
        yield
    finally:
        app_logger.disabled = prev_disabled


def compare_algorithms_placeholder(is_blurry: bool, expected_blurry: bool) -> str:
    """Apply the decision rule and summarize the comparison to the dataset label.
    Returns a compact verdict: tp/tn/fp/fn with final and expected.
    """
    final_pred = decide_final_blurry(is_blurry)
    if final_pred and expected_blurry:
        verdict = "tp"
    elif (not final_pred) and (not expected_blurry):
        verdict = "tn"
    elif final_pred and (not expected_blurry):
        verdict = "fp"
    else:
        verdict = "fn"
    return f"final={int(final_pred)} expected={int(expected_blurry)} verdict={verdict}"


def _safe_copy_to_bucket(src: Path, dest_base: Path, relative_under: Path) -> None:
    """Copie le fichier `src` sous dest_base/relative_path en créant les dossiers.
    `relative_under` doit être un parent de `src` pour calculer un chemin relatif stable.
    """
    try:
        rel = src.relative_to(relative_under)
    except Exception:
        # fallback: nom simple
        rel = src.name
    dest = dest_base / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dest)
    except Exception:
        # tente une copie simple si copy2 échoue
        try:
            shutil.copy(src, dest)
        except Exception:
            pass


def _process_one_file(file_path: Path, label_blurry: bool) -> dict:
    """Traite un fichier en isolant la logique pour exécution en thread.
    Retourne une ligne dict prête à être écrite dans le CSV. Assure le cleanup local.
    Copie les FP/FN dans result/TP_NP/FP et result/TP_NP/FN.
    """
    ctx = None
    try:
        mime = infer_mime_type(file_path)
        downloaded = DownloadedFile(
            file_name=file_path.stem,
            file_path=str(file_path),
            file_type=mime,
        )

        # Créer des instances locales des tasks pour éviter tout état partagé entre threads
        prepare_task = PrepareDataForAnalysis()
        analyse_task = AnalyseFiles()

        # Contexte minimal; file_id non pertinent en offline
        ctx = BlurryExecutionContext(queue_message=BlurryQueueMessage(file_id=0))
        ctx.downloaded_file = downloaded

        # Prépare (convert PDF -> images)
        if prepare_task.has_to_apply(ctx):
            prepare_task.run(ctx)
        else:
            ctx.input_analysis_data = None

        # Analyse
        if analyse_task.has_to_apply(ctx):
            analyse_task.run(ctx)
        else:
            # No input data, return no_result
            return {
                "relative_path": str(file_path.relative_to(DATASET_ROOT)),
                "label_blurry": int(label_blurry),
                "file_type": mime,
                "predicted_is_blurry": "",
                "is_blank": "",
                "is_readable_tesseract": "",
                "ocr_mean_score": "",
                "ocr_tokens_count": "",
                "compare_summary": "no_input",
            }

        result = ctx.blurry_result
        if result is None:
            return {
                "relative_path": str(file_path.relative_to(DATASET_ROOT)),
                "label_blurry": int(label_blurry),
                "file_type": mime,
                "predicted_is_blurry": "",
                "is_blank": "",
                "is_readable_tesseract": "",
                "ocr_mean_score": "",
                "ocr_tokens_count": "",
                "compare_summary": "no_result",
            }

        final_pred_blurry = decide_final_blurry(result.is_blurry)

        # verdict pour routing FP/FN
        if final_pred_blurry and (not label_blurry):
            # False Positive
            _safe_copy_to_bucket(file_path, FP_DIR, DATASET_ROOT)
            verdict = "fp"
        elif (not final_pred_blurry) and label_blurry:
            # False Negative
            _safe_copy_to_bucket(file_path, FN_DIR, DATASET_ROOT)
            verdict = "fn"
        elif final_pred_blurry and label_blurry:
            verdict = "tp"
        else:
            verdict = "tn"

        compare_summary = f"final={int(final_pred_blurry)} expected={int(label_blurry)} verdict={verdict}"

        return {
            "relative_path": str(file_path.relative_to(DATASET_ROOT)),
            "label_blurry": int(label_blurry),
            "file_type": mime,
            "predicted_is_blurry": int(bool(final_pred_blurry)),
            "is_blank": int(bool(result.is_blank)),
            "is_readable_tesseract": int(bool(result.is_blurry)),
            "ocr_mean_score": result.ocr_mean_score,
            "ocr_tokens_count": result.ocr_tokens,
            "compare_summary": compare_summary,
        }
    except Exception as e:
        return {
            "relative_path": str(file_path.relative_to(DATASET_ROOT)) if DATASET_ROOT in file_path.parents else str(file_path),
            "label_blurry": int(label_blurry),
            "file_type": infer_mime_type(file_path),
            "predicted_is_blurry": "",
            "is_blank": "",
            "is_readable_tesseract": "",
            "ocr_mean_score": "",
            "ocr_tokens_count": "",
            "compare_summary": f"error: {e}",
        }
    finally:
        # Nettoyage des images temporaires créées, sans toucher aux fichiers source du dataset
        try:
            if ctx is not None and hasattr(ctx, "input_analysis_data") and ctx.input_analysis_data is not None:
                for img in ctx.input_analysis_data.list_of_images:
                    try:
                        if img and os.path.exists(img):
                            os.remove(img)
                    except Exception:
                        pass
        except Exception:
            pass


def main():
    # Ensure output dirs exist
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    # Dossiers FP/FN pour inspection des erreurs
    FP_DIR.mkdir(parents=True, exist_ok=True)
    FN_DIR.mkdir(parents=True, exist_ok=True)

    # Direct PrepareDataForAnalysis to write temporary images where we can clean them
    os.environ.setdefault("LOCAL_FILE_PATH", str(TMP_IMAGES_DIR))

    # Pré-calcul des fichiers à traiter pour connaître le total
    files_with_label = list(iter_files_with_label())
    total = len(files_with_label)

    # CSV header
    fieldnames = [
        "relative_path",
        "label_blurry",
        "file_type",
        "predicted_is_blurry",
        "is_blank",
        "is_readable_tesseract",
        "ocr_mean_score",
        "ocr_tokens_count",
        "compare_summary"
    ]

    with RESULT_CSV.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0

        # Désactivation des logs applicatifs pendant le traitement pour une progression propre
        with muted_app_logger():
            # Exécute en parallèle sur 4 threads et maintient la progression dans le thread principal
            if RICH_AVAILABLE:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("Evaluating files"),
                    BarColumn(),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                    transient=False,
                    expand=True,
                )
                with progress:
                    task_id = progress.add_task("evaluate", total=total)
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [
                            executor.submit(_process_one_file, file_path, label_blurry)
                            for file_path, label_blurry in files_with_label
                        ]
                        for fut in as_completed(futures):
                            row = fut.result()
                            writer.writerow(row)
                            processed += 1
                            progress.update(task_id, advance=1)
            else:
                # Fallback simple sur stderr avec progression manuelle
                _update_progress_bar(processed, total, prefix="Evaluating files")
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(_process_one_file, file_path, label_blurry)
                        for file_path, label_blurry in files_with_label
                    ]
                    for fut in as_completed(futures):
                        row = fut.result()
                        writer.writerow(row)
                        processed += 1
                        _update_progress_bar(processed, total, prefix="Evaluating files")

    print(f"Evaluation completed. Results written to: {RESULT_CSV}")


if __name__ == "__main__":
    main()
