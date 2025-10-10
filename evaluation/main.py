import csv
import os
import sys
from pathlib import Path
from typing import Iterator, Tuple

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


DATASET_ROOT = Path(__file__).parent / "dataset"
BLURRY_DIR = DATASET_ROOT / "blurry"
NOT_BLURRY_DIR = DATASET_ROOT / "not_blurry"
RESULT_DIR = DATASET_ROOT / "result"
TMP_IMAGES_DIR = RESULT_DIR / "tmp_images"
RESULT_CSV = RESULT_DIR / "results.csv"


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


def decide_final_blurry(laplace_is_blurry: bool, is_readable: bool) -> bool:
    """Decision rule (updated):
    - If Laplace says blurry (True) => final is blurry.
    - Else, if the document is not readable (is_readable=False) => final is blurry.
    - Else => final is not blurry.
    Equivalent: laplace_is_blurry or (not is_readable)
    """
    return bool(laplace_is_blurry) or (not bool(is_readable))


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


def compare_algorithms_placeholder(is_blurry: bool, is_readable: bool, expected_blurry: bool) -> str:
    """Apply the decision rule and summarize the comparison to the dataset label.
    Returns a compact verdict: tp/tn/fp/fn with final and expected.
    """
    final_pred = decide_final_blurry(is_blurry, is_readable)
    if final_pred and expected_blurry:
        verdict = "tp"
    elif (not final_pred) and (not expected_blurry):
        verdict = "tn"
    elif final_pred and (not expected_blurry):
        verdict = "fp"
    else:
        verdict = "fn"
    return f"final={int(final_pred)} expected={int(expected_blurry)} verdict={verdict}"


def main():
    # Ensure output dirs exist
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Direct PrepareDataForAnalysis to write temporary images where we can clean them
    os.environ.setdefault("LOCAL_FILE_PATH", str(TMP_IMAGES_DIR))

    prepare_task = PrepareDataForAnalysis()
    analyse_task = AnalyseFiles()

    # Pré-calcul des fichiers à traiter pour connaître le total
    files_with_label = list(iter_files_with_label())
    total = len(files_with_label)

    # CSV header
    fieldnames = [
        "relative_path",
        "label_blurry",
        "file_type",
        "laplace_is_blurry",
        "predicted_is_blurry",
        "laplacian_variance",
        "is_blank",
        "is_readable_tesseract",
        "compare_summary"
    ]

    with RESULT_CSV.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0

        # Désactivation des logs applicatifs pendant le traitement pour une progression propre
        with muted_app_logger():
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

                    for file_path, label_blurry in files_with_label:
                        ctx = None
                        try:
                            mime = infer_mime_type(file_path)
                            downloaded = DownloadedFile(
                                file_name=file_path.stem,
                                file_path=str(file_path),
                                file_type=mime
                            )

                            # Build a minimal context; file_id is irrelevant for offline eval
                            ctx = BlurryExecutionContext(queue_message=BlurryQueueMessage(file_id=0))
                            ctx.downloaded_file = downloaded

                            # Prepare (convert PDF -> images)
                            if prepare_task.has_to_apply(ctx):
                                prepare_task.run(ctx)
                            else:
                                # Should not happen, but keep safe
                                ctx.input_analysis_data = None

                            # Analyse
                            if analyse_task.has_to_apply(ctx):
                                analyse_task.run(ctx)
                            else:
                                # No input data, skip
                                continue

                            result = ctx.blurry_result
                            if result is None:
                                # Nothing computed; skip with a line indicating failure
                                writer.writerow({
                                    "relative_path": str(file_path.relative_to(DATASET_ROOT)),
                                    "label_blurry": int(label_blurry),
                                    "file_type": mime,
                                    "laplace_is_blurry": "",
                                    "predicted_is_blurry": "",
                                    "laplacian_variance": "",
                                    "is_blank": "",
                                    "is_readable_tesseract": "",
                                    "compare_summary": "no_result"
                                })
                                continue

                            # Compute final decision with the updated rule
                            final_pred_blurry = decide_final_blurry(result.is_blurry, result.is_readable)
                            compare_summary = compare_algorithms_placeholder(
                                is_blurry=result.is_blurry,
                                is_readable=result.is_readable,
                                expected_blurry=label_blurry,
                            )

                            writer.writerow({
                                "relative_path": str(file_path.relative_to(DATASET_ROOT)),
                                "label_blurry": int(label_blurry),
                                "file_type": mime,
                                "laplace_is_blurry": int(bool(result.is_blurry)),
                                "predicted_is_blurry": int(bool(final_pred_blurry)),
                                "laplacian_variance": result.laplacian_variance,
                                "is_blank": int(bool(result.is_blank)),
                                "is_readable_tesseract": int(bool(result.is_readable)),
                                "compare_summary": compare_summary,
                            })
                        except Exception as e:
                            # Persist the error line but keep going
                            writer.writerow({
                                "relative_path": str(file_path.relative_to(DATASET_ROOT)) if DATASET_ROOT in file_path.parents else str(file_path),
                                "label_blurry": int(label_blurry),
                                "file_type": infer_mime_type(file_path),
                                "laplace_is_blurry": "",
                                "predicted_is_blurry": "",
                                "laplacian_variance": "",
                                "is_blank": "",
                                "is_readable_tesseract": "",
                                "compare_summary": f"error: {e}",
                            })
                        finally:
                            # Clean up only generated images; do not delete original dataset file
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
                            processed += 1
                            progress.update(task_id, advance=1)
            else:
                # Fallback simple sur stderr
                _update_progress_bar(processed, total, prefix="Evaluating files")

                for file_path, label_blurry in files_with_label:
                    ctx = None
                    try:
                        mime = infer_mime_type(file_path)
                        downloaded = DownloadedFile(
                            file_name=file_path.stem,
                            file_path=str(file_path),
                            file_type=mime
                        )

                        # Build a minimal context; file_id is irrelevant for offline eval
                        ctx = BlurryExecutionContext(queue_message=BlurryQueueMessage(file_id=0))
                        ctx.downloaded_file = downloaded

                        # Prepare (convert PDF -> images)
                        if prepare_task.has_to_apply(ctx):
                            prepare_task.run(ctx)
                        else:
                            # Should not happen, but keep safe
                            ctx.input_analysis_data = None

                        # Analyse
                        if analyse_task.has_to_apply(ctx):
                            analyse_task.run(ctx)
                        else:
                            # No input data, skip
                            continue

                        result = ctx.blurry_result
                        if result is None:
                            # Nothing computed; skip with a line indicating failure
                            writer.writerow({
                                "relative_path": str(file_path.relative_to(DATASET_ROOT)),
                                "label_blurry": int(label_blurry),
                                "file_type": mime,
                                "laplace_is_blurry": "",
                                "predicted_is_blurry": "",
                                "laplacian_variance": "",
                                "is_blank": "",
                                "is_readable_tesseract": "",
                                "compare_summary": "no_result"
                            })
                            continue

                        # Compute final decision with the updated rule
                        final_pred_blurry = decide_final_blurry(result.is_blurry, result.is_readable)
                        compare_summary = compare_algorithms_placeholder(
                            is_blurry=result.is_blurry,
                            is_readable=result.is_readable,
                            expected_blurry=label_blurry,
                        )

                        writer.writerow({
                            "relative_path": str(file_path.relative_to(DATASET_ROOT)),
                            "label_blurry": int(label_blurry),
                            "file_type": mime,
                            "laplace_is_blurry": int(bool(result.is_blurry)),
                            "predicted_is_blurry": int(bool(final_pred_blurry)),
                            "laplacian_variance": result.laplacian_variance,
                            "is_blank": int(bool(result.is_blank)),
                            "is_readable_tesseract": int(bool(result.is_readable)),
                            "compare_summary": compare_summary,
                        })
                    except Exception as e:
                        # Persist the error line but keep going
                        writer.writerow({
                            "relative_path": str(file_path.relative_to(DATASET_ROOT)) if DATASET_ROOT in file_path.parents else str(file_path),
                            "label_blurry": int(label_blurry),
                            "file_type": infer_mime_type(file_path),
                            "laplace_is_blurry": "",
                            "predicted_is_blurry": "",
                            "laplacian_variance": "",
                            "is_blank": "",
                            "is_readable_tesseract": "",
                            "compare_summary": f"error: {e}",
                        })
                    finally:
                        # Clean up only generated images; do not delete original dataset file
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
                        processed += 1
                        _update_progress_bar(processed, total, prefix="Evaluating files")

    print(f"Evaluation completed. Results written to: {RESULT_CSV}")


if __name__ == "__main__":
    main()
