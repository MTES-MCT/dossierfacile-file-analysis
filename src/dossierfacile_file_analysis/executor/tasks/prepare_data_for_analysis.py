import math
import os

import pymupdf

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.exceptions.invalid_mime_type import InvalidMimeTypeException
from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.input_analysis_data import InputAnalysisData
from dossierfacile_file_analysis.models.supported_content_type import SupportedContentType


class PrepareDataForAnalysis(AbstractBlurryTask):

    def __init__(self):
        super().__init__(task_name="PrepareDataForAnalysis")
        self.local_file_path = os.getenv("LOCAL_FILE_PATH")
        # Valeurs par défaut plus basses pour accélérer Tesseract tout en gardant une qualité OCR correcte
        self.targeted_dpi = int(os.getenv("TARGETED_DPI", 180))
        self.max_long_edge = int(os.getenv("MAX_LONG_EDGE", 1800))  # en pixels
        self.max_pixel_size = int(os.getenv("MAX_PIXEL_SIZE", 2_000_000))
        self.min_long_edge = int(os.getenv("MIN_LONG_EDGE", 1500))  # en pixels
        # Permettre de forcer la sortie en niveaux de gris
        self.force_grayscale = os.getenv("FORCE_GRAYSCALE", "1").lower() in ("1", "true", "yes")

    def has_to_apply(self, context: BlurryExecutionContext) -> bool:
        if context.file_dto is None and context.downloaded_file is None:
            return False
        return True

    def _internal_run(self, context: BlurryExecutionContext):
        if context.downloaded_file.file_type is None:
            raise InvalidMimeTypeException(context.file_dto.id)
        if context.downloaded_file.file_type == SupportedContentType.PDF:
            list_of_images = self._pdf_to_images(context.downloaded_file.file_name, context.downloaded_file.file_path)
            context.input_analysis_data = InputAnalysisData(downloaded_file=context.downloaded_file,
                                                            list_of_images=list_of_images)
        else:
            context.input_analysis_data = InputAnalysisData(downloaded_file=context.downloaded_file)

    def _pdf_to_images(self, pdf_file_name: str, pdf_path: str) -> list:
        logger.info(f"Converting PDF to images for file: {pdf_file_name}")
        image_paths = []

        doc = pymupdf.open(filename=pdf_path)
        try:
            for page in doc:
                page_width = page.rect.width
                page_height = page.rect.height

                # Zoom cible basé sur le DPI souhaité
                zoom = self.targeted_dpi / 72.0

                # Taille prédite avec ce zoom
                predicted_width = page_width * zoom
                predicted_height = page_height * zoom

                # On borne par la longueur max et le nombre total de pixels
                scale_by_edge = self.max_long_edge / max(predicted_width, predicted_height)
                scale_by_mp = math.sqrt(self.max_pixel_size / (predicted_width * predicted_height))
                clamp_factor = min(1.0, scale_by_edge, scale_by_mp)

                # Si trop petit, on remonte au minimum requis
                if clamp_factor == 1.0 and max(predicted_width, predicted_height) < self.min_long_edge:
                    clamp_factor = self.min_long_edge / max(predicted_width, predicted_height)

                effective_zoom = zoom * clamp_factor
                mat = pymupdf.Matrix(effective_zoom, effective_zoom)

                # Préparation des kwargs pour get_pixmap (niveau de gris + pas d’alpha)
                kwargs = {"matrix": mat, "alpha": False}
                if self.force_grayscale and hasattr(pymupdf, "csGRAY"):
                    kwargs["colorspace"] = pymupdf.csGRAY

                pix = page.get_pixmap(**kwargs)
                try:
                    logger.debug(
                        f"Page {page.number}: pts={int(predicted_width)}x{int(predicted_height)} -> "
                        f"px={pix.width}x{pix.height}  (eff_dpi≈{effective_zoom * 72.0:.1f})"
                    )

                    image_path = os.path.join(self.local_file_path or "", f"{pdf_file_name}_{page.number}.png")
                    pix.save(image_path)
                    image_paths.append(image_path)
                finally:
                    pix = None  # Libère la mémoire du pixmap

        finally:
            doc.close()  # Ferme le document PDF

        return image_paths
