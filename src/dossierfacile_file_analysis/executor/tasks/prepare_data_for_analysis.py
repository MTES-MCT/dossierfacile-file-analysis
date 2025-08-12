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
        self.targeted_dpi = 220
        self.max_long_edge = 2500 # in pixels
        self.max_pixel_size = 4_000_000
        self.min_long_edge = 1500 # in pixels

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

                # Targeted zoom for 300 DPI
                zoom = self.targeted_dpi / 72.0

                # Predicted size at 300 dpi
                predicted_width = page_width * zoom
                predicted_height = page_height * zoom

                # We clamp the size to ensure it does not exceed the maximum pixel size
                scale_by_dge = self.max_long_edge / max(predicted_width, predicted_height)
                scale_by_mp = math.sqrt(self.max_pixel_size / (predicted_width * predicted_height))
                clamp_factor = min(1.0, scale_by_dge, scale_by_mp)

                # If the picture is too small
                if clamp_factor == 1.0 and max(predicted_width, predicted_height) < self.min_long_edge:
                    clamp_factor = self.min_long_edge / max(predicted_width, predicted_height)

                effective_zoom = zoom * clamp_factor
                mat = pymupdf.Matrix(effective_zoom, effective_zoom)

                pix = page.get_pixmap(matrix=mat)
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
