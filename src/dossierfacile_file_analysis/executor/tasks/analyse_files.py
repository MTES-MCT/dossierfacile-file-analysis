import time

import os

from pytesseract import image_to_data
import cv2

from dossierfacile_file_analysis.custom_logging.logging_config import logger

from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_result import BlurryResult
from dossierfacile_file_analysis.models.supported_content_type import SupportedContentType


class AnalyseFiles(AbstractBlurryTask):

    def __init__(self):
        super().__init__(task_name="AnalyseFiles")
        self.mean_gray_threshold = 350
        self.average_confidence_threshold = 55
        self.min_ocr_tokens = 15
        self.tesseract_psm = 6
        self.tesseract_oem = 1
        self.tesseract_config = f"--psm {self.tesseract_psm} --oem {self.tesseract_oem} --dpi 300 -c preserve_interword_spaces=1"
        self.tesseract_lang = "fra"
        self.tesseract_timeout = int(os.getenv('TESSERACT_TIMEOUT', '60'))

    def has_to_apply(self, context: BlurryExecutionContext) -> bool:
        if context.file_dto is None and context.downloaded_file is None and context.input_analysis_data is None:
            return False
        return True

    def _internal_run(self, context: BlurryExecutionContext):
        logger.info("Processing input analysis data...")
        list_of_results: list[BlurryResult] = []
        if context.input_analysis_data.type == SupportedContentType.PDF:
            # Process each image in the list of images
            for image_path in context.input_analysis_data.list_of_images:
                result = self._is_blurry(image_path)
                list_of_results.append(result)
                # We skip remaining images if one is not readable
                if result.is_blurry:
                    break
        else:
            # Process the single image file
            list_of_results.append(self._is_blurry(context.input_analysis_data.initial_file))

        if list_of_results:
            # filter result to remove blank images
            filtered_list_of_result = [result for result in list_of_results if not result.is_blank]
            if not filtered_list_of_result:
                most_blurry = list_of_results[0]
            else:
                most_blurry = min(filtered_list_of_result, key=lambda r: r.ocr_mean_score)
            context.blurry_result = most_blurry

    def _is_blurry(self, file_path: str) -> BlurryResult:
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            logger.error(f"Failed to load image: {file_path}")
            return BlurryResult(
                is_blurry=True,
                is_blank=False,
            )

        try:
            result = self._detect_blur(gray)
            return result
        finally:
            # Libérer explicitement la mémoire OpenCV
            del gray

    def is_readable(self, gray_roi) -> tuple[bool, float, int]:
        try:
            data = image_to_data(
                gray_roi,
                output_type='dict',
                config=self.tesseract_config,
                lang=self.tesseract_lang,
                timeout=self.tesseract_timeout
            )
            confs = [int(c) for c in data.get('conf', []) if c != '-1']
            tokens = len(confs)
            if tokens < self.min_ocr_tokens:
                return False, 0.0, tokens
            confs.sort()
            k = max(1, int(tokens * 0.2))
            avg = sum(confs[k:]) / (tokens - k)
            return avg >= self.average_confidence_threshold, float(avg), tokens

        except RuntimeError as e:
            logger.warning(f"Tesseract timeout/error: {e}")
            return False, 0.0, 0
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return False, 0.0, 0

    def _is_blank(self, gray, white_thr=245, ratio=0.985):
        return (gray > white_thr).mean() > ratio

    def _detect_blur(self, gray):
        start_time = time.time()

        if self._is_blank(gray):
            return BlurryResult(is_blurry=False, is_blank=True)

        readable, avg, tokens = self.is_readable(gray)

        logger.info(f"Blurry analysis calculation took: {time.time() - start_time:.2f} s")

        return BlurryResult(
            is_blurry=not readable,
            is_blank=False,
            ocr_mean_score=avg,
            ocr_tokens=tokens
        )