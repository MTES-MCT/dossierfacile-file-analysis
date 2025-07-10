import time

import cv2
import numpy as np

from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_result import BlurryResult
from dossierfacile_file_analysis.models.supported_content_type import SupportedContentType


class AnalyseFiles(AbstractBlurryTask):

    def __init__(self):
        super().__init__(task_name="AnalyseFiles")
        self.laplacian_variance_threshold = 250
        self.mean_gray_threshold = 245
        self.proj_threshold = 0.6

    def has_to_apply(self, context: BlurryExecutionContext) -> bool:
        if context.file_dto is None and context.downloaded_file is None and context.input_analysis_data is None:
            return False
        return True

    def _internal_run(self, context: BlurryExecutionContext):
        print("Processing input analysis data...")
        list_of_results: list[BlurryResult] = []
        if context.input_analysis_data.type == SupportedContentType.PDF:
            # Process each image in the list of images
            for image_path in context.input_analysis_data.list_of_images:
                list_of_results.append(self._is_blurry(image_path))
        else:
            # Process the single image file
            list_of_results.append(self._is_blurry(context.input_analysis_data.initial_file))

        if list_of_results:
            # filter result to remove blank images
            filtered_list_of_result = [result for result in list_of_results if not result.is_blank]
            if not filtered_list_of_result:
                most_blurry = list_of_results[0]
            else :
                most_blurry = min(filtered_list_of_result, key=lambda r: r.laplacian_variance)
            context.blurry_result = most_blurry

    def _is_blurry(self, file_path: str):
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        return self._detect_blur_laplacian(gray)

    def _detect_blur_laplacian(self, gray):
        # Calculate variance of Laplacian
        start_time = time.time()

        if np.mean(gray) > self.mean_gray_threshold:  # seuil à ajuster selon les cas
            return BlurryResult(
                laplacian_variance=-1,
                is_blurry=False,
                is_blank=True
            )

        y0, y1 = self._extract_text_band(gray)
        if y0 is None: return BlurryResult(
            laplacian_variance=-1,
            is_blurry=True,
            is_blank=False
        )

        laplacian_var = float(cv2.Laplacian(gray[y0:y1], cv2.CV_64F).var())

        end_time = time.time()
        print(f"Laplacian variance calculation took: {end_time - start_time:.2f} seconds")
        return BlurryResult(
            laplacian_variance=laplacian_var,
            is_blurry=laplacian_var < self.laplacian_variance_threshold,
            is_blank=False
        )

    def _extract_text_band(self, gray):
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=25, C=10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        proj = np.sum(bw // 255, axis=1)
        th = proj.max() * self.proj_threshold
        rows = np.where(proj > th)[0]

        return (int(rows[0]), int(rows[-1])) if rows.size else (None, None)
