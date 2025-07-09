import os

import pymupdf

from dossierfacile_file_analysis.exceptions.invalid_mime_type import InvalidMimeTypeException
from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.input_analysis_data import InputAnalysisData
from dossierfacile_file_analysis.models.supported_content_type import SupportedContentType


class PrepareDataForAnalysis(AbstractBlurryTask):

    def __init__(self):
        super().__init__(task_name="PrepareDataForAnalysis")
        self.local_file_path = os.getenv("LOCAL_FILE_PATH")

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
        zoom_x = 2.0
        zoom_y = 2.0
        mat = pymupdf.Matrix(zoom_x, zoom_y)

        print("Converting PDF to images for file: ", pdf_file_name)
        image_paths = []

        doc = pymupdf.open(filename=pdf_path)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            image_path = os.path.join(self.local_file_path or "", f"{pdf_file_name}_{page.number}.png")
            pix.save(image_path)
            image_paths.append(image_path)

        return image_paths
