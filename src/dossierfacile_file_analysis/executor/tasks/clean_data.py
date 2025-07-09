import os.path

from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext


class CleanData(AbstractBlurryTask):
    def __init__(self):
        super().__init__(task_name="CleanData")

    def _internal_run(self, context: BlurryExecutionContext):
        if context.downloaded_file is not None and os.path.exists(context.downloaded_file.file_path):
            os.remove(context.downloaded_file.file_path)
        if context.input_analysis_data is not None:
            for image in context.input_analysis_data.list_of_images:
                if os.path.exists(image):
                    os.remove(image)
