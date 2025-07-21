from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.executor.tasks.analyse_files import AnalyseFiles
from dossierfacile_file_analysis.executor.tasks.clean_data import CleanData
from dossierfacile_file_analysis.executor.tasks.download_file import DownloadFile
from dossierfacile_file_analysis.executor.tasks.get_data_from_db import GetDataFromDB
from dossierfacile_file_analysis.executor.tasks.prepare_data_for_analysis import PrepareDataForAnalysis
from dossierfacile_file_analysis.executor.tasks.save_blurry_result_to_db import SaveBlurryResultToDB
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage


class BlurryExecutor:
    """
    Executor for blurry file analysis.
    This class is responsible for executing the analysis of blurry files.
    """

    def __init__(self, queue_message: BlurryQueueMessage):
        self.blurry_execution_context = BlurryExecutionContext(queue_message)
        self.blurry_tasks: list[AbstractBlurryTask] = [
            GetDataFromDB(),
            DownloadFile(),
            PrepareDataForAnalysis(),
            AnalyseFiles(),
            SaveBlurryResultToDB()
        ]
        self.cleanTask = CleanData()

    def execute(self):
        """
        Execute the analysis of the blurry file.
        """
        try:
            for task in self.blurry_tasks:
                if task.has_to_apply(self.blurry_execution_context):
                    task.run(self.blurry_execution_context)
                else:
                    logger.info(f"Skipping task: {task.task_name} for file_id: {self.blurry_execution_context.file_id}")
            logger.info(
                f"Blurry file analysis completed for file_id: {self.blurry_execution_context.file_id} with execution_id: {self.blurry_execution_context.execution_id}"
            )
            logger.info(
                f"Blurry result: {self.blurry_execution_context.blurry_result if self.blurry_execution_context.blurry_result else 'No result'}"
            )
        except Exception as e:
            raise e
        finally:
            if self.cleanTask:
                self.cleanTask.run(self.blurry_execution_context)
            else:
                logger.info("No clean task defined, skipping cleanup.")

