import os.path

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext


class CleanData(AbstractBlurryTask):
    def __init__(self):
        super().__init__(task_name="CleanData")

    def _internal_run(self, context: BlurryExecutionContext):
        logger.info("Starting data cleanup process.")
        try:
            if context.downloaded_file and context.downloaded_file.file_path:
                file_path = context.downloaded_file.file_path
                logger.debug(f"Attempting to delete downloaded file: {file_path}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Successfully deleted file: {file_path}")
                else:
                    logger.warning(f"File not found, skipping deletion: {file_path}")

            if context.input_analysis_data and context.input_analysis_data.list_of_images:
                logger.info("Attempting to delete temporary analysis images.")
                for image_path in context.input_analysis_data.list_of_images:
                    logger.debug(f"Attempting to delete image: {image_path}")
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        logger.debug(f"Successfully deleted image: {image_path}")
                    else:
                        logger.warning(f"Image not found, skipping deletion: {image_path}")
                logger.info("Finished deleting temporary analysis images.")
        except Exception as e:
            logger.error(f"An error occurred during data cleanup: {e}", exc_info=True)
            # On ne propage pas l'exception pour ne pas faire échouer tout le traitement
            # juste pour une erreur de nettoyage.
        logger.info("Data cleanup process finished.")
