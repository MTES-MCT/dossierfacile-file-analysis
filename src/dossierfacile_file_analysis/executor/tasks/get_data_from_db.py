from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.exceptions.data_not_found import DataNotFoundException
from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.services.dossier_facile_database_service import database_service


class GetDataFromDB(AbstractBlurryTask):
    """
    This class is responsible for retrieving data from the database.
    It is used to fetch data that will be processed by other tasks in the pipeline.
    """

    def __init__(self):
        super().__init__(task_name="GetDataFromDB")

    def _internal_run(self, context: BlurryExecutionContext):
        try:
            data = database_service.get_file_by_id(context.file_id)
            if data is None:
                raise DataNotFoundException(file_id=context.file_id)
            context.file_dto = data
        except DataNotFoundException as e:
            logger.error(f"❌ Data not found for file_id {context.file_id}: {e}")
            raise e
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching data for file_id {context.file_id}: {e}")
            raise DataNotFoundException(file_id=context.file_id) from e
