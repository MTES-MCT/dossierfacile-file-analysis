from dossierfacile_file_analysis.exceptions.data_not_found import DataNotFoundException
from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.services.dossier_facile_database_service import DossierFacileDatabaseService


class GetDataFromDB(AbstractBlurryTask):
    """
    This class is responsible for retrieving data from the database.
    It is used to fetch data that will be processed by other tasks in the pipeline.
    """

    def __init__(self):
        super().__init__(task_name="GetDataFromDB")
        self.database_service = DossierFacileDatabaseService()

    def _internal_run(self, context: BlurryExecutionContext):
        data = self.database_service.get_file_by_id(context.file_id)
        if data is None:
            raise DataNotFoundException(file_id=context.file_id)
        context.file_dto = data


