from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.services.dossier_facile_database_service import DossierFacileDatabaseService


class SaveBlurryResultToDB(AbstractBlurryTask):
    def __init__(self):
        super().__init__(task_name="SaveBlurryResultToDB")
        self.database_service = DossierFacileDatabaseService()

    def has_to_apply(self, context: BlurryExecutionContext) -> bool:
        if context.file_dto is None and context.blurry_result is None:
            return False
        return True

    def _internal_run(self, context: BlurryExecutionContext):
        self.database_service.save_blurry_result(
            file_id=context.file_dto.id,
            blurry_result=context.blurry_result,
        )
