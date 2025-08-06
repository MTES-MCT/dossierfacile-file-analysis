from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.services.file_downloader.local_file_downloader import local_file_downloader
from dossierfacile_file_analysis.services.file_downloader.ovh_file_downloader import ovh_file_downloader
from dossierfacile_file_analysis.services.file_downloader.s3_file_downloader import s3_file_downloader


class DownloadFile(AbstractBlurryTask):

    def __init__(self):
        super().__init__(task_name="DownloadFile")

    def has_to_apply(self, context: BlurryExecutionContext) -> bool:
        if context.file_dto is None:
            return False
        return True

    def _internal_run(self, context: BlurryExecutionContext):
        match context.file_dto.provider:
            case "LOCAL":
                downloaded_file = local_file_downloader.download_file(context.file_dto)
                context.downloaded_file = downloaded_file
            case "OVH":
                downloaded_file = ovh_file_downloader.download_file(context.file_dto)
                context.downloaded_file = downloaded_file
            case "S3":
                downloaded_file = s3_file_downloader.download_file(context.file_dto)
                context.downloaded_file = downloaded_file
