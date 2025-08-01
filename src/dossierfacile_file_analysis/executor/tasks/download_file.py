from dossierfacile_file_analysis.executor.tasks.abstract_blurry_task import AbstractBlurryTask
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.services.file_downloader.local_file_downloader import LocalFileDownloader
from dossierfacile_file_analysis.services.file_downloader.ovh_file_downloader import OVHFileDownloader
from dossierfacile_file_analysis.services.file_downloader.s3_file_downloader import S3FileDownloader


class DownloadFile(AbstractBlurryTask):

    def __init__(self):
        super().__init__(task_name="DownloadFile")
        self.local_file_downloader = LocalFileDownloader()
        self.ovh_file_downloader = OVHFileDownloader()
        self.s3_file_downloader = S3FileDownloader()

    def has_to_apply(self, context: BlurryExecutionContext) -> bool:
        if context.file_dto is None:
            return False
        return True

    def _internal_run(self, context: BlurryExecutionContext):
        match context.file_dto.provider:
            case "LOCAL":
                downloaded_file = self.local_file_downloader.download_file(context.file_dto)
                context.downloaded_file = downloaded_file
            case "OVH":
                downloaded_file = self.ovh_file_downloader.download_file(context.file_dto)
                context.downloaded_file = downloaded_file
            case "S3":
                downloaded_file = self.s3_file_downloader.download_file(context.file_dto)
                context.downloaded_file = downloaded_file
