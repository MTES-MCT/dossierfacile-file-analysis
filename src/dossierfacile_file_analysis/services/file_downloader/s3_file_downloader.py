import os
import time

import boto3
from botocore.config import Config

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.data.file_dto import FileDto
from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile
from dossierfacile_file_analysis.services.file_downloader.file_downloader import FileDownloader


class S3FileDownloader(FileDownloader):

    def __init__(self):
        logger.info("Initializing s3 file downloader")
        super().__init__()
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            config=Config(signature_version="s3v4"),
            region_name=os.getenv("S3_REGION")
        )
        self.bucket_name = os.getenv("S3_BUCKET")

    def download_file(self, file_dto: FileDto):
        logger.info("Downloading file from s3 storage")
        start_time = time.time()
        try:
            output_path = f"{self.local_file_path}{file_dto.get_system_name()}"
            self.s3_client.download_file(self.bucket_name, file_dto.path, output_path)
        except Exception as e:
            raise RetryableException("Failed to download file from OVH storage") from e
        end_time = time.time()
        logger.info(f"download file take : {end_time - start_time:.2f} seconds")
        return DownloadedFile(file_name=file_dto.path, file_path=output_path, file_type=file_dto.content_type)
