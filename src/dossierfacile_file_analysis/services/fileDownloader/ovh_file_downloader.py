import os
import time

import boto3
from botocore.config import Config

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.data.FileDto import FileDto
from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException
from dossierfacile_file_analysis.services.fileDownloader.file_downloader import FileDownloader


class OVHFileDownloader(FileDownloader):

    def __init__(self):
        logger.info("Initializing OVHFileDownloader")
        super().__init__()
        self.encrypted_file_path = "/tmp/encrypted_file"
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("OVH_S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("OVH_S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("OVH_S3_SECRET_KEY"),
            config=Config(signature_version="s3v4"),
            region_name=os.getenv("OVH_S3_REGION")  # Adapter selon la région OVH
        )

    def download_file(self, file_dto: FileDto):
        logger.info("Downloading file from OVH storage")
        start_time = time.time()
        if not os.path.exists(self.encrypted_file_path):
            os.makedirs(self.encrypted_file_path)
        encrypted_file_path = os.path.join(self.encrypted_file_path, os.path.basename(file_dto.path))
        try:
            self.s3_client.download_file(os.getenv("OVH_S3_BUCKET"), file_dto.path, encrypted_file_path)
        except Exception as e:
            raise RetryableException("Failed to download file from OVH storage") from e
        downloaded_file = self.decrypt_file_with_key(encrypted_file_path, file_dto)
        os.remove(encrypted_file_path)
        end_time = time.time()
        logger.info(f"download and decrypt file take : {end_time - start_time:.2f} seconds")
        return downloaded_file
