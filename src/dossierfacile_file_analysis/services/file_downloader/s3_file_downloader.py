import base64
import hashlib
import os
import time
import threading
import uuid

import boto3
from botocore.config import Config

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.data.file_dto import FileDto
from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile
from dossierfacile_file_analysis.services.file_downloader.file_downloader import FileDownloader


class S3FileDownloader(FileDownloader):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            logger.info("Initializing s3 file downloader")
            super().__init__()
            self.bucket_name = os.getenv("S3_RAW_BUCKET_NAME")
            self._initialized = True

    def _create_s3_client(self):
        """Créer un client S3 par thread pour éviter les conflits"""
        return boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("S3_FILE_ANALYSIS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_FILE_ANALYSIS_SECRET_KEY"),
            config=Config(signature_version="s3v4"),
            region_name=os.getenv("S3_REGION")
        )

    def download_file(self, file_dto: FileDto):
        logger.info("Downloading file from s3 storage")
        start_time = time.time()
        try:
            # Générer un nom de fichier unique pour éviter les collisions entre threads
            unique_filename = f"{uuid.uuid4()}_{file_dto.get_system_name()}"
            output_path = f"{self.local_file_path}{unique_filename}"

            # From the file_dto, we get the encryption key and compute the necessary headers for SSE-C
            sse_customer_key = base64.b64encode(file_dto.encryption_key).decode('utf-8')
            md5_hash = hashlib.md5(file_dto.encryption_key).digest()
            sse_customer_key_md5 = base64.b64encode(md5_hash).decode('utf-8')

            # Créer un client S3 par thread
            s3_client = self._create_s3_client()

            with open(output_path, 'wb') as data:
                s3_client.download_fileobj(
                    self.bucket_name,
                    file_dto.path,
                    data,
                    ExtraArgs={
                        "SSECustomerAlgorithm": "AES256",
                        "SSECustomerKey": sse_customer_key,
                        "SSECustomerKeyMD5": sse_customer_key_md5
                    })

        except Exception as e:
            raise RetryableException("Failed to download file from S3 storage") from e
        end_time = time.time()
        logger.info(f"download file take : {end_time - start_time:.2f} seconds")
        return DownloadedFile(file_name=unique_filename, file_path=output_path,
                              file_type=file_dto.content_type)
