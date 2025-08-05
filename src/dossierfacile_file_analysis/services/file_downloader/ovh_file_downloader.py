import os
import time
import uuid
import threading

import boto3
from botocore.config import Config

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.data.file_dto import FileDto
from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException
from dossierfacile_file_analysis.services.file_downloader.file_downloader import FileDownloader


class OVHFileDownloader(FileDownloader):
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
            logger.info("Initializing OVHFileDownloader")
            super().__init__()
            self.encrypted_file_path = "/tmp/encrypted_file"
            # Créer le répertoire une seule fois
            if not os.path.exists(self.encrypted_file_path):
                os.makedirs(self.encrypted_file_path, exist_ok=True)
            self._initialized = True

    def _create_s3_client(self):
        """Créer un client S3 par thread pour éviter les conflits"""
        return boto3.client(
            "s3",
            endpoint_url=os.getenv("OVH_S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("OVH_S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("OVH_S3_SECRET_KEY"),
            config=Config(signature_version="s3v4"),
            region_name=os.getenv("OVH_S3_REGION")
        )

    def download_file(self, file_dto: FileDto):
        logger.info("Downloading file from OVH storage")
        start_time = time.time()

        # Générer un nom de fichier unique pour éviter les collisions entre threads
        unique_filename = f"{uuid.uuid4()}_{os.path.basename(file_dto.path)}"
        encrypted_file_path = os.path.join(self.encrypted_file_path, unique_filename)

        try:
            # Créer un client S3 par thread
            s3_client = self._create_s3_client()
            s3_client.download_file(os.getenv("OVH_S3_BUCKET"), file_dto.path, encrypted_file_path)
        except Exception as e:
            raise RetryableException("Failed to download file from OVH storage") from e

        try:
            downloaded_file = self.decrypt_file_with_key(encrypted_file_path, file_dto)
        finally:
            # Toujours nettoyer le fichier temporaire
            if os.path.exists(encrypted_file_path):
                os.remove(encrypted_file_path)

        end_time = time.time()
        logger.info(f"download and decrypt file take : {end_time - start_time:.2f} seconds")
        return downloaded_file
