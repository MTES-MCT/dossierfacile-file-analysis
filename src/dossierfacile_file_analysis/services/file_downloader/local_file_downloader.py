import os
import threading

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.data.file_dto import FileDto
from dossierfacile_file_analysis.services.file_downloader.file_downloader import FileDownloader


class LocalFileDownloader(FileDownloader):
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
            super().__init__()
            self.local_provider_path = os.getenv("LOCAL_FILE_PROVIDER_PATH", None)
            self._initialized = True

    def download_file(self, file_dto: FileDto):
        file_path = f"{self.local_provider_path}/{file_dto.path}"
        logger.info(f"Downloading encrypted file from {file_path}")
        # Destination du fichier déchiffré
        return self.decrypt_file_with_key(file_path, file_dto)
