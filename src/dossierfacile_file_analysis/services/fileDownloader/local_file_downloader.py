import os

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.data.FileDto import FileDto
from dossierfacile_file_analysis.services.fileDownloader.file_downloader import FileDownloader


class LocalFileDownloader(FileDownloader):

    def __init__(self):
        super().__init__()
        self.local_provider_path = os.getenv("LOCAL_FILE_PROVIDER_PATH", None)

    def download_file(self, file_dto: FileDto):
        file_path = f"{self.local_provider_path}/{file_dto.path}"
        logger.info(f"Downloading encrypted file from {file_path}")
        # Destination du fichier déchiffré
        return self.decrypt_file_with_key(file_path, file_dto)
