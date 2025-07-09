from dossierfacile_file_analysis.models.supported_content_type import SupportedContentType


class DownloadedFile:
    def __init__(self, file_name: str, file_path: str, file_type: str):
        self.file_name = file_name
        self.file_path = file_path
        self.file_type = SupportedContentType.get_supported_content_type(file_type)
