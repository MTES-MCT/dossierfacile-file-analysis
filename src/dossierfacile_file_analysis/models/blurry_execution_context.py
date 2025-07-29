import uuid
from typing import Optional

from dossierfacile_file_analysis.data.file_dto import FileDto
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage
from dossierfacile_file_analysis.models.blurry_result import BlurryResult
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile
from dossierfacile_file_analysis.models.input_analysis_data import InputAnalysisData


class BlurryExecutionContext:

    def __init__(self, queue_message: BlurryQueueMessage):
        self.file_id = queue_message.file_id
        self.execution_id = uuid.uuid4()
        self.file_dto: Optional[FileDto] = None
        self.downloaded_file: Optional[DownloadedFile] = None
        self.input_analysis_data: Optional[InputAnalysisData] = None
        self.blurry_result: Optional[BlurryResult] = None
