from unittest.mock import patch, MagicMock

import pytest

from dossierfacile_file_analysis.data.FileDto import FileDto
from dossierfacile_file_analysis.executor.tasks.download_file import DownloadFile
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile


@pytest.fixture
def download_file_task():
    with patch(
            'dossierfacile_file_analysis.services.fileDownloader.local_file_downloader.LocalFileDownloader') as mock_local_downloader, \
            patch(
                'dossierfacile_file_analysis.services.fileDownloader.ovh_file_downloader.OVHFileDownloader') as mock_ovh_downloader:
        task = DownloadFile()
        task.local_file_downloader = mock_local_downloader
        task.ovh_file_downloader = mock_ovh_downloader
        yield task, mock_local_downloader, mock_ovh_downloader


def test_has_to_apply(download_file_task):
    # Given
    task, _, _ = download_file_task
    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))

    # When
    context.file_dto = None
    # Then
    assert not task.has_to_apply(context)

    # When
    context.file_dto = MagicMock()
    # Then
    assert task.has_to_apply(context)


def test_run_local_provider(download_file_task):
    # Given
    task, mock_local_downloader, mock_ovh_downloader = download_file_task
    file_dto = FileDto(id="1", path="path/to/file.pdf", content_type="application/pdf", encryption_key="",
                       encryption_key_version=1, provider="LOCAL")
    queue_message = BlurryQueueMessage(file_id=1)
    context = BlurryExecutionContext(queue_message)
    context.file_dto = file_dto

    mock_downloaded_file = DownloadedFile(file_name="file.pdf", file_path="/tmp/file.pdf", file_type="application/pdf")
    mock_local_downloader.download_file.return_value = mock_downloaded_file

    # When
    task.run(context)

    # Then
    mock_local_downloader.download_file.assert_called_once_with(file_dto)
    mock_ovh_downloader.download_file.assert_not_called()
    assert context.downloaded_file == mock_downloaded_file


def test_run_ovh_provider(download_file_task):
    # Given
    task, mock_local_downloader, mock_ovh_downloader = download_file_task
    file_dto = FileDto(id="1", path="path/to/file.pdf", content_type="application/pdf", encryption_key="",
                       encryption_key_version=1, provider="OVH")
    queue_message = BlurryQueueMessage(file_id=1)
    context = BlurryExecutionContext(queue_message)
    context.file_dto = file_dto

    mock_downloaded_file = DownloadedFile(file_name="file.pdf", file_path="/tmp/file.pdf", file_type="application/pdf")
    mock_ovh_downloader.download_file.return_value = mock_downloaded_file

    # When
    task.run(context)

    # Then
    mock_ovh_downloader.download_file.assert_called_once_with(file_dto)
    mock_local_downloader.download_file.assert_not_called()
    assert context.downloaded_file == mock_downloaded_file
