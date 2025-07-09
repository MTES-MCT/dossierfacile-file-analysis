from unittest.mock import patch

import pytest

from dossierfacile_file_analysis.data.FileDto import FileDto
from dossierfacile_file_analysis.exceptions.data_not_found import DataNotFoundException
from dossierfacile_file_analysis.executor.tasks.get_data_from_db import GetDataFromDB
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage


@pytest.fixture
def get_data_from_db_task():
    with patch('dossierfacile_file_analysis.executor.tasks.get_data_from_db.DossierFacileDatabaseService') as mock_db_service:
        task = GetDataFromDB()
        task.database_service = mock_db_service
        yield task, mock_db_service


def test_run_success(get_data_from_db_task):
    # Given
    task, mock_db_service = get_data_from_db_task
    mock_file_dto = FileDto(id="1", path="path/to/file.pdf", content_type="application/pdf", encryption_key="",
                            encryption_key_version=1, provider="local")
    mock_db_service.get_file_by_id.return_value = mock_file_dto
    queue_message = BlurryQueueMessage(file_id=1)
    context = BlurryExecutionContext(queue_message)

    # When
    task.run(context)

    # Then
    mock_db_service.get_file_by_id.assert_called_once_with(1)
    assert context.file_dto == mock_file_dto


def test_run_data_not_found(get_data_from_db_task):
    # Given
    task, mock_db_service = get_data_from_db_task
    mock_db_service.get_file_by_id.return_value = None
    queue_message = BlurryQueueMessage(file_id=1)
    context = BlurryExecutionContext(queue_message)

    # When / Then
    with pytest.raises(DataNotFoundException):
        task.run(context)
    mock_db_service.get_file_by_id.assert_called_once_with(1)
    assert context.file_dto is None
