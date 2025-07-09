from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from dossierfacile_file_analysis.executor.tasks.analyse_files import AnalyseFiles
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage
from dossierfacile_file_analysis.models.blurry_result import BlurryResult
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile
from dossierfacile_file_analysis.models.input_analysis_data import InputAnalysisData


@pytest.fixture
def analyse_files_task():
    with patch('cv2.imread') as mock_imread, \
            patch('cv2.Laplacian') as mock_laplacian, \
            patch('cv2.adaptiveThreshold') as mock_adaptive_threshold, \
            patch('cv2.getStructuringElement'), \
            patch('cv2.morphologyEx') as mock_morphology_ex, \
            patch('numpy.mean') as mock_np_mean:
        task = AnalyseFiles()

        # Mock for _extract_text_band to return a valid band
        mock_adaptive_threshold.return_value = np.array([[255, 0], [255, 0]])
        mock_morphology_ex.return_value = np.array([[255, 0], [255, 0]])

        mocks = {
            "imread": mock_imread,
            "laplacian": mock_laplacian,
            "np_mean": mock_np_mean
        }
        yield task, mocks


def test_has_to_apply(analyse_files_task):
    task, _ = analyse_files_task
    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))

    context.input_analysis_data = None
    assert not task.has_to_apply(context)

    context.input_analysis_data = MagicMock()
    assert task.has_to_apply(context)


def test_run_single_blurry_image(analyse_files_task):
    task, mocks = analyse_files_task

    # Simulate a blurry image
    mocks["np_mean"].return_value = 100  # Not a blank image
    mock_laplacian_result = MagicMock()
    mock_laplacian_result.var.return_value = 100  # Low variance = blurry
    mocks["laplacian"].return_value = mock_laplacian_result

    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))
    downloaded_file = DownloadedFile(file_name="test.jpg", file_path="/tmp/test.jpg", file_type="image/jpeg")
    context.input_analysis_data = InputAnalysisData(downloaded_file=downloaded_file)

    task.run(context)

    assert isinstance(context.blurry_result, BlurryResult)
    assert context.blurry_result.is_blurry is True
    assert context.blurry_result.laplacian_variance == 100


def test_run_single_clear_image(analyse_files_task):
    task, mocks = analyse_files_task

    # Simulate a clear image
    mocks["np_mean"].return_value = 100
    mock_laplacian_result = MagicMock()
    mock_laplacian_result.var.return_value = 500  # High variance = clear
    mocks["laplacian"].return_value = mock_laplacian_result

    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))
    downloaded_file = DownloadedFile(file_name="test.jpg", file_path="/tmp/test.jpg", file_type="image/jpeg")
    context.input_analysis_data = InputAnalysisData(downloaded_file=downloaded_file)

    task.run(context)

    assert context.blurry_result.is_blurry is False
    assert context.blurry_result.laplacian_variance == 500


def test_run_pdf_with_mixed_images(analyse_files_task):
    task, mocks = analyse_files_task

    # Simulate multiple images with different properties
    # 1. Blurry, 2. Clear, 3. Blank
    mocks["np_mean"].side_effect = [100, 100, 250]  # Third one is blank
    mock_laplacian_result_blurry = MagicMock()
    mock_laplacian_result_blurry.var.return_value = 100
    mock_laplacian_result_clear = MagicMock()
    mock_laplacian_result_clear.var.return_value = 500
    mocks["laplacian"].side_effect = [mock_laplacian_result_blurry, mock_laplacian_result_clear]

    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))
    downloaded_file = DownloadedFile(file_name="test.pdf", file_path="/tmp/test.pdf", file_type="application/pdf")
    context.input_analysis_data = InputAnalysisData(
        downloaded_file=downloaded_file,
        list_of_images=["/tmp/img1.png", "/tmp/img2.png", "/tmp/img3.png"]
    )

    task.run(context)

    # The blank image should be filtered out.
    # The result should be the one with the lowest variance (the most blurry one).
    assert mocks["imread"].call_count == 3
    assert context.blurry_result.is_blurry is True
    assert context.blurry_result.laplacian_variance == 100
