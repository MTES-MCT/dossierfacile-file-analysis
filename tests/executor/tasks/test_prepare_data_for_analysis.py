import os
import glob
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from dossierfacile_file_analysis.exceptions.invalid_mime_type import InvalidMimeTypeException
from dossierfacile_file_analysis.executor.tasks.prepare_data_for_analysis import PrepareDataForAnalysis
from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile
from dossierfacile_file_analysis.models.input_analysis_data import InputAnalysisData


@pytest.fixture
def prepare_data_task():
    with patch.dict(os.environ, {"LOCAL_FILE_PATH": "/tmp"}), patch('pymupdf.open') as mock_pymupdf_open:
        task = PrepareDataForAnalysis()
        yield task, mock_pymupdf_open


def test_has_to_apply(prepare_data_task):
    # Given
    task, _ = prepare_data_task
    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))

    # When
    context.file_dto = None
    context.downloaded_file = None
    # Then
    assert not task.has_to_apply(context)

    # When
    context.downloaded_file = MagicMock()
    # Then
    assert task.has_to_apply(context)


def test_run_invalid_mime_type(prepare_data_task):
    # Given
    task, _ = prepare_data_task
    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))
    context.downloaded_file = DownloadedFile(file_name="file.txt", file_path="/tmp/file.txt", file_type="text/plain")
    context.file_dto = MagicMock()
    context.downloaded_file.file_type = None  # Simulate invalid type

    # When / Then
    with pytest.raises(InvalidMimeTypeException):
        task.run(context)


def test_run_pdf_conversion(prepare_data_task):
    # Given
    task, mock_pymupdf_open = prepare_data_task

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.number = 0
    mock_pix = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.__iter__.return_value = [mock_page]
    mock_pymupdf_open.return_value = mock_doc

    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))
    context.downloaded_file = DownloadedFile(file_name="test.pdf", file_path="/tmp/test.pdf",
                                             file_type="application/pdf")

    # When
    task.run(context)

    # Then
    mock_pymupdf_open.assert_called_once_with(filename="/tmp/test.pdf")
    mock_page.get_pixmap.assert_called_once()
    mock_pix.save.assert_called_once_with("/tmp/test.pdf_0.png")
    assert isinstance(context.input_analysis_data, InputAnalysisData)
    assert context.input_analysis_data.list_of_images == ["/tmp/test.pdf_0.png"]


def test_run_other_file_type(prepare_data_task):
    # Given
    task, mock_pymupdf_open = prepare_data_task
    context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))
    context.downloaded_file = DownloadedFile(file_name="image.jpg", file_path="/tmp/image.jpg", file_type="image/jpeg")

    # When
    task.run(context)

    # Then
    mock_pymupdf_open.assert_not_called()
    assert isinstance(context.input_analysis_data, InputAnalysisData)
    assert context.input_analysis_data.list_of_images == []


def test_real_pdf_conversion():
    """Test real PDF conversion using files in tests/data/pdf"""
    # Given
    test_pdf_dir = os.path.join(os.path.dirname(__file__), "../../data/pdf")
    pdf_files = glob.glob(os.path.join(test_pdf_dir, "*.pdf"))

    if not pdf_files:
        pytest.skip("No PDF files found in tests/data/pdf directory")

    # Create temporary directory for output images
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.dict(os.environ, {"LOCAL_FILE_PATH": temp_dir}):
            task = PrepareDataForAnalysis()

            for pdf_file in pdf_files:
                print(f"\n🔍 Testing PDF file: {os.path.basename(pdf_file)}")

                # When
                context = BlurryExecutionContext(BlurryQueueMessage(file_id=1))
                context.downloaded_file = DownloadedFile(
                    file_name=os.path.basename(pdf_file).replace('.pdf', ''),
                    file_path=pdf_file,
                    file_type="application/pdf"
                )

                # Execute the conversion
                task.run(context)

                # Then - verify conversion was successful
                assert context.input_analysis_data is not None
                assert isinstance(context.input_analysis_data, InputAnalysisData)
                assert hasattr(context.input_analysis_data, 'list_of_images')
                assert len(context.input_analysis_data.list_of_images) > 0

                # Verify images were actually created
                for image_path in context.input_analysis_data.list_of_images:
                    assert os.path.exists(image_path), f"Image not created: {image_path}"
                    file_size = os.path.getsize(image_path)
                    assert file_size > 0, f"Empty image file: {image_path}"
                    print(f"  ✅ Generated image: {os.path.basename(image_path)} ({file_size / (1024 * 1024):.2f} MB)")

                print(f"  📊 Total images generated: {len(context.input_analysis_data.list_of_images)}")
