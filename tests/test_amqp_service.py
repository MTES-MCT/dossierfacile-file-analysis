import pytest
from unittest.mock import MagicMock, patch

from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException
from dossierfacile_file_analysis.services.amqp_service import AmqpService


@pytest.fixture
def amqp_service():
    with patch('os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda x: {
            "AMQP_IP": "localhost",
            "AMQP_PORT": "5672",
            "AMQP_QUEUE_NAME": "test_queue",
            "AMQP_LOGIN": "guest",
            "AMQP_PASSWORD": "guest"
        }.get(x)
        service = AmqpService()
        service.connection = MagicMock()
        # Ensure that add_callback_threadsafe directly calls the function passed to it
        service.connection.add_callback_threadsafe.side_effect = lambda f: f()
        service.channel = MagicMock()
        service.executor = MagicMock()
        yield service

def test_message_callback_success(amqp_service):
    # Given
    mock_channel = MagicMock()
    amqp_service.channel = mock_channel # Assign the mock channel to the service
    mock_method_frame = MagicMock(delivery_tag=1)
    mock_properties = MagicMock(headers={})
    mock_body = b"test_message"

    with patch('dossierfacile_file_analysis.services.blurry_message_processor.BlurryMessageProcessor.process') as mock_process:
        mock_process.return_value = None

        # When
        amqp_service._message_callback(mock_channel, mock_method_frame, mock_properties, mock_body)

        # Then
        amqp_service.executor.submit.assert_called_once_with(mock_process, mock_body)
        future = amqp_service.executor.submit.return_value
        future.add_done_callback.assert_called_once()

        # Simulate the callback being called
        callback = future.add_done_callback.call_args[0][0]
        callback(future)

        # Verify that basic_ack was called
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=mock_method_frame.delivery_tag)
        amqp_service.connection.add_callback_threadsafe.assert_called_once()


def test_message_callback_retryable_exception_under_max_retries(amqp_service):
    # Given
    mock_channel = MagicMock()
    amqp_service.channel = mock_channel # Assign the mock channel to the service
    mock_method_frame = MagicMock(delivery_tag=1)
    mock_properties = MagicMock(headers={"x-retry-count": 0})
    mock_body = b"test_message"

    with patch('dossierfacile_file_analysis.services.blurry_message_processor.BlurryMessageProcessor.process') as mock_process :
        mock_process.side_effect = RetryableException("Test retryable error")

        # When
        amqp_service._message_callback(mock_channel, mock_method_frame, mock_properties, mock_body)

        # Then
        future = amqp_service.executor.submit.return_value
        future.result.side_effect = RetryableException("Test retryable error")
        callback = future.add_done_callback.call_args[0][0]
        callback(future)

        # The callback passed to add_callback_threadsafe should publish the message
        # We need to get the function passed to add_callback_threadsafe and call it
        # Since add_callback_threadsafe is mocked to call the function directly, basic_publish should have been called
        mock_channel.basic_publish.assert_called_once()
        args, kwargs = mock_channel.basic_publish.call_args
        assert kwargs['routing_key'] == amqp_service.queue_name
        assert kwargs['body'] == mock_body
        assert kwargs['properties'].headers['x-retry-count'] == 1
        mock_channel.basic_ack.assert_called_once()

def test_message_callback_retryable_exception_max_retries_reached(amqp_service):
    # Given
    mock_channel = MagicMock()
    amqp_service.channel = mock_channel # Assign the mock channel to the service
    mock_method_frame = MagicMock(delivery_tag=1)
    mock_properties = MagicMock(headers={"x-retry-count": 3}) # Max retries reached
    mock_body = b"test_message"

    with patch('dossierfacile_file_analysis.services.blurry_message_processor.BlurryMessageProcessor.process') as mock_process:
        mock_process.side_effect = RetryableException("Test retryable error")

        # When
        amqp_service._message_callback(mock_channel, mock_method_frame, mock_properties, mock_body)

        # Then
        future = amqp_service.executor.submit.return_value
        callback = future.add_done_callback.call_args[0][0]
        callback(future)

        # Expect ack, not retry
        amqp_service.connection.add_callback_threadsafe.assert_called_once()
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=mock_method_frame.delivery_tag)
        mock_channel.basic_publish.assert_not_called()

def test_message_callback_non_retryable_exception(amqp_service):
    # Given
    mock_channel = MagicMock()
    amqp_service.channel = mock_channel # Assign the mock channel to the service
    mock_method_frame = MagicMock(delivery_tag=1)
    mock_properties = MagicMock(headers={})
    mock_body = b"test_message"

    with patch('dossierfacile_file_analysis.services.blurry_message_processor.BlurryMessageProcessor.process') as mock_process:
        mock_process.side_effect = ValueError("Non-retryable error")

        # When
        amqp_service._message_callback(mock_channel, mock_method_frame, mock_properties, mock_body)

        # Then
        future = amqp_service.executor.submit.return_value
        callback = future.add_done_callback.call_args[0][0]
        callback(future)

        # Expect ack, not retry
        amqp_service.connection.add_callback_threadsafe.assert_called_once()
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=mock_method_frame.delivery_tag)
        mock_channel.basic_publish.assert_not_called()