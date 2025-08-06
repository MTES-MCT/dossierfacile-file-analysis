import json

import elasticapm

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.exceptions.invalid_message_body_format import InvalidMessageBodyFormat
from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException
from dossierfacile_file_analysis.executor.blurry_executor import BlurryExecutor
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage
from dossierfacile_file_analysis.services.dossier_facile_database_service import database_service


class BlurryMessageProcessor:

    @staticmethod
    def process(body, retry_count: int):
        client = elasticapm.get_client()
        client.begin_transaction("task")
        decoded_body = body.decode()
        logger.info(f"Received message: {decoded_body}")
        try:
            blurry_queue_message = BlurryQueueMessage.from_dict(json.loads(decoded_body))
        except Exception as e:
            client.capture_exception()
            client.end_transaction("message_processing", "failure")
            raise InvalidMessageBodyFormat(e)
        try:
            if blurry_queue_message is None:
                client.end_transaction("message_processing", "failure")
                raise InvalidMessageBodyFormat("Message body is empty or invalid")

            elasticapm.set_custom_context({"blurry_queue_message": blurry_queue_message.to_dict()})
            executor = BlurryExecutor(blurry_queue_message)
            executor.execute()

            client.end_transaction("message_processing", "success")
            return True
        except Exception as e:
            client.capture_exception()
            client.end_transaction("message_processing", "failure")
            # If the exception is not retryable we save in database the failed analysis
            # or if exception is retryable and the retry count is greater than 3 we save the failed analysis
            if not isinstance(e, RetryableException) or (isinstance(e, RetryableException) and retry_count >= 3):
                try:
                    database_service.save_failed_analysis(blurry_queue_message.file_id)
                except Exception as save_error:
                    logger.error(f"Failed to save failed analysis: {save_error}")
            raise e
