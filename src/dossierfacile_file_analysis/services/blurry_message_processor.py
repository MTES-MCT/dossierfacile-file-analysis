import json
import elasticapm

from dossierfacile_file_analysis.exceptions.invalid_message_body_format import InvalidMessageBodyFormat
from dossierfacile_file_analysis.models.blurry_queue_message import BlurryQueueMessage


class BlurryMessageProcessor:

    @staticmethod
    def process(body):
        client = elasticapm.get_client()
        client.begin_transaction("task")
        try:
            decoded_body = body.decode()
            print(f"Received message: {decoded_body}")
            try:
                blurry_queue_message = BlurryQueueMessage.from_dict(json.loads(decoded_body))
                elasticapm.set_custom_context({"blurry_queue_message": blurry_queue_message.to_dict()})
            except Exception as e:
                client.capture_exception()
                client.end_transaction("message_processing", "failure")
                raise InvalidMessageBodyFormat(e)

            if blurry_queue_message is None:
                client.end_transaction("message_processing", "failure")
                raise InvalidMessageBodyFormat("Message body is empty or invalid")

            client.end_transaction("message_processing", "success")
            return True
        except Exception as e:
            client.capture_exception()
            client.end_transaction("message_processing", "failure")
            raise e
