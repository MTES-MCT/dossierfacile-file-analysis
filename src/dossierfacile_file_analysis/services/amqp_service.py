import os
import time
from concurrent.futures.thread import ThreadPoolExecutor

import pika
from pika.exceptions import AMQPConnectionError

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException
from dossierfacile_file_analysis.exceptions.duplicate_key_exception import DuplicateKeyException
from dossierfacile_file_analysis.services.arg_provider_service import arg_provider_service
from dossierfacile_file_analysis.services.blurry_message_processor import BlurryMessageProcessor
from dossierfacile_file_analysis.services.dossier_facile_database_service import database_service

from pika.exceptions import ConnectionWrongStateError, ChannelWrongStateError, ChannelClosedByBroker, StreamLostError


class AmqpService:
    def __init__(self):
        self.amqp_ip = os.getenv("AMQP_IP")
        self.amqp_port = int(os.getenv("AMQP_PORT"))
        self.queue_name = os.getenv("AMQP_QUEUE_NAME")
        self.amqp_login = os.getenv("AMQP_LOGIN")
        self.amqp_password = os.getenv("AMQP_PASSWORD")
        self.executor = None
        self.connection = None
        self.channel = None

    def _connect(self):
        """Establishes a connection to the RabbitMQ server."""
        if not self.amqp_ip:
            raise ValueError("AMQP_IP environment variable not set.")
        if not self.amqp_port:
            raise ValueError("AMQP_PORT environment variable not set.")
        if not self.amqp_login:
            raise ValueError("AMQP_LOGIN environment variable not set.")
        if not self.amqp_password:
            raise ValueError("AMQP_PASSWORD environment variable not set.")
        credentials = pika.PlainCredentials(self.amqp_login, self.amqp_password)
        parameters = pika.ConnectionParameters(self.amqp_ip, self.amqp_port, "/", credentials=credentials)
        while True:
            try:
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.queue_name, durable=True)
                logger.info("✅ Successfully connected to RabbitMQ")
                return
            except AMQPConnectionError as e:
                logger.error(f"❌ Failed to connect to RabbitMQ: {e}. Retrying in 10 seconds...")
                time.sleep(10)

    def _safe_schedule(self, fn, action_name: str = "callback"):
        """Schedule a function on the pika I/O thread if connection is open, otherwise log and skip.
        Avoids ConnectionWrongStateError when the connection is closed/closing.
        """
        try:
            if self.connection and getattr(self.connection, "is_open", False):
                self.connection.add_callback_threadsafe(fn)
            else:
                logger.warning(f"Cannot schedule {action_name}: connection is closed or closing; skipping.")
        except (ConnectionWrongStateError, StreamLostError) as e:
            logger.warning(f"Failed to schedule {action_name}: connection is not usable: {e}")

    def _message_callback(self, channel, method_frame, properties, body):
        delivery_tag = method_frame.delivery_tag
        logger.info(
            f"📥 Received message from queue '{self.queue_name}': {body.decode()}; delivery_tag={delivery_tag}; header_frame={properties}"
        )

        def _ack():
            try:
                # Check that the channel is still open before ack
                if channel and getattr(channel, "is_open", False):
                    channel.basic_ack(delivery_tag=delivery_tag)
                else:
                    logger.warning(
                        "Channel is closed when trying to ack; skipping ack (message will be re-queued by broker).")
            except (ChannelWrongStateError, ChannelClosedByBroker, StreamLostError) as e:
                logger.warning(f"Failed to ack message {delivery_tag}: {e}")

        def _retry_message():
            try:
                if not (self.connection and getattr(self.connection, "is_open", False) and channel and getattr(channel,
                                                                                                               "is_open",
                                                                                                               False)):
                    logger.warning("Cannot publish retry: connection/channel is closed; skipping retry publish.")
                    return
                retry_delay_ms = 5000  # 5 seconds
                retry_queue = f"{self.queue_name}_retry_5s"

                channel.queue_declare(
                    queue=retry_queue,
                    durable=True,
                    arguments={
                        "x-message-ttl": retry_delay_ms,
                        "x-dead-letter-exchange": "",
                        "x-dead-letter-routing-key": self.queue_name
                    }
                )
                headers = properties.headers or {}
                new_properties = pika.BasicProperties(
                    headers={"x-retry-count": headers.get('x-retry-count', 0) + 1}
                )
                channel.basic_publish(
                    exchange='',
                    routing_key=retry_queue,
                    body=body,
                    properties=new_properties
                )
            except (ChannelWrongStateError, ChannelClosedByBroker, StreamLostError) as e:
                logger.warning(f"Failed to publish retry for message {delivery_tag}: {e}")

        def _on_done(future):
            try:
                future.result()
            except DuplicateKeyException as e:
                # No retry in this case, just ack
                logger.info(f"ℹ️ Analysis already exists for file_id {e.file_id}, acknowledging message without retry")
            except RetryableException as e:
                logger.warning(f"⚠️ Error processing message: {e}")
                headers = properties.headers or {}
                retry_count = headers.get('x-retry-count', 0)
                if retry_count < 3:
                    logger.info(f"🔄 Retrying message (attempt {retry_count + 1})")
                    self._safe_schedule(_retry_message, action_name="retry")
                else:
                    logger.error("❌ Maximum retry attempts reached. Acknowledging message.")
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
                logger.error("Not retrying message due to non-retryable exception.")
            finally:
                # We use the safe ack to avoid issues if connection is lost
                self._safe_schedule(_ack, action_name="ack")

        headers = properties.headers or {}
        futur = self.executor.submit(BlurryMessageProcessor.process, body, headers.get('x-retry-count', 0))
        futur.add_done_callback(_on_done)

    def start_listening(self):
        """Starts listening for messages on the configured queue with auto-reconnect."""
        thread_number = arg_provider_service.get_thread_number()

        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=thread_number)

        while True:
            try:
                self._connect()

                # Configure prefetch to optimize distribution across hosts and threads
                self.channel.basic_qos(prefetch_count=thread_number)  # 1 message per thread max

                self.channel.basic_consume(
                    queue=self.queue_name,
                    on_message_callback=self._message_callback,
                    auto_ack=False  # Manual acknowledgment to avoid duplication
                )

                logger.info(
                    f"Listening for messages on queue '{self.queue_name}' with {thread_number} workers per host")
                self.channel.start_consuming()
            except (AMQPConnectionError, ConnectionWrongStateError, ChannelClosedByBroker, StreamLostError) as e:
                logger.error(f"RabbitMQ connection lost or unusable: {e}. Reconnecting in 5 seconds...")
                # Try to close gracefully
                try:
                    if self.connection and getattr(self.connection, "is_open", False):
                        self.connection.close()
                except Exception:
                    pass
                time.sleep(5)
                continue  # retry loop
            except KeyboardInterrupt:
                self.stop_listening()
                break

    def stop_listening(self):
        """Closes the connection to RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        database_service.close_all_connections()
        logger.info("🔌 Connection to RabbitMQ closed.")
