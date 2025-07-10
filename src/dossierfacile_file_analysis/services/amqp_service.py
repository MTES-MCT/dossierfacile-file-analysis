import os
import time
from concurrent.futures.thread import ThreadPoolExecutor

import pika
from pika.exceptions import AMQPConnectionError

from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException
from dossierfacile_file_analysis.services.blurry_message_processor import BlurryMessageProcessor


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
                print("✅ Successfully connected to RabbitMQ")
                return
            except AMQPConnectionError as e:
                print(f"❌ Failed to connect to RabbitMQ: {e}. Retrying in 10 seconds...")
                time.sleep(10)

    def _message_callback(self, channel, method_frame, properties, body):
        delivery_tag = method_frame.delivery_tag
        print(
            f"📥 Received message from queue '{self.queue_name}': {body.decode()}; delivery_tag={delivery_tag}; header_frame={properties}")

        def _ack():
            channel.basic_ack(delivery_tag=delivery_tag)

        def _retry_message():
            retry_delay_ms = 3000  # 3 secondes
            retry_queue = f"{self.queue_name}_retry"
            # Déclare la file de retry avec TTL et DLX
            channel.queue_declare(
                queue=retry_queue,
                durable=True,
                arguments={
                    "x-message-ttl": retry_delay_ms,
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": self.queue_name
                }
            )
            new_properties = pika.BasicProperties(
                headers={"x-retry-count": properties.headers.get('x-retry-count', 0) + 1}
            )
            channel.basic_publish(
                exchange='',
                routing_key=retry_queue,
                body=body,
                properties=new_properties
            )

        def _on_done(future):
            try:
                future.result()
            except RetryableException as e:
                print(f"⚠️ Error processing message: {e}")
                retry_count = properties.headers.get('x-retry-count', 0)
                if retry_count < 3:
                    print(f"🔄 Retrying message (attempt {retry_count + 1})")
                    self.connection.add_callback_threadsafe(_retry_message)
                else:
                    print("❌ Maximum retry attempts reached. Acknowledging message.")
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                print(f"Not retrying message due to non-retryable exception.")
            finally:
                self.connection.add_callback_threadsafe(_ack)

        futur = self.executor.submit(BlurryMessageProcessor.process, body, properties.headers.get('x-retry-count', 0))
        futur.add_done_callback(_on_done)

    def start_listening(self):
        """Starts listening for messages on the configured queue."""
        self._connect()
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self._message_callback
        )

        print(f"👂 Listening for messages on queue '{self.queue_name}'. To exit press CTRL+C")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.stop_listening()

    def stop_listening(self):
        """Closes the connection to RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            print("🔌 Connection to RabbitMQ closed.")
