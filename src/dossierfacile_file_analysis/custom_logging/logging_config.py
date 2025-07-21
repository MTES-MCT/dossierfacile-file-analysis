import logging
import os

import elasticapm

from elasticapm.handlers.logging import LoggingHandler

from dossierfacile_file_analysis.custom_logging.json_formatter import JsonFormatter
from dossierfacile_file_analysis.custom_logging.tcp_handler import TCPLogHandler

print("Initializing logging configuration...")

elasticapm.Client({
    'SERVICE_NAME': os.getenv("ELASTIC_APM_SERVICE_NAME"),
    'SERVER_URL': os.getenv("ELASTIC_APM_SERVER_URL"),
    'SECRET_TOKEN': os.getenv("ELASTIC_APM_SECRET_TOKEN"),  # si pas de token
    'ENVIRONMENT': os.getenv("ELASTIC_APM_ENVIRONMENT"),
})

client = elasticapm.get_client()

if client:
    print("Elastic APM client configured successfully.")
else:
    print("Could not configure Elastic APM client. Check your environment variables.")

# --- Setup unique logger ---
logger = logging.getLogger("FileAnalysisLogger")
if not logger.hasHandlers():  # Assure qu’on le configure une seule fois
    logger.setLevel(logging.INFO)

    # APM Handler pour ERROR+
    apm_handler = LoggingHandler(client=client)
    apm_handler.setLevel(logging.ERROR)

    # TCP Logstash pour INFO+
    tcp_handler = TCPLogHandler(host=os.getenv("LOGSTASH_HOST"), port=os.getenv("LOGSTASH_PORT"))
    tcp_handler.setLevel(logging.INFO)

    json_formatter = JsonFormatter()
    apm_handler.setFormatter(json_formatter)
    tcp_handler.setFormatter(json_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)

    logger.addHandler(apm_handler)
    logger.addHandler(tcp_handler)
    logger.addHandler(console_handler)
