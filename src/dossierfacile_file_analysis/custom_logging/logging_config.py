import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener


import elasticapm
from elasticapm.handlers.logging import LoggingHandler

from dossierfacile_file_analysis.custom_logging.json_formatter import JsonFormatter
from dossierfacile_file_analysis.custom_logging.tcp_handler import TCPLogHandler

# Contrôle d’activation APM: ELASTIC_APM_ENABLED prioritaire
_truthy = {"1", "true", "yes", "on"}
_falsy = {"0", "false", "no", "off"}
raw_enabled = os.getenv("ELASTIC_APM_ENABLED")
df_enable = os.getenv("DF_ENABLE_APM")
apm_url_present = bool(os.getenv("ELASTIC_APM_SERVER_URL"))

if raw_enabled is not None:
    val = raw_enabled.strip().lower()
    if val in _truthy:
        APM_ENABLED = True
    elif val in _falsy or val == "":
        APM_ENABLED = False
    else:
        APM_ENABLED = False
else:
    # Pas de consigne explicite: activer si DF_ENABLE_APM truthy ou si SERVER_URL fourni
    APM_ENABLED = (str(df_enable).strip().lower() in _truthy) or apm_url_present

client = None
if APM_ENABLED:
    try:
        elasticapm.Client({
            'service_name': os.getenv("ELASTIC_APM_SERVICE_NAME"),
            'server_url': os.getenv("ELASTIC_APM_SERVER_URL"),
            'secret_token': os.getenv("ELASTIC_APM_SECRET_TOKEN"),
            'environment': os.getenv("ELASTIC_APM_ENVIRONMENT"),
            'enabled': True,
        })
        client = elasticapm.get_client()
    except Exception:
        client = None

# --- Setup unique logger ---
logger = logging.getLogger("FileAnalysisLogger")
log_queue = queue.Queue(-1)  # -1 = taille illimitée

if not logger.hasHandlers():  # Assure qu’on le configure une seule fois
    logger.setLevel(logging.INFO)

    handlers = []

    # APM Handler pour ERROR+ (uniquement si client valide)
    if client is not None:
        apm_handler = LoggingHandler(client=client)
        apm_handler.setLevel(logging.ERROR)
        handlers.append(apm_handler)

    tcp_handler = TCPLogHandler(host=os.getenv("LOGSTASH_HOST"), port=os.getenv("LOGSTASH_PORT"))
    tcp_handler.setLevel(logging.INFO)

    json_formatter = JsonFormatter()
    if client is not None:
        apm_handler.setFormatter(json_formatter)
    tcp_handler.setFormatter(json_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s"))

    queue_handler = QueueHandler(log_queue)

    # Attacher les handlers
    logger.addHandler(queue_handler)
    if client is not None:
        logger.addHandler(apm_handler)
    logger.addHandler(console_handler)

    # Le listener déporte l’écriture TCP hors du thread principal
    listener = QueueListener(log_queue, tcp_handler, respect_handler_level=True)
    listener.start()
