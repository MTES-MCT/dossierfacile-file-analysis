import json
import logging
import os


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger_name": record.name,
            "env": os.getenv("ELASTIC_APM_ENVIRONMENT"),
            "log_thread_id": record.thread,
            "log_thread_name": record.threadName,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)