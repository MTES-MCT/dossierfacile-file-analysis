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
            "message": record.getMessage(),
        }
        return json.dumps(log_record)