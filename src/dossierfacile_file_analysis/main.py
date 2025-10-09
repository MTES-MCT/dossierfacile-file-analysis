import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from dossierfacile_file_analysis.services.amqp_service import AmqpService
from dossierfacile_file_analysis.custom_logging.logging_config import logger

def main():

    project_name = os.getenv("PROJECT_NAME")
    logger.info(f"🚀 Starting {project_name}")
    
    amqp_service = AmqpService()
    amqp_service.start_listening()

if __name__ == "__main__":
    main()
