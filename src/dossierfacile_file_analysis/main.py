import os
from dotenv import load_dotenv
from dossierfacile_file_analysis.services.amqp_service import AmqpService
import elasticapm

load_dotenv()

def main():
    """
    Initializes and starts the AMQP service to listen for messages.
    """
    elasticapm.Client({
        'SERVICE_NAME': os.getenv("ELASTIC_APM_SERVICE_NAME"),
        'SERVER_URL': os.getenv("ELASTIC_APM_SERVER_URL"),
        'SECRET_TOKEN': os.getenv("f3a1c2d4e5b67890abcdef1234567890"),  # si pas de token
        'ENVIRONMENT': os.getenv("ELASTIC_APM_ENVIRONMENT"),
    })
    client = elasticapm.get_client()
    if client:
        print("Elastic APM client configured successfully.")
    else:
        print("Could not configure Elastic APM client. Check your environment variables.")

    project_name = os.getenv("PROJECT_NAME")
    print(f"🚀 Starting {project_name}")
    
    amqp_service = AmqpService()
    amqp_service.start_listening()

if __name__ == "__main__":
    main()
