import time
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    """
    Prints a message to the console every 10 seconds.
    """
    project_name = os.getenv("PROJECT_NAME")
    while True:
        print(f"hello from {project_name}")
        time.sleep(10)

if __name__ == "__main__":
    main()
