import argparse
import os


class ArgProviderService:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--thread-number", type=str, help="Nombre de threads")
        args = parser.parse_args()

        if args.thread_number:
            os.environ["THREAD_NUMBER"] = args.thread_number
        else:
            os.environ["THREAD_NUMBER"] = "4"

    def get_thread_number(self) -> int:
        """Retourne le nombre de threads configuré."""
        thread_number = os.getenv("THREAD_NUMBER")
        if thread_number is None:
            raise ValueError("THREAD_NUMBER environment variable is not set.")
        return int(thread_number)

arg_provider_service = ArgProviderService()