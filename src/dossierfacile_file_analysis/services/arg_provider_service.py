import argparse
import os


class ArgProviderService:
    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--thread-number", type=str, help="Nombre de threads")
        try:
            # parse_known_args prevent crash with pytest when he pass unexpected args
            args, _ = parser.parse_known_args(argv)
        except SystemExit:
            args = argparse.Namespace(thread_number=None)

        # Use command line argument if provided, otherwise fallback to environment variable or default to "4"
        thread_number = args.thread_number or os.getenv("THREAD_NUMBER") or "4"
        os.environ["THREAD_NUMBER"] = str(thread_number)

    def get_thread_number(self) -> int:
        """Return the number of thread to use."""
        thread_number = os.getenv("THREAD_NUMBER")
        if thread_number is None:
            raise ValueError("THREAD_NUMBER environment variable is not set.")
        return int(thread_number)


arg_provider_service = ArgProviderService()