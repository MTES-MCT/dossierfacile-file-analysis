class RetryableException(Exception):
    """Base class for exceptions that can be retried."""
    
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"RetryableException: {self.message}"