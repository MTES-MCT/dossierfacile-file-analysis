class DuplicateKeyException(Exception):
    """Exception raised when a duplicate key constraint violation occurs in the database."""
    
    def __init__(self, message: str, file_id: int = None):
        super().__init__(message)
        self.message = message
        self.file_id = file_id

    def __str__(self):
        return f"DuplicateKeyException: {self.message}"
