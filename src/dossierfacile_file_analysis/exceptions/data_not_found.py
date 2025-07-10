from dossierfacile_file_analysis.exceptions.retryable_exception import RetryableException


class DataNotFoundException(RetryableException):
    
    def __init__(self, file_id: int):
        super().__init__(f"Data not found for file_id: {file_id}")