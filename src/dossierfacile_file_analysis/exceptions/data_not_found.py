class DataNotFoundException(Exception):
    
    def __init__(self, file_id: int):
        super().__init__(f"Data not found for file_id: {file_id}")