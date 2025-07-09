class InvalidMimeTypeException(Exception):

    def __init__(self, file_id: int):
        super().__init__(f"Invalid mime type for : {file_id}")
