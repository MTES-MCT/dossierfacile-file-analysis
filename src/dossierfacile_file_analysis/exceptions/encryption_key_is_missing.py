class EncryptionKeyIsMissingException(Exception):

    def __init__(self, file_id: int):
        super().__init__(f"Encryption key is missing for : {file_id}")
