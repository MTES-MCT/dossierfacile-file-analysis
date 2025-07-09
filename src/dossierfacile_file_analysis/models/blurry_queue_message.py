class BlurryQueueMessage:
    """
    Represents a message in the queue.
    """

    def __init__(self, file_id: str):
        self.file_id = file_id

    def __repr__(self):
        return f"BlurryQueueMessage(file_id={self.file_id})"
    
    @staticmethod
    def from_dict(data: dict) -> 'BlurryQueueMessage':
        """
        Creates a BlurryQueueMessage instance from a dictionary.
        """
        return BlurryQueueMessage(file_id=data.get("fileId"))

    def to_dict(self):
        """
        Converts the BlurryQueueMessage instance to a dictionary.
        """
        return {
            "fileId": self.file_id
        }