class FileDto:
    def __init__(self, id, path, content_type, encryption_key, encryption_key_version, provider):
        self.id = id
        self.path = path
        self.content_type = content_type
        self.encryption_key = encryption_key
        self.encryption_key_version = encryption_key_version
        self.provider = provider

    def get_system_name(self):
        return self.path.replace("/", "_")
