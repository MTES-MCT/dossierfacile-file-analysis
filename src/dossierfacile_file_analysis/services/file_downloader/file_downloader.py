import os
from abc import ABC, abstractmethod
from hashlib import sha256, md5

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.data.file_dto import FileDto
from dossierfacile_file_analysis.exceptions.encryption_key_is_missing import EncryptionKeyIsMissingException
from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile


class FileDownloader(ABC):

    def __init__(self):
        self.local_file_path = os.getenv("LOCAL_FILE_PATH")

    @abstractmethod
    def download_file(self, file_dto: FileDto) -> DownloadedFile | None:
        pass

    @staticmethod
    def decrypt_file(encrypted_data, encryption_key, path, key_version):
        """
        Déchiffre les données chiffrées à l'aide de la clé de chiffrement et du chemin pour générer l'IV.
        """

        try:
            # Générer l'IV à partir du chemin
            if key_version == 2:
                iv = sha256(path.encode()).digest()
            else:
                iv = md5(path.encode()).digest()

            # Convertir la clé de chiffrement en bytes si nécessaire
            if isinstance(encryption_key, memoryview):
                encryption_key = encryption_key.tobytes()  # Convertir memoryview en bytes
            elif isinstance(encryption_key, str):
                encryption_key = bytes.fromhex(encryption_key)  # Convertir une chaîne hexadécimale en bytes

            # Séparer les données chiffrées et le tag d'authentification
            tag = encrypted_data[-16:]  # Le tag est stocké dans les 16 derniers octets
            ciphertext = encrypted_data[:-16]  # Le reste est le texte chiffré

            # Déchiffrement avec AES/GCM/NoPadding
            backend = default_backend()
            cipher = Cipher(algorithms.AES(encryption_key), modes.GCM(iv, tag), backend=backend)
            decryptor = cipher.decryptor()

            # Déchiffrer les données
            decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
            return decrypted_data
        except Exception as e:
            logger.error(f"An error occurred during decryption: {e}")
            raise

    @staticmethod
    def get_file_extension_from_content_type(content_type):
        # Map des types de contenu aux extensions de fichiers
        content_type_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "application/pdf": ".pdf",
            # Ajoutez d'autres types de contenu si nécessaire
        }

        return content_type_map.get(content_type, None)

    def decrypt_file_with_key(self, file_path, file_dto: FileDto):
        destination_path = f"{self.local_file_path}{file_dto.path}{self.get_file_extension_from_content_type(file_dto.content_type)}"
        try:
            # Lire le fichier chiffré
            with open(file_path, "rb") as encrypted_file:
                encrypted_data = encrypted_file.read()

            # Déchiffrer le fichier si une clé est fournie
            if file_dto.encryption_key:
                decrypted_data = self.decrypt_file(encrypted_data, file_dto.encryption_key, file_dto.path,
                                                   file_dto.encryption_key_version)
            else:
                raise EncryptionKeyIsMissingException(file_id=file_dto.id)

            # Écrire le fichier déchiffré
            with open(destination_path, "wb") as decrypted_file:
                decrypted_file.write(decrypted_data)

            return DownloadedFile(file_name=file_dto.path, file_path=destination_path, file_type=file_dto.content_type)
        except Exception as e:
            logger.error(f"An error occurred while downloading or decrypting the file: {e}")
            raise e
