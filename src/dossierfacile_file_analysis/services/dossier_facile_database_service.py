import os
import threading
import time
import json

import psycopg2

from dossierfacile_file_analysis.data.FileDto import FileDto
from dossierfacile_file_analysis.models.blurry_result import BlurryResult


class DossierFacileDatabaseService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            print("Initializing DatabaseService")
            db_config = {
                "dbname": os.getenv("DB_NAME"),
                "user": os.getenv("DB_USER"),
                "password": os.getenv("DB_PASSWORD"),
                "host": os.getenv("DB_HOST"),
                "port": os.getenv("DB_PORT")
            }
            self.__connection = psycopg2.connect(**db_config)
            self._initialized = True

    def get_file_by_id(self, file_id):
        start_time = time.time()
        print("Retrieving file by ID from the database")
        cursor = self.__connection.cursor()
        # Requête SQL pour récupérer les données du fichier
        query = "SELECT " \
                "f.id as id, " \
                "sf.path as path, " \
                "sf.content_type as content_type, " \
                "ek.encoded as encryption_key, " \
                "ek.version as encryption_key_version, " \
                "sf.provider as provider " \
                "FROM file as f " \
                "join storage_file as sf on f.storage_file_id = sf.id " \
                "join encryption_key as ek on sf.encryption_key_id = ek.id " \
                "WHERE f.id = %s"
        cursor.execute(query, (file_id,))

        # Récupérer les résultats
        file_data = cursor.fetchone()
        if file_data:
            # Convertir le tuple en dictionnaire
            column_names = [desc[0] for desc in cursor.description]
            file_data_dict = dict(zip(column_names, file_data))
            end_time = time.time()
            print(f"Database read take : {end_time - start_time:.2f} seconds")
            return FileDto(**file_data_dict)
        else:
            print(f"No file found with file_id {file_id}")
            return None

    def save_blurry_result(self, file_id: int, blurry_result: BlurryResult):
        cursor = self.__connection.cursor()
        query = (
            "INSERT INTO blurry_file_analysis (file_id, blurry_results, analysis_status) "
            "VALUES (%s, %s, %s)"
        )
        blurry_results_json = json.dumps(blurry_result.to_dict())
        cursor.execute(query, (file_id, blurry_results_json, "COMPLETED"))
        self.__connection.commit()
        cursor.close()

