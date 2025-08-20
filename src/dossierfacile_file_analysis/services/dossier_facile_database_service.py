import os
import threading
import time
import json

import psycopg2
from psycopg2 import pool

from dossierfacile_file_analysis.custom_logging.logging_config import logger
from dossierfacile_file_analysis.data.file_dto import FileDto
from dossierfacile_file_analysis.models.blurry_result import BlurryResult
from dossierfacile_file_analysis.services.arg_provider_service import arg_provider_service


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
            logger.info("Initializing DatabaseService with connection pool")
            db_config = {
                "dbname": os.getenv("DB_NAME"),
                "user": os.getenv("DB_USER"),
                "password": os.getenv("DB_PASSWORD"),
                "host": os.getenv("DB_HOST"),
                "port": os.getenv("DB_PORT")
            }
            # Pool optimisé pour 4 threads + marge de sécurité
            # minconn=2 : Toujours 2 connexions prêtes
            # maxconn=8 : Permet pics de charge et retry logic
            thread_number = arg_provider_service.get_thread_number()
            self.__connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=thread_number,  # Connexions persistantes pour réactivité
                maxconn=(thread_number*2 + 2),  # Double des threads + marge pour retry/erreurs
                **db_config
            )
            self._initialized = True

    def _get_connection(self):
        """Obtenir une connexion du pool"""
        return self.__connection_pool.getconn()

    def _put_connection(self, conn):
        """Remettre une connexion dans le pool"""
        self.__connection_pool.putconn(conn)

    def get_file_by_id(self, file_id):
        start_time = time.time()
        logger.info(f"🔍 Retrieving file by ID from database for file_id: {file_id}")

        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

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
                    "left join encryption_key as ek on sf.encryption_key_id = ek.id " \
                    "WHERE f.id = %s"
            cursor.execute(query, (file_id,))

            # Récupérer les résultats
            file_data = cursor.fetchone()
            if file_data:
                # Convertir le tuple en dictionnaire
                column_names = [desc[0] for desc in cursor.description]
                file_data_dict = dict(zip(column_names, file_data))
                end_time = time.time()
                logger.debug(f"✅ Database read completed in {end_time - start_time:.2f} seconds for file_id: {file_id}")
                return FileDto(**file_data_dict)
            else:
                logger.error(f"❌ No file found with file_id {file_id}")
                return None

        except Exception as e:
            logger.error(f"❌ Error retrieving file {file_id}: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)

    def save_blurry_result(self, file_id: int, blurry_result: BlurryResult):
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = (
                "INSERT INTO blurry_file_analysis (file_id, blurry_results, analysis_status, data_file_id) "
                "VALUES (%s, %s, %s, %s)"
            )
            blurry_results_json = json.dumps(blurry_result.to_dict())
            cursor.execute(query, (file_id, blurry_results_json, "COMPLETED", file_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save blurry result for file_id {file_id}: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)

    def save_failed_analysis(self, file_id: int):
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = (
                "INSERT INTO blurry_file_analysis (file_id, analysis_status, data_file_id) "
                "VALUES (%s, %s, %s)"
            )
            cursor.execute(query, (file_id, "FAILED", file_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save failed analysis for file_id {file_id}: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)

    def close_all_connections(self):
        """Fermer toutes les connexions du pool (à appeler à l'arrêt de l'application)"""
        if hasattr(self, '__connection_pool'):
            self.__connection_pool.closeall()

database_service = DossierFacileDatabaseService()