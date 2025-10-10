import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv
from contextlib import contextmanager

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

import psycopg2
from dossierfacile_file_analysis.data.file_dto import FileDto
from dossierfacile_file_analysis.services.file_downloader.ovh_file_downloader import OVHFileDownloader
from dossierfacile_file_analysis.custom_logging.logging_config import logger as app_logger

# Support riche (optionnel)
try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn, SpinnerColumn
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

DATASET_SIZE = 500

# Répertoires dataset
EVAL_ROOT = Path(__file__).parent
DATASET_DIR = EVAL_ROOT / "dataset"
BLURRY_DIR = DATASET_DIR / "blurry"
NOT_BLURRY_DIR = DATASET_DIR / "not_blurry"
DOWNLOAD_STAGING_DIR = DATASET_DIR / "_staging_downloads"

ovh_file_downloader = OVHFileDownloader(str(DOWNLOAD_STAGING_DIR) + "/")


def _mask_sensitive(d: dict) -> dict:
    masked = dict(d)
    if "password" in masked and masked["password"] is not None:
        masked["password"] = "***"
    return masked


def _collect_dbt_explicit_env() -> tuple[dict | None, list[str]]:
    """Collect required DBT DB env vars. Returns (kwargs or None, missing_vars).
    Required: DBT_HOST, DBT_PORT, DBT_NAME, DBT_USER, DBT_PASSWORD
    Optional: DBT_SSLMODE, DBT_SSLROOTCERT
    """
    required_keys = ["DBT_HOST", "DBT_PORT", "DBT_NAME", "DBT_USER", "DBT_PASSWORD"]
    optional_keys = ["DBT_SSLMODE"]

    env = {k: os.getenv(k) for k in required_keys + optional_keys}
    missing = [k for k in required_keys if not env.get(k)]
    if missing:
        return None, missing

    # Build kwargs for psycopg2
    try:
        kwargs = {
            "host": env["DBT_HOST"],
            "port": int(env["DBT_PORT"]),
            "dbname": env["DBT_NAME"],
            "user": env["DBT_USER"],
            "password": env["DBT_PASSWORD"],
        }
    except ValueError:
        return None, ["DBT_PORT (must be an integer)"]

    if env.get("DBT_SSLMODE"):
        kwargs["sslmode"] = env["DBT_SSLMODE"]

    return kwargs, []


def _collect_df_explicit_env() -> tuple[dict | None, list[str]]:
    """Collect required DF DB env vars. Returns (kwargs or None, missing_vars).
    Required: DF_DB_HOST, DF_DB_PORT, DF_DB_NAME, DF_DB_USER, DF_DB_PASSWORD
    Optional: DF_DB_SSLMODE
    """
    required_keys = ["DF_DB_HOST", "DF_DB_PORT", "DF_DB_NAME", "DF_DB_USER", "DF_DB_PASSWORD"]
    optional_keys = ["DF_DB_SSLMODE"]

    env = {k: os.getenv(k) for k in required_keys + optional_keys}
    missing = [k for k in required_keys if not env.get(k)]
    if missing:
        return None, missing

    try:
        kwargs = {
            "host": env["DF_DB_HOST"],
            "port": int(env["DF_DB_PORT"]),
            "dbname": env["DF_DB_NAME"],
            "user": env["DF_DB_USER"],
            "password": env["DF_DB_PASSWORD"],
        }
    except ValueError:
        return None, ["DF_DB_PORT (must be an integer)"]

    if env.get("DF_DB_SSLMODE"):
        kwargs["sslmode"] = env["DF_DB_SSLMODE"]

    return kwargs, []


def _execute_dbt_sql(query: str) -> list[int]:
    """Execute the given SQL query and return the first column of all rows as a list[int].
    Loads evaluation/.env and uses explicit DBT_* env variables for connection.
    """
    conn_kwargs, missing = _collect_dbt_explicit_env()
    if missing:
        raise RuntimeError(
            "Missing DBT connection variables: " + ", ".join(
                missing) + ". Set them in evaluation/.env or your environment."
        )
    with psycopg2.connect(**conn_kwargs) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return [r[0] for r in rows]


def _execute_df_sql(query: str, params=None) -> list[int]:
    """Execute the given SQL query on DF DB and return the first column as list[int].
    Loads evaluation/.env and uses explicit DF_DB_* env variables for connection.
    """
    conn_kwargs, missing = _collect_df_explicit_env()
    if missing:
        raise RuntimeError(
            "Missing DF DB connection variables: " + ", ".join(
                missing) + ". Set them in evaluation/.env or your environment."
        )
    with psycopg2.connect(**conn_kwargs) as conn:
        with conn.cursor() as cur:
            if params is not None:
                cur.execute(query, params)
            else:
                cur.execute(query)
            rows = cur.fetchall()
            return [r[0] for r in rows]


def get_dbt_blurry_document() -> list[int]:
    """Return the list of document_id from dbt_prod.core_document_blurry_evaluations
    where document_denied_at is not null, ordered by blurry_rule_created_at desc.
    """
    query = (
        "select document_id from dbt_prod.core_document_blurry_evaluations "
        "where document_denied_at is not null "
        "order by blurry_rule_created_at desc limit 1000;"
    )
    return _execute_dbt_sql(query)


def get_dbt_not_blurry_document() -> list[int]:
    """Return the list of document_id from dbt_prod.core_document_blurry_evaluations
    where document_denied_at is null, ordered by blurry_rule_created_at desc.
    """
    query = (
        "select document_id from dbt_prod.core_document_blurry_evaluations "
        "where document_denied_at is null "
        "order by blurry_rule_created_at desc limit 1000;"
    )
    return _execute_dbt_sql(query)


def filter_document_ids_deleted(list_id: list[int]):
    """Fetch a subset of document IDs available in DF DB from the provided list_id.
    Uses a safe, parameterized query.
    """
    if not list_id:
        return []

    query = (
        "select id from document "
        "where id = ANY(%s) "
        "order by id desc limit %s;"
    )
    params = (list_id, DATASET_SIZE)
    return _execute_df_sql(query, params)


def get_file_dto_list(document_id_list: list[int]) -> list[FileDto]:
    """Return a list of FileDto for the given document IDs from the DF database.
    For each document_id, executes the query with a single parameter and maps the first row to FileDto.
    """
    if not document_id_list:
        return []

    # Load env for DF connection
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    conn_kwargs, missing = _collect_df_explicit_env()
    if missing:
        raise RuntimeError(
            "Missing DF DB connection variables: " + ", ".join(
                missing) + ". Set them in evaluation/.env or your environment."
        )

    query = (
        "SELECT "
        "    f.id as id, "
        "    f.document_id as document_id, "
        "    sf.path as path, "
        "    sf.content_type as content_type, "
        "    ek.encoded as encryption_key, "
        "    ek.version as encryption_key_version, "
        "    sf.provider as provider "
        "FROM document as d "
        "JOIN file as f on d.id = f.document_id "
        "JOIN storage_file as sf on f.storage_file_id = sf.id "
        "LEFT JOIN encryption_key as ek on sf.encryption_key_id = ek.id "
        "WHERE d.id = %s"
    )

    results: list[FileDto] = []
    with psycopg2.connect(**conn_kwargs) as conn:
        with conn.cursor() as cursor:
            for doc_id in document_id_list:
                cursor.execute(query, (doc_id,))
                file_data = cursor.fetchone()
                if file_data:
                    column_names = [desc[0] for desc in cursor.description]
                    file_data_dict = dict(zip(column_names, file_data))
                    results.append(FileDto(**file_data_dict))
    return results


def _ensure_dirs():
    BLURRY_DIR.mkdir(parents=True, exist_ok=True)
    NOT_BLURRY_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_staging_dir():
    """Remove the whole staging directory and its contents."""
    try:
        if DOWNLOAD_STAGING_DIR.exists():
            shutil.rmtree(DOWNLOAD_STAGING_DIR, ignore_errors=True)
    except Exception:
        # Best effort cleanup
        pass


def _get_safe_target_name(dto: FileDto, src_path: str) -> str:
    # Préserve l’extension source si possible
    ext = Path(src_path).suffix or ""
    try:
        base = dto.get_system_name()
    except Exception:
        base = f"file_{dto.id}"
    return f"{base}{ext}"


@contextmanager
def muted_app_logger():
    """Désactive temporairement le logger applicatif pour éviter d’interférer avec la barre de progression."""
    prev_disabled = getattr(app_logger, "disabled", False)
    app_logger.disabled = True
    try:
        yield
    finally:
        app_logger.disabled = prev_disabled


def download_file_to(dto: FileDto, target_dir: Path) -> Path | None:
    """Télécharge un fichier via le OVH downloader (OVH uniquement) puis le copie dans target_dir.
    Retourne le chemin de destination ou None si échec.
    """
    try:
        provider = (dto.provider or "").upper() if hasattr(dto, "provider") else ""
        if provider != "OVH":
            print(
                f"Skip download for id={getattr(dto, 'id', 'unknown')}: unsupported provider '{dto.provider}' (only OVH supported)")
            return None

        # Mute les logs applicatifs pendant le téléchargement pour préserver l’affichage de la barre
        with muted_app_logger():
            downloaded = ovh_file_downloader.download_file(dto)
        if not downloaded or not getattr(downloaded, "file_path", None):
            print(f"Download failed (no file) for id={getattr(dto, 'id', 'unknown')}")
            return None

        src = Path(downloaded.file_path)
        if not src.exists():
            print(f"Downloaded file missing on disk: {src}")
            return None

        # Copie vers la destination avec un nom stable
        target_dir.mkdir(parents=True, exist_ok=True)
        dst_name = _get_safe_target_name(dto, str(src))
        dst = target_dir / dst_name

        # Évite recopie si déjà présent (idempotent)
        if not dst.exists():
            shutil.copyfile(src, dst)
        # Supprime le fichier du staging pour éviter l'accumulation
        try:
            src.unlink(missing_ok=True)
        except Exception:
            pass
        return dst
    except Exception as e:
        print(f"Download error for id={getattr(dto, 'id', 'unknown')}: {e}")
        return None


def _render_progress_bar(completed: int, total: int, prefix: str = "Downloading") -> str:
    if total <= 0:
        return f"{prefix} 0% [--------------------------------------------------] 0/0"
    percent = int((completed / total) * 100)
    bar_len = 50
    filled = int(bar_len * completed / total)
    bar = "=" * filled + ">" + "-" * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
    return f"{prefix} {percent:3d}% [{bar}] {completed}/{total}"


def _update_progress_bar(completed: int, total: int, prefix: str = "Downloading"):
    line = _render_progress_bar(completed, total, prefix)
    # Écrit sur stderr pour éviter l'entrelacement avec les logs stdout
    sys.stderr.write("\r" + line)
    sys.stderr.flush()
    if completed >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def download_dataset_lists(blurry_file_dto_list: list[FileDto], not_blurry_dto_list: list[FileDto]):
    """Télécharge les fichiers des deux listes dans les dossiers du dataset."""
    _ensure_dirs()

    total = len(blurry_file_dto_list) + len(not_blurry_dto_list)
    processed = 0

    if RICH_AVAILABLE:
        # Barre de progression “fixée” avec rich qui se redessine proprement sous les logs
        progress = Progress(
            SpinnerColumn(),
            TextColumn("Downloading dataset"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=False,
            expand=True,
        )
        with progress:
            task_id = progress.add_task("download", total=total)

            ok_blurry = 0
            for dto in blurry_file_dto_list:
                if download_file_to(dto, BLURRY_DIR):
                    ok_blurry += 1
                processed += 1
                progress.update(task_id, advance=1)
            print(f"Downloaded {ok_blurry}/{len(blurry_file_dto_list)} files into {BLURRY_DIR}")

            ok_not = 0
            for dto in not_blurry_dto_list:
                if download_file_to(dto, NOT_BLURRY_DIR):
                    ok_not += 1
                processed += 1
                progress.update(task_id, advance=1)
            print(f"Downloaded {ok_not}/{len(not_blurry_dto_list)} files into {NOT_BLURRY_DIR}")
    else:
        # Fallback simple sur stdout
        _update_progress_bar(processed, total, prefix="Downloading dataset")

        ok_blurry = 0
        for dto in blurry_file_dto_list:
            if download_file_to(dto, BLURRY_DIR):
                ok_blurry += 1
            processed += 1
            _update_progress_bar(processed, total, prefix="Downloading dataset")
        print(f"Downloaded {ok_blurry}/{len(blurry_file_dto_list)} files into {BLURRY_DIR}")

        ok_not = 0
        for dto in not_blurry_dto_list:
            if download_file_to(dto, NOT_BLURRY_DIR):
                ok_not += 1
            processed += 1
            _update_progress_bar(processed, total, prefix="Downloading dataset")
        print(f"Downloaded {ok_not}/{len(not_blurry_dto_list)} files into {NOT_BLURRY_DIR}")


def main():
    try:
        print("Starting dataset creation...")
        print("Sep 1 : prepare data")
        true_blurry_document_ids = get_dbt_blurry_document()
        print(f"Fetched {len(true_blurry_document_ids)} document_id into true_blurry_document_ids")
        not_blurry_document_ids = get_dbt_not_blurry_document()
        print(f"Fetched {len(not_blurry_document_ids)} document_id into not_blurry_document_ids")

        final_blurry_document_ids = filter_document_ids_deleted(true_blurry_document_ids)
        print(f"Fetched {len(final_blurry_document_ids)} document_id into final_blurry_document_ids")
        final_not_blurry_document_ids = filter_document_ids_deleted(not_blurry_document_ids)
        print(f"Fetched {len(final_not_blurry_document_ids)} document_id into final_not_blurry_document_ids")

        # Récupère les FileDto correspondants
        blurry_file_dto_list = get_file_dto_list(final_blurry_document_ids[:DATASET_SIZE])
        not_blurry_file_dto_list = get_file_dto_list(final_not_blurry_document_ids[:DATASET_SIZE])
        print(f"Built {len(blurry_file_dto_list)} FileDto for blurry, {len(not_blurry_file_dto_list)} for not_blurry")

        # Télécharge et copie dans les dossiers du dataset
        download_dataset_lists(blurry_file_dto_list, not_blurry_file_dto_list)

        print("Finished")
    except Exception as e:
        print(f"Database/Download error: {e}")
        sys.exit(2)
    finally:
        # Nettoyage du répertoire de staging, succès ou échec
        _cleanup_staging_dir()


if __name__ == "__main__":
    main()
