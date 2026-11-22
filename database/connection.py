"""
Database connection and pool management using psycopg (v3) and pgvector.
Configured for Supabase PostgreSQL and standard PostgreSQL instances.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)


def get_database_url(conn_str: Optional[str] = None) -> str:
    """
    Retrieves and normalizes the database connection URL.
    Handles 'postgres://' -> 'postgresql://' conversion (common in Railway/Supabase).
    Ensures 'sslmode=require' is attached for remote Supabase poolers.
    """
    raw_url = conn_str or os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/odpc_compliance"
    )
    clean_url = raw_url.strip()

    # Normalize protocol from legacy postgres:// to postgresql://
    if clean_url.startswith("postgres://"):
        clean_url = "postgresql://" + clean_url[len("postgres://"):]

    # For remote cloud databases (e.g. Supabase), ensure SSL is required
    is_local = any(h in clean_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if not is_local and "sslmode=" not in clean_url:
        separator = "&" if "?" in clean_url else "?"
        clean_url = f"{clean_url}{separator}sslmode=require"

    return clean_url


def get_connection_kwargs() -> dict:
    """
    Returns standard connection parameters for psycopg (v3).
    Includes TCP keepalives and connect timeouts to keep Supabase Session
    Pooler connections healthy and prevent hung workers on Railway.
    """
    return {
        "row_factory": dict_row,
        "autocommit": False,
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }


def create_connection(conn_str: Optional[str] = None) -> psycopg.Connection:
    """
    Creates a new psycopg connection and registers the pgvector extension.
    
    Args:
        conn_str: Optional connection URI. If None, uses DATABASE_URL.
        
    Returns:
        psycopg.Connection: Configured connection with vector support registered.
    """
    target_url = get_database_url(conn_str)
    safe_host = target_url.split("@")[-1].split("?")[0] if "@" in target_url else "configured host"
    try:
        conn = psycopg.connect(target_url, **get_connection_kwargs())
        # Register pgvector type handlers with this connection
        try:
            register_vector(conn)
        except Exception as vec_err:
            logger.warning(
                "Could not register pgvector extension on connection. "
                "Ensure 'CREATE EXTENSION IF NOT EXISTS vector;' has run. Error: %s",
                vec_err
            )
        return conn
    except Exception as e:
        logger.error("Failed to connect to PostgreSQL database (%s): %s", safe_host, e)
        raise ConnectionError(
            f"Could not connect to PostgreSQL database at {safe_host}. "
            f"Please verify DATABASE_URL in your Railway / Supabase project settings. Details: {e}"
        ) from e


@contextmanager
def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """
    Context manager that yields an active database connection with autocommit=False.
    Commits on successful completion, rolls back on exception, and closes on exit.
    
    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM legal_chunks LIMIT 5")
                rows = cur.fetchall()
    """
    conn = create_connection()
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Database transaction rolled back due to error: %s", exc)
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """
    Tests database connectivity and checks for the vector extension.
    
    Returns:
        bool: True if connection is healthy.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS alive;")
                res = cur.fetchone()
                return bool(res and res["alive"] == 1)
    except Exception as e:
        logger.warning("Database connection test failed: %s", e)
        return False


def init_db(schema_path: str = "database/schema.sql") -> bool:
    """
    Executes the schema.sql file to set up extensions and tables.
    
    Args:
        schema_path: Filepath to the SQL schema definition.
        
    Returns:
        bool: True if schema creation succeeded.
    """
    if not os.path.exists(schema_path):
        logger.error("Schema file not found at: %s", schema_path)
        return False

    with open(schema_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    logger.info("Initializing database schema from %s...", schema_path)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_script)
    logger.info("Database schema initialized successfully.")
    return True
