import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("PG_DB", "deepface_db")
DB_USER = os.getenv("PG_USER", "postgres")
DB_PASSWORD = os.getenv("PG_PASSWORD", "")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")


def connect(dbname="postgres"):
    return psycopg2.connect(
        dbname=dbname,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def main():
    print("[INFO] Connecting to PostgreSQL...")
    conn = connect()
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Create database if not exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    if not cur.fetchone():
        print(f"[INFO] Creating database: {DB_NAME}")
        cur.execute(f"CREATE DATABASE {DB_NAME};")
    else:
        print(f"[INFO] Database '{DB_NAME}' already exists")

    cur.close()
    conn.close()

    # Connect to target DB
    print("[INFO] Connecting to DeepFace database...")
    conn = connect(DB_NAME)
    cur = conn.cursor()

    # Enable pgvector
    print("[INFO] Enabling pgvector extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    # Create tables
    print("[INFO] Creating tables...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS identities (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS representations (
            id SERIAL PRIMARY KEY,
            identity_id INTEGER REFERENCES identities(id) ON DELETE CASCADE,
            embedding vector(512),
            image_path TEXT
        );
    """)

    # Create ANN index
    print("[INFO] Creating ANN index (HNSW)...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_rep_embedding
        ON representations
        USING hnsw (embedding vector_cosine_ops);
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("[SUCCESS] PostgreSQL DeepFace setup completed!")


if __name__ == "__main__":
    main()
