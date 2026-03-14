# connectors/db_connector.py
from __future__ import annotations

import pandas as pd
from typing import Dict, List
from sqlalchemy import create_engine, inspect, text, MetaData
from sqlalchemy.engine import Engine

from config import config


class DBConnector:
    """
    Universal database connector.
    Supports SQLite, PostgreSQL, MySQL, MSSQL via SQLAlchemy URL strings.
    """

    def __init__(self, db_url: str = None):
        self.db_url = db_url or config.DB_URL
        self.engine: Engine = create_engine(self.db_url)
        self.inspector = inspect(self.engine)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    # ── Connection ────────────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        """Verify the database is reachable."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"[DBConnector] Connection failed: {e}")
            return False

    # ── Schema Introspection ──────────────────────────────────────────────────

    def get_table_names(self) -> List[str]:
        return self.inspector.get_table_names()

    def get_raw_schema(self) -> Dict:
        """Return full schema metadata for all tables."""
        schema = {}
        for table_name in self.get_table_names():
            schema[table_name] = {
                "columns": self.inspector.get_columns(table_name),
                "pk": self.inspector.get_pk_constraint(table_name),
                "fks": self.inspector.get_foreign_keys(table_name),
                "indexes": self.inspector.get_indexes(table_name),
                "unique_constraints": self.inspector.get_unique_constraints(table_name),
                "check_constraints": self._safe_get_checks(table_name),
            }
        return schema

    def _safe_get_checks(self, table_name: str) -> List:
        try:
            return self.inspector.get_check_constraints(table_name)
        except Exception:
            return []

    # ── Data Sampling ─────────────────────────────────────────────────────────

    def get_row_count(self, table_name: str) -> int:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                return result.scalar() or 0
        except Exception:
            return 0

    def sample_table(self, table_name: str, n: int = None) -> pd.DataFrame:
        """Load a random sample of rows from a table."""
        n = n or config.SAMPLE_SIZE
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(f'SELECT * FROM "{table_name}" LIMIT :n'),
                    conn,
                    params={"n": n},
                )
            return df
        except Exception as e:
            print(f"[DBConnector] Could not sample {table_name}: {e}")
            return pd.DataFrame()

    def get_all_samples(self, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Load samples from every table."""
        tables = self.get_table_names()
        samples = {}
        for i, table in enumerate(tables):
            samples[table] = self.sample_table(table)
            if progress_callback:
                progress_callback(i + 1, len(tables), table)
        return samples

    # ── Metadata helpers ──────────────────────────────────────────────────────

    @property
    def db_name(self) -> str:
        """Return a safe display name for the database."""
        # Mask credentials from URL
        url = self.db_url
        if "@" in url:
            url = url.split("@", 1)[-1]
        return url