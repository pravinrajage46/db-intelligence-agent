# analyzers/schema_analyzer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict

from connectors.db_connector import DBConnector


# ── Pydantic Models ───────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    data_type: str
    nullable: bool
    default: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: str = ""
    references_column: str = ""


class ForeignKey(BaseModel):
    constrained_columns: List[str]
    referred_table: str
    referred_columns: List[str]


class TableInfo(BaseModel):
    name: str
    row_count: int
    columns: List[ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[ForeignKey]
    index_count: int


class SchemaInfo(BaseModel):
    db_display_name: str
    total_tables: int
    total_columns: int
    total_rows: int
    tables: Dict[str, TableInfo]


# ── Analyzer ──────────────────────────────────────────────────────────────────

class SchemaAnalyzer:
    """Extracts and structures complete database schema metadata."""

    def __init__(self, connector: DBConnector):
        self.connector = connector

    def analyze(self) -> SchemaInfo:
        raw = self.connector.get_raw_schema()
        tables: Dict[str, TableInfo] = {}

        for table_name, meta in raw.items():
            # Build FK lookup: column_name → (referred_table, referred_column)
            fk_map: Dict[str, tuple] = {}
            fk_list: List[ForeignKey] = []

            for fk in meta["fks"]:
                for col, ref_col in zip(
                    fk.get("constrained_columns", []),
                    fk.get("referred_columns", []),
                ):
                    fk_map[col] = (fk.get("referred_table", ""), ref_col)

                fk_list.append(
                    ForeignKey(
                        constrained_columns=fk.get("constrained_columns", []),
                        referred_table=fk.get("referred_table", ""),
                        referred_columns=fk.get("referred_columns", []),
                    )
                )

            pk_cols = meta["pk"].get("constrained_columns", [])

            # Build column list
            columns: List[ColumnInfo] = []
            for col in meta["columns"]:
                col_name = col["name"]
                ref_table, ref_col = fk_map.get(col_name, ("", ""))
                columns.append(
                    ColumnInfo(
                        name=col_name,
                        data_type=str(col.get("type", "UNKNOWN")),
                        nullable=bool(col.get("nullable", True)),
                        default=str(col.get("default", "")) or None,
                        is_primary_key=col_name in pk_cols,
                        is_foreign_key=col_name in fk_map,
                        references_table=ref_table,
                        references_column=ref_col,
                    )
                )

            tables[table_name] = TableInfo(
                name=table_name,
                row_count=self.connector.get_row_count(table_name),
                columns=columns,
                primary_keys=pk_cols,
                foreign_keys=fk_list,
                index_count=len(meta.get("indexes", [])),
            )

        total_rows = sum(t.row_count for t in tables.values())
        total_cols = sum(len(t.columns) for t in tables.values())

        return SchemaInfo(
            db_display_name=self.connector.db_name,
            total_tables=len(tables),
            total_columns=total_cols,
            total_rows=total_rows,
            tables=tables,
        )