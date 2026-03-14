# generators/ai_narrator.py
from __future__ import annotations

import json
from typing import Dict, List

from config import config


class AINarrator:
    """
    Uses Claude (claude-sonnet-4-20250514) to generate:
    - Executive schema summaries
    - Per-table business purpose descriptions
    - Column-level plain-English annotations
    - Causal relationship business narratives

    Falls back gracefully to template-based text if no API key is set.
    """

    def __init__(self):
        self.client = None
        self.enabled = bool(config.ANTHROPIC_API_KEY)

        if self.enabled:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            except ImportError:
                print("[AINarrator] anthropic package not installed. Falling back.")
                self.enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_schema_summary(
        self, schema_dict: Dict, er_summary: Dict
    ) -> str:
        if not self.enabled:
            return self._fallback_schema_summary(schema_dict)

        prompt = f"""You are a senior data architect reviewing a database for the first time.

Analyze this database schema and write a concise executive summary (3-4 sentences) that explains:
1. What business domain this database serves
2. The most important tables and their role
3. Any notable design patterns, strengths, or data quality concerns

Schema overview:
- Total tables: {schema_dict.get('total_tables')}
- Total columns: {schema_dict.get('total_columns')}
- Total rows: {schema_dict.get('total_rows', 'N/A'):,}

Table names and sizes:
{json.dumps({name: {'rows': t['row_count'], 'columns': len(t['columns'])} for name, t in schema_dict.get('tables', {}).items()}, indent=2)}

Relationships (edges):
{json.dumps(er_summary.get('edges', [])[:15], indent=2)}

Write for a business audience. Be direct. No bullet points."""

        return self._call_claude(prompt, max_tokens=512)

    def generate_table_summaries(
        self, schema_dict: Dict, quality_metrics: Dict
    ) -> Dict[str, str]:
        summaries = {}
        for table_name, table_data in schema_dict.get("tables", {}).items():
            summaries[table_name] = self._generate_table_summary(
                table_name, table_data, quality_metrics.get(table_name, {})
            )
        return summaries

    def generate_causal_narrative(self, relationships: list) -> str:
        if not self.enabled or not relationships:
            return (
                "Causal analysis identified column influence patterns. "
                "See the relationships table below for details. "
                "Enable AI narration (set ANTHROPIC_API_KEY) for business interpretation."
            )

        rel_data = [
            r.to_dict() if hasattr(r, "to_dict") else r
            for r in relationships[:12]
        ]

        prompt = f"""You are a data scientist explaining analytical findings to a business team.

The following causal relationships were discovered automatically in a database.
Write a 3-4 sentence business narrative explaining:
1. What the most significant relationships mean for the business
2. Which column influences are most actionable
3. Any data quality or design implications

Discovered relationships (ranked by strength):
{json.dumps(rel_data, indent=2)}

Write in plain English. No technical jargon. No bullet points."""

        return self._call_claude(prompt, max_tokens=400)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_table_summary(
        self, table_name: str, table_data: Dict, quality: Dict
    ) -> str:
        if not self.enabled:
            return self._fallback_table_summary(table_name, table_data)

        col_list = [
            {"name": c["name"], "type": c["data_type"]}
            for c in table_data.get("columns", [])[:20]
        ]

        prompt = f"""Describe the '{table_name}' database table in 2 sentences for a business user.
Then write a one-line description for each column listed below.

Table: {table_name}
Rows: {table_data.get('row_count', 'N/A'):,}
Overall data completeness: {quality.get('overall_completeness', 'N/A')}
Columns: {json.dumps(col_list, indent=2)}

Respond in this exact format:
BUSINESS PURPOSE: <2 sentences>
COLUMNS:
- column_name: <one-line description>"""

        return self._call_claude(prompt, max_tokens=600)

    def _call_claude(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            message = self.client.messages.create(
                model=config.AI_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"[AI narration unavailable: {e}]"

    # ── Fallbacks (no API key) ────────────────────────────────────────────────

    @staticmethod
    def _fallback_schema_summary(schema_dict: Dict) -> str:
        n = schema_dict.get("total_tables", 0)
        c = schema_dict.get("total_columns", 0)
        r = schema_dict.get("total_rows", 0)
        return (
            f"This database contains {n} tables, {c} columns, and approximately "
            f"{r:,} total rows. Set ANTHROPIC_API_KEY in your .env file to enable "
            f"AI-generated business context summaries."
        )

    @staticmethod
    def _fallback_table_summary(table_name: str, table_data: Dict) -> str:
        n_cols = len(table_data.get("columns", []))
        rows = table_data.get("row_count", 0)
        return (
            f"BUSINESS PURPOSE: The '{table_name}' table contains {rows:,} records "
            f"across {n_cols} columns. Enable AI narration for business context.\n"
            f"COLUMNS:\n"
            + "\n".join(
                f"- {c['name']}: No description available."
                for c in table_data.get("columns", [])
            )
        )