# analyzers/quality_profiler.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analyzers.schema_analyzer import SchemaInfo
from config import config


class QualityProfiler:
    """
    Profiles every column in every table for:
    - Null rates and completeness
    - Value distributions (min, max, mean, std)
    - Timestamp freshness
    - Cardinality and uniqueness
    - Top-N value frequencies
    """

    def __init__(self, schema: SchemaInfo, samples: Dict[str, pd.DataFrame]):
        self.schema = schema
        self.samples = samples

    # ── Public API ────────────────────────────────────────────────────────────

    def profile_all(self) -> Dict[str, Dict]:
        """Profile every table. Returns nested dict of quality metrics."""
        results = {}
        for table_name, df in self.samples.items():
            if df is None or df.empty:
                results[table_name] = self._empty_profile(table_name)
                continue
            results[table_name] = self._profile_table(table_name, df)
        return results

    # ── Table-level ───────────────────────────────────────────────────────────

    def _profile_table(self, table_name: str, df: pd.DataFrame) -> Dict:
        column_profiles = {}
        for col_name in df.columns:
            column_profiles[col_name] = self._profile_column(df[col_name])

        completeness_values = [
            v["completeness"] for v in column_profiles.values()
        ]
        overall_completeness = round(float(np.mean(completeness_values)), 4) if completeness_values else 0.0

        # Find worst columns
        worst = sorted(
            column_profiles.items(), key=lambda x: x[1]["null_rate"], reverse=True
        )[:3]

        return {
            "table_name": table_name,
            "sampled_rows": len(df),
            "column_count": len(df.columns),
            "overall_completeness": overall_completeness,
            "worst_columns": [w[0] for w in worst if w[1]["null_rate"] > 0],
            "columns": column_profiles,
        }

    # ── Column-level ──────────────────────────────────────────────────────────

    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        total = len(series)
        null_count = int(series.isna().sum())
        null_rate = round(null_count / total, 6) if total > 0 else 0.0
        completeness = round(1.0 - null_rate, 6)
        unique_count = int(series.nunique(dropna=True))
        unique_rate = round(unique_count / max(total - null_count, 1), 6)

        profile = {
            "dtype": str(series.dtype),
            "total_count": total,
            "null_count": null_count,
            "null_rate": null_rate,
            "completeness": completeness,
            "unique_count": unique_count,
            "unique_rate": unique_rate,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "std_value": None,
            "top_values": [],
            "freshness_days": None,
            "is_fresh": None,
            "column_type_inferred": "unknown",
        }

        clean = series.dropna()
        if clean.empty:
            return profile

        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            profile.update(self._numeric_stats(clean))
            profile["column_type_inferred"] = "numeric"

        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update(self._datetime_stats(clean))
            profile["column_type_inferred"] = "datetime"

        # Object/string columns — try to detect datetime strings
        elif series.dtype == object:
            if self._looks_like_datetime(clean):
                try:
                    parsed = pd.to_datetime(clean, errors="coerce").dropna()
                    if not parsed.empty:
                        profile.update(self._datetime_stats(parsed))
                        profile["column_type_inferred"] = "datetime_string"
                except Exception:
                    profile.update(self._categorical_stats(clean))
                    profile["column_type_inferred"] = "categorical"
            else:
                profile.update(self._categorical_stats(clean))
                profile["column_type_inferred"] = "categorical"

        return profile

    def _numeric_stats(self, s: pd.Series) -> Dict:
        return {
            "min_value": self._safe_float(s.min()),
            "max_value": self._safe_float(s.max()),
            "mean_value": self._safe_float(s.mean()),
            "std_value": self._safe_float(s.std()),
        }

    def _datetime_stats(self, s: pd.Series) -> Dict:
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce").dropna()
        if s.empty:
            return {}

        max_ts = s.max()
        try:
            max_naive = max_ts.to_pydatetime().replace(tzinfo=None)
        except Exception:
            max_naive = datetime.now()

        freshness_days = (datetime.now() - max_naive).days

        return {
            "min_value": str(s.min()),
            "max_value": str(s.max()),
            "freshness_days": freshness_days,
            "is_fresh": freshness_days <= config.FRESHNESS_THRESHOLD_DAYS,
        }

    def _categorical_stats(self, s: pd.Series) -> Dict:
        vc = s.astype(str).str[:100].value_counts().head(10)
        return {
            "top_values": [
                {"value": str(k), "count": int(v)} for k, v in vc.items()
            ]
        }

    @staticmethod
    def _looks_like_datetime(s: pd.Series) -> bool:
        """Heuristic check: does this string column look like dates?"""
        sample = s.dropna().astype(str).head(20)
        patterns = ["-", "/", "T", ":"]
        matches = sum(
            1 for val in sample
            if any(p in val for p in patterns) and len(val) >= 8
        )
        return matches > len(sample) * 0.6

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        try:
            f = float(val)
            return round(f, 6) if np.isfinite(f) else None
        except Exception:
            return None

    @staticmethod
    def _empty_profile(table_name: str) -> Dict:
        return {
            "table_name": table_name,
            "sampled_rows": 0,
            "column_count": 0,
            "overall_completeness": 0.0,
            "worst_columns": [],
            "columns": {},
        }