"""
local_ai.py  –  Offline AI Engine (No API Key Required)

Uses rule-based Natural Language Generation (NLG) and data-driven
heuristics to produce intelligent summaries and chatbot responses
by analysing the actual database metrics passed in prompts.
"""

import re
import json

# ── Business domain detector ──────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "E-Commerce":           ["order", "product", "cart", "payment", "customer",
                             "invoice", "item", "sale", "purchase", "checkout", "price", "discount"],
    "HR / Payroll":         ["employee", "salary", "department", "payroll",
                             "attendance", "leave", "staff", "hire", "position", "role"],
    "Healthcare":           ["patient", "doctor", "appointment", "prescription",
                             "diagnosis", "hospital", "medical", "clinic", "drug", "treatment"],
    "Finance / Banking":    ["account", "transaction", "balance", "ledger",
                             "budget", "expense", "revenue", "loan", "credit", "debit"],
    "Education":            ["student", "course", "grade", "teacher", "enrollment",
                             "class", "assignment", "exam", "faculty", "curriculum"],
    "Inventory / Supply":   ["warehouse", "stock", "supplier", "category",
                             "inventory", "shipment", "sku", "bin", "reorder"],
    "Social / Community":   ["user", "post", "comment", "like", "follow",
                             "message", "profile", "friend", "feed", "notification"],
    "Logistics / Fleet":    ["shipment", "delivery", "route", "driver",
                             "vehicle", "tracking", "dispatch", "fleet", "cargo"],
}


def _detect_domain(table_names: list) -> str:
    text = " ".join(table_names).lower()
    scores = {d: sum(1 for k in kws if k in text) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Business"


def _safe_json(text: str):
    """Try to parse JSON from a string; return None on failure."""
    try:
        return json.loads(text)
    except Exception:
        return None


# ── Table-purpose classifier ──────────────────────────────────────────────
_PURPOSE_MAP = {
    ("user", "customer", "person", "people", "employee", "staff", "member", "client"): "people / entity",
    ("order", "transaction", "sale", "invoice", "payment", "purchase", "booking"):     "transactional",
    ("product", "item", "catalog", "category", "sku", "inventory", "stock"):           "product catalog",
    ("log", "audit", "history", "event", "activity", "trail"):                         "audit / activity log",
    ("config", "setting", "param", "lookup", "ref", "type", "code"):                   "configuration / lookup",
    ("report", "summary", "stat", "metric", "analytic", "kpi"):                        "reporting / analytics",
    ("address", "contact", "location", "geo"):                                          "geographic / contact",
}

def _classify_table(name: str) -> str:
    nl = name.lower()
    for keys, label in _PURPOSE_MAP.items():
        if any(k in nl for k in keys):
            return label
    return "data"


# ═════════════════════════════════════════════════════════════════════════════
#  LocalAI  –  main class
# ═════════════════════════════════════════════════════════════════════════════
class LocalAI:
    """
    Fully offline AI engine.  No API key, no internet, no model download.
    Generates professional insights by parsing the structured prompts that
    dashboard.py  sends to the  AI.ask()  method.
    """

    # ── Public interface ───────────────────────────────────────────────────
    def ask(self, prompt: str, max_tokens: int = 1024) -> str:
        p = prompt.lower()

        # Route by detecting which dashboard prompt-template this is
        if ("senior data architect" in p
                or ("tables:" in p and "relationships:" in p and "fks" in p)):
            return self._schema_summary(prompt)

        if ("3 sentences on business meaning" in p
                or ("method" in p and "strength" in p and "insight" in p
                    and p.startswith("["))):
            return self._causal_summary(prompt)

        if ("describe '" in p and "rows)" in p and "columns:" in p):
            return self._table_description(prompt)

        if (("expert analyst" in p or "use only this data" in p.replace("\n", " "))
                and "\nq:" in p):
            return self._chatbot(prompt)

        return "✅ Analysis complete. Explore the tabs above for detailed insights."

    # ── Schema overview ────────────────────────────────────────────────────
    def _schema_summary(self, prompt: str) -> str:
        tables = {}
        m = re.search(r'Tables:\s*(\{[^}]*\}|\{.*?\})', prompt, re.DOTALL)
        if m:
            tables = _safe_json(m.group(1)) or {}

        fk_m = re.search(r'Relationships:\s*(\d+)\s*FK', prompt)
        fk_count = int(fk_m.group(1)) if fk_m else 0

        n = len(tables)
        total_rows = sum(t.get("rows", 0) for t in tables.values())
        domain = _detect_domain(list(tables.keys()))

        largest_name = max(tables, key=lambda x: tables[x].get("rows", 0), default=None)

        if n == 0:
            return "Database analysis complete. No tables were detected."

        complexity = (
            "highly interconnected" if fk_count > n
            else "moderately connected" if fk_count > 0
            else "flat (no detected FK constraints)"
        )

        parts = [
            f"This is a **{domain}** database containing {n} table{'s' if n != 1 else ''} "
            f"with approximately {total_rows:,} total records.",
        ]
        if largest_name:
            parts.append(
                f"The central entity appears to be **{largest_name}** "
                f"({tables[largest_name].get('rows', 0):,} rows), interconnected via "
                f"{fk_count} foreign key relationship{'s' if fk_count != 1 else ''}."
            )
        parts.append(
            f"The schema is {complexity}, consistent with "
            f"{'a normalized relational design' if fk_count > 0 else 'denormalized or independent table layout'}."
        )
        return "  ".join(parts)

    # ── Causal summary ─────────────────────────────────────────────────────
    def _causal_summary(self, prompt: str) -> str:
        causal = []
        m = re.search(r'(\[.*\])', prompt, re.DOTALL)
        if m:
            causal = _safe_json(m.group(1)) or []

        if not causal:
            return (
                "No statistically significant causal relationships were detected "
                "in this dataset (minimum mutual information threshold not met)."
            )

        strong   = [r for r in causal if r.get("strength", 0) > 0.3]
        moderate = [r for r in causal if 0.1 < r.get("strength", 0) <= 0.3]
        methods  = {r.get("method", "") for r in causal} - {""}

        parts = [
            f"Analysis found {len(causal)} statistical relationship{'s' if len(causal) != 1 else ''} "
            f"using {', '.join(methods)} ({len(strong)} strong, {len(moderate)} moderate)."
        ]
        if strong:
            t = strong[0]
            parts.append(
                f"The strongest driver is **{t.get('from','')}** → **{t.get('to','')}** "
                f"(strength {t.get('strength',0):.3f}): {t.get('insight','')}."
            )
        if moderate:
            t = moderate[0]
            parts.append(
                f"Notable moderate association: **{t.get('from','')}** ↔ **{t.get('to','')}** "
                f"(strength {t.get('strength',0):.3f}) — worth monitoring in business reporting."
            )
        return "  ".join(parts)

    # ── Per-table description ──────────────────────────────────────────────
    def _table_description(self, prompt: str) -> str:
        nm = re.search(r"Describe '(.+?)'\s*\(([^)]+?)\s*rows?\)", prompt)
        table_name = nm.group(1) if nm else "this table"
        try:
            row_count = int(nm.group(2).replace(",", "").strip()) if nm else 0
        except Exception:
            row_count = 0

        cols_m = re.search(r'Columns:\s*(\[.*?\])', prompt, re.DOTALL)
        cols = _safe_json(cols_m.group(1)) if cols_m else [] or []

        purpose = _classify_table(table_name)

        numeric = [c["name"] for c in cols if any(
            t in c.get("type", "").lower()
            for t in ("int", "float", "double", "decimal", "numeric", "real", "number"))]
        text_c  = [c["name"] for c in cols if any(
            t in c.get("type", "").lower()
            for t in ("char", "text", "string", "varchar", "clob"))]
        date_c  = [c["name"] for c in cols if any(
            t in c.get("type", "").lower()
            for t in ("date", "time", "timestamp", "datetime"))]

        parts = [
            f"The **{table_name}** table is a {purpose} table with "
            f"{row_count:,} record{'s' if row_count != 1 else ''} and "
            f"{len(cols)} column{'s' if len(cols) != 1 else ''}."
        ]
        feat = []
        if numeric:
            feat.append(
                f"{len(numeric)} numeric field{'s' if len(numeric) != 1 else ''} "
                f"({', '.join(numeric[:3])}{'…' if len(numeric) > 3 else ''})"
            )
        if text_c:
            feat.append(
                f"{len(text_c)} text field{'s' if len(text_c) != 1 else ''} "
                f"({', '.join(text_c[:3])}{'…' if len(text_c) > 3 else ''})"
            )
        if date_c:
            feat.append(
                f"{len(date_c)} date/time field{'s' if len(date_c) != 1 else ''} "
                f"({', '.join(date_c[:2])})"
            )
        if feat:
            parts.append(f"It contains {', '.join(feat)}.")
        return "  ".join(parts)

    # ── Chatbot Q&A ────────────────────────────────────────────────────────
    def _chatbot(self, prompt: str) -> str:
        q_m = re.search(r'\nQ:\s*(.+?)$', prompt, re.MULTILINE)
        question = q_m.group(1).strip() if q_m else ""
        q = question.lower()

        # ─ parse context ─
        db_m = re.search(r'DB:\s*(\d+)\s*tables,\s*([\d,]+)\s*rows', prompt)
        n_tables  = int(db_m.group(1)) if db_m else 0
        total_rows = int(db_m.group(2).replace(",", "")) if db_m else 0

        tables: dict = {}
        tbl_sec = re.search(r'TABLES:\n(.*?)\nFKs:', prompt, re.DOTALL)
        if tbl_sec:
            for line in tbl_sec.group(1).strip().split('\n'):
                m = re.match(r'[-\s]*(\S+):\s*([\d,]+)\s*rows?,\s*health=(\S+)', line.strip())
                if m:
                    try:
                        h = float(m.group(3))
                    except Exception:
                        h = 0.0
                    tables[m.group(1)] = {
                        "rows":   int(m.group(2).replace(",", "")),
                        "health": h,
                    }

        edges: list = []
        fk_m = re.search(r'FKs:\s*(\[.*?\])', prompt, re.DOTALL)
        if fk_m:
            edges = _safe_json(fk_m.group(1)) or []

        causal: list = []
        cm = re.search(r'CAUSAL:\s*(\[.*?\])', prompt, re.DOTALL)
        if cm:
            causal = _safe_json(cm.group(1)) or []

        # ─ route ─
        if any(k in q for k in ["worst", "poor quality", "bad quality", "low quality", "lowest"]):
            return self._qa_worst(tables)
        if any(k in q for k in ["null", "missing", "empty", "incomplete", "blank"]):
            return self._qa_nulls(tables)
        if any(k in q for k in ["causal", "correlation", "influence", "relationship", "depend", "associate"]):
            return self._qa_causal(causal)
        if any(k in q for k in ["health", "score", "risk", "healthy"]):
            return self._qa_health(tables)
        if any(k in q for k in ["fix", "improve", "recommend", "priority", "action", "what to", "should i"]):
            return self._qa_recommendations(tables, causal)
        if any(k in q for k in ["foreign key", "fk", "connect", "link", "join", "relationship"]):
            return self._qa_relationships(edges, n_tables)
        if any(k in q for k in ["row", "count", "how many", "size", "large", "big", "small"]):
            return self._qa_counts(tables, n_tables, total_rows)
        if any(k in q for k in ["table", "schema", "structure", "overview", "summary", "about"]):
            return self._qa_overview(tables, n_tables, total_rows, edges, causal)
        return self._qa_general(tables, causal, edges, n_tables, total_rows, question)

    # ─ chatbot answer helpers ──────────────────────────────────────────────
    def _qa_worst(self, tables: dict) -> str:
        if not tables:
            return "No table quality data is available."
        srt = sorted(tables.items(), key=lambda x: x[1].get("health", 100))
        lines = ["**📉 Tables with the lowest quality scores:**\n"]
        for name, info in srt[:5]:
            h = info.get("health", 0)
            icon = "🚨" if h < 50 else ("⚠️" if h < 75 else "✅")
            lines.append(f"- **{name}**: {h:.0f}/100 {icon} — {info.get('rows',0):,} rows")
        lines.append("\n_Prioritise tables scoring below **75** for data-quality remediation._")
        return "\n".join(lines)

    def _qa_nulls(self, tables: dict) -> str:
        return (
            "**🔍 Missing / Null Value Details:**\n\n"
            "Per-column null rates are displayed in the **Schema** tab → expand any table → "
            "look at the **Null%** column (red = >30 %, amber = >10 %).\n\n"
            "The **Health** tab ranks every table by its composite score, which weights:\n"
            "- 35 % null rate\n"
            "- 35 % overall completeness\n"
            "- 30 % primary-key uniqueness\n\n"
            f"_Your database has **{len(tables)}** tables. "
            "Any table scoring below 75 likely has significant null or completeness issues._"
        )

    def _qa_causal(self, causal: list) -> str:
        if not causal:
            return (
                "No statistically significant causal relationships were found "
                "(mutual information or χ² threshold not reached).\n\n"
                "Try uploading a larger dataset or a dataset with stronger column correlations."
            )
        lines = [f"**⭐ {len(causal)} Statistical Relationship(s) Discovered:**\n"]
        for r in causal[:7]:
            icon = "🔴" if r.get("strength", 0) > 0.3 else ("🟡" if r.get("strength", 0) > 0.1 else "🟢")
            lines.append(
                f"{icon} `{r.get('from','')}` {r.get('direction','→')} `{r.get('to','')}`  "
                f"(strength **{r.get('strength',0):.3f}**, {r.get('method','')})\n"
                f"   _{r.get('insight','')}_"
            )
        if len(causal) > 7:
            lines.append(f"\n_…and {len(causal)-7} more. Open the **Causal** tab for the full table._")
        return "\n\n".join(lines)

    def _qa_health(self, tables: dict) -> str:
        if not tables:
            return "No health score data is available yet."
        healthy  = [(n, t) for n, t in tables.items() if t.get("health", 0) >= 75]
        at_risk  = [(n, t) for n, t in tables.items() if 50 <= t.get("health", 0) < 75]
        critical = [(n, t) for n, t in tables.items() if t.get("health", 0) < 50]
        avg = sum(t.get("health", 0) for t in tables.values()) / max(len(tables), 1)
        lines = [f"**💊 Health Summary  ·  {len(tables)} tables  ·  avg {avg:.1f}/100**\n"]
        lines.append(f"✅ **Healthy** (≥75): {len(healthy)}  — {', '.join(n for n,_ in healthy) or 'None'}")
        lines.append(f"⚠️ **At Risk** (50–74): {len(at_risk)}  — {', '.join(n for n,_ in at_risk) or 'None'}")
        lines.append(f"🚨 **Critical** (<50): {len(critical)}  — {', '.join(n for n,_ in critical) or 'None'}")
        if critical:
            lines.append(f"\n🔥 **Immediate action required on:** {', '.join(n for n,_ in critical[:3])}")
        return "\n".join(lines)

    def _qa_recommendations(self, tables: dict, causal: list) -> str:
        srt = sorted(tables.items(), key=lambda x: x[1].get("health", 100))
        lines = ["**🛠️ Recommended Actions (ranked by urgency):**\n"]
        for i, (name, info) in enumerate(srt[:6], 1):
            h = info.get("health", 0)
            if h < 50:
                action = "⚡ URGENT — full audit, check for corrupt / missing records"
            elif h < 75:
                action = "⚠️ Review null values, validate referential integrity, add constraints"
            else:
                action = "✅ Maintain current quality standards; monitor periodically"
            lines.append(f"{i}. **{name}** (score {h:.0f}/100)\n   → {action}")
        if causal:
            top = causal[0]
            lines.append(
                f"\n📊 **Analytical opportunity:** relationship "
                f"`{top.get('from','')}` → `{top.get('to','')}` "
                f"(strength {top.get('strength',0):.3f}) is a candidate for predictive modelling."
            )
        return "\n".join(lines)

    def _qa_relationships(self, edges: list, n_tables: int) -> str:
        if not edges:
            return (
                f"No explicit foreign key relationships were detected among the {n_tables} table(s).  "
                "The schema appears to use flat or independent tables.  "
                "Consider adding FK constraints to enforce referential integrity."
            )
        lines = [f"**🔗 {len(edges)} Foreign Key Relationship(s):**\n"]
        for e in edges[:10]:
            lines.append(f"- `{e.get('from','')}`.{e.get('fk_col','')} → `{e.get('to','')}`.{e.get('pk_col','')}")
        if len(edges) > 10:
            lines.append(f"_…and {len(edges)-10} more. See the **ER Diagram** tab._")
        return "\n".join(lines)

    def _qa_counts(self, tables: dict, n_tables: int, total_rows: int) -> str:
        lines = [f"**📊 Row Counts  ·  {n_tables} tables  ·  {total_rows:,} total rows**\n"]
        for name, info in sorted(tables.items(), key=lambda x: x[1].get("rows", 0), reverse=True):
            lines.append(f"- **{name}**: {info.get('rows',0):,} rows")
        return "\n".join(lines)

    def _qa_overview(self, tables, n_tables, total_rows, edges, causal) -> str:
        domain = _detect_domain(list(tables.keys()))
        avg = sum(t.get("health", 0) for t in tables.values()) / max(len(tables), 1)
        largest = max(tables.items(), key=lambda x: x[1].get("rows", 0), default=(None, {}))
        lines = [
            f"**🧠 Database Overview — {domain}**\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Tables | {n_tables} |",
            f"| Total Rows | {total_rows:,} |",
            f"| FK Relationships | {len(edges)} |",
            f"| Causal Findings | {len(causal)} |",
            f"| Avg Health Score | {avg:.1f} / 100 |",
        ]
        if largest[0]:
            lines.append(f"| Largest Table | {largest[0]} ({largest[1].get('rows',0):,} rows) |")
        return "\n".join(lines)

    def _qa_general(self, tables, causal, edges, n_tables, total_rows, question) -> str:
        domain = _detect_domain(list(tables.keys()))
        avg = sum(t.get("health", 0) for t in tables.values()) / max(len(tables), 1)
        return (
            f"**Based on your {domain} database:**\n\n"
            f"- {n_tables} tables · {total_rows:,} rows · {len(edges)} FK relationships\n"
            f"- {len(causal)} causal relationship(s) detected\n"
            f"- Average health score: **{avg:.1f} / 100**\n\n"
            f'💬 Your question: _"{question}"_\n\n'
            "For detailed exploration, use the **Schema**, **Health**, **Causal**, and **ER Diagram** tabs."
        )
