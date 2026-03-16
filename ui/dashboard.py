from __future__ import annotations
import io, json, os, tempfile, warnings, zipfile, base64
from datetime import datetime
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from dotenv import load_dotenv
from scipy.stats import chi2_contingency
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sqlalchemy import create_engine, inspect, text

load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEASYPRINT PDF BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_pdf_weasy(
    schema, quality, health, causal,
    edges, ai_text, contracts, mermaid_svg=None
):
    """Generate PDF from HTML using WeasyPrint."""
    try:
        from weasyprint import HTML
    except ImportError:
        return None

    avg_h = (
        np.mean([h["score"] for h in health.values()])
        if health else 0
    )

    # Build metric boxes
    metrics_html = ""
    metric_items = [
        ("Tables", str(schema["total_tables"]), "#6c5ce7"),
        ("Columns", str(schema["total_columns"]), "#00cec9"),
        ("Rows", f"{schema['total_rows']:,}", "#0984e3"),
        ("Relations", str(len(edges)), "#fdcb6e"),
        ("Causal", str(len(causal)), "#e17055"),
        ("Health", f"{avg_h:.0f}/100", "#00b894"),
    ]
    for label, value, color in metric_items:
        metrics_html += f"""
        <div class="metric-box" style="border-top: 4px solid {color};">
            <div class="metric-val">{value}</div>
            <div class="metric-lbl">{label}</div>
        </div>"""

    # Schema table rows
    schema_rows = ""
    for n, t in schema["tables"].items():
        h = health.get(n, {})
        q = quality.get(n, {})
        score = h.get("score", "-")
        label = h.get("label", "-")
        comp = q.get("overall_completeness", 0)
        health_class = (
            "healthy" if h.get("score", 0) >= 75
            else "warning" if h.get("score", 0) >= 50
            else "critical"
        )
        schema_rows += f"""
        <tr>
            <td><strong>{n}</strong></td>
            <td>{t['row_count']:,}</td>
            <td>{len(t['columns'])}</td>
            <td>{', '.join(t['primary_keys'][:3]) or '-'}</td>
            <td>{len(t['foreign_keys'])}</td>
            <td>{comp:.0%}</td>
            <td><span class="badge {health_class}">
                {score}/100 {label}
            </span></td>
        </tr>"""

    # Health bars
    health_bars = ""
    for h in sorted(health.values(), key=lambda x: x["score"]):
        bar_color = (
            "#00b894" if h["score"] >= 75
            else "#fdcb6e" if h["score"] >= 50
            else "#e17055"
        )
        health_bars += f"""
        <div class="health-row">
            <div class="health-name">{h['table']}</div>
            <div class="health-bar-bg">
                <div class="health-bar-fill"
                     style="width:{h['score']}%;
                            background:{bar_color};">
                </div>
            </div>
            <div class="health-score">{h['score']:.0f}/100</div>
            <div class="health-label badge
                 {'healthy' if h['score']>=75
                  else 'warning' if h['score']>=50
                  else 'critical'}">
                {h['label']}
            </div>
        </div>"""

    # Health breakdown table
    health_breakdown = ""
    for h in sorted(health.values(), key=lambda x: x["score"]):
        bd = h.get("breakdown", {})
        health_breakdown += f"""
        <tr>
            <td><strong>{h['table']}</strong></td>
            <td>{h['score']:.0f}</td>
            <td>{bd.get('Nulls', 0):.0f}</td>
            <td>{bd.get('Completeness', 0):.0f}</td>
            <td>{bd.get('PK_Uniqueness', 0):.0f}</td>
            <td><em>{h.get('recommendation', '')}</em></td>
        </tr>"""

    # ER diagram section
    er_section = ""
    if mermaid_svg:
        er_section = f"""
        <div class="er-diagram">
            {mermaid_svg}
        </div>"""
    else:
        er_section = """
        <p class="muted">
            ER diagram available in interactive version.
        </p>"""

    # FK table
    fk_rows = ""
    for e in edges:
        fk_rows += f"""
        <tr>
            <td>{e['from']}</td>
            <td><code>{e['fk_col']}</code></td>
            <td>{e['to']}</td>
            <td><code>{e['pk_col']}</code></td>
        </tr>"""

    # Table detail sections
    table_details = ""
    for tn, t in schema["tables"].items():
        h = health.get(tn, {})
        q = quality.get(tn, {})
        qc = q.get("columns", {})
        ai_desc = ai_text.get("tables", {}).get(tn, "")
        score_display = h.get("score", "-")
        health_class = (
            "healthy" if h.get("score", 0) >= 75
            else "warning" if h.get("score", 0) >= 50
            else "critical"
        )

        col_rows = ""
        for col in t["columns"]:
            cd = qc.get(col["name"], {})
            fk_str = (
                f"&rarr; {col['ref_table']}.{col['ref_col']}"
                if col["is_fk"] else ""
            )
            null_rate = cd.get("null_rate", 0)
            null_class = (
                "critical" if null_rate > 0.3
                else "warning" if null_rate > 0.1
                else ""
            )
            col_rows += f"""
            <tr>
                <td><code>{col['name']}</code></td>
                <td>{col['type']}</td>
                <td>{'🔑' if col['is_pk'] else ''}</td>
                <td>{fk_str}</td>
                <td class="{null_class}">
                    {null_rate:.1%}
                </td>
                <td>{cd.get('unique_rate', 0):.1%}</td>
                <td>{cd.get('col_kind', '-')}</td>
            </tr>"""

        table_details += f"""
        <div class="table-detail">
            <h3>
                {tn}
                <span class="badge {health_class}">
                    {score_display}/100
                </span>
            </h3>
            <div class="table-meta">
                Rows: {t['row_count']:,} |
                Columns: {len(t['columns'])} |
                PKs: {', '.join(t['primary_keys']) or 'None'}
            </div>
            {'<div class="ai-desc">' + ai_desc + '</div>'
             if ai_desc else ''}
            <table>
                <thead>
                    <tr>
                        <th>Column</th><th>Type</th>
                        <th>PK</th><th>FK</th>
                        <th>Null%</th><th>Unique%</th>
                        <th>Kind</th>
                    </tr>
                </thead>
                <tbody>{col_rows}</tbody>
            </table>
        </div>"""

    # Causal table
    causal_rows = ""
    for r in causal[:30]:
        strength_class = (
            "healthy" if r["strength"] > 0.3
            else "warning" if r["strength"] > 0.1
            else ""
        )
        causal_rows += f"""
        <tr>
            <td><code>{r['from']}</code></td>
            <td class="direction">{r['direction']}</td>
            <td><code>{r['to']}</code></td>
            <td>{r['method']}</td>
            <td class="{strength_class}">
                {r['strength']:.4f}
            </td>
            <td>{r['p_value']:.4f}</td>
            <td><em>{r['insight']}</em></td>
        </tr>"""

    # Contract sections
    contract_sections = ""
    for tn, ys in contracts.items():
        h = health.get(tn, {})
        health_class = (
            "healthy" if h.get("score", 0) >= 75
            else "warning" if h.get("score", 0) >= 50
            else "critical"
        )
        # Escape HTML in YAML
        safe_yaml = (
            ys.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        contract_sections += f"""
        <div class="contract-block">
            <h4>
                {tn}
                <span class="badge {health_class}">
                    {h.get('score', '-')}/100
                </span>
            </h4>
            <pre><code>{safe_yaml}</code></pre>
        </div>"""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: A4;
            margin: 1.5cm 1.5cm 2cm 1.5cm;
            @bottom-center {{
                content: "Page " counter(page) " of "
                         counter(pages);
                font-size: 8pt;
                color: #999;
            }}
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, sans-serif;
            font-size: 10pt;
            color: #2d3436;
            line-height: 1.5;
            background: white;
        }}

        /* Cover Page */
        .cover {{
            page-break-after: always;
            text-align: center;
            padding-top: 120px;
        }}
        .cover h1 {{
            font-size: 32pt;
            color: #6c5ce7;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }}
        .cover .subtitle {{
            font-size: 14pt;
            color: #636e72;
            margin-bottom: 40px;
        }}
        .cover .date {{
            font-size: 11pt;
            color: #b2bec3;
            margin-top: 30px;
        }}
        .cover-box {{
            background: linear-gradient(135deg, #6c5ce7, #00cec9);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            margin: 30px auto;
            max-width: 500px;
            text-align: left;
            font-size: 10pt;
            line-height: 1.8;
        }}

        /* Metrics */
        .metrics {{
            display: flex;
            gap: 10px;
            margin: 15px 0;
            flex-wrap: wrap;
        }}
        .metric-box {{
            flex: 1;
            min-width: 80px;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px 8px;
            text-align: center;
        }}
        .metric-val {{
            font-size: 18pt;
            font-weight: 700;
            color: #2d3436;
        }}
        .metric-lbl {{
            font-size: 7pt;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #636e72;
            margin-top: 4px;
        }}

        /* Sections */
        h2 {{
            font-size: 16pt;
            color: #6c5ce7;
            border-bottom: 3px solid #6c5ce7;
            padding-bottom: 5px;
            margin: 25px 0 12px 0;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 12pt;
            color: #2d3436;
            margin: 15px 0 8px 0;
            page-break-after: avoid;
        }}
        h4 {{
            font-size: 11pt;
            color: #00cec9;
            margin: 10px 0 5px 0;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 8px 0 15px 0;
            font-size: 8pt;
            page-break-inside: auto;
        }}
        tr {{ page-break-inside: avoid; }}
        th {{
            background: #6c5ce7;
            color: white;
            padding: 6px 8px;
            text-align: left;
            font-weight: 600;
            font-size: 8pt;
        }}
        td {{
            padding: 5px 8px;
            border-bottom: 1px solid #eee;
        }}
        tr:nth-child(even) td {{ background: #f8f9fa; }}
        tr:hover td {{ background: #f0f0ff; }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 7pt;
            font-weight: 600;
        }}
        .healthy {{
            background: #d4edda;
            color: #155724;
        }}
        .warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .critical {{
            background: #f8d7da;
            color: #721c24;
        }}

        /* Health bars */
        .health-row {{
            display: flex;
            align-items: center;
            margin: 6px 0;
            gap: 8px;
        }}
        .health-name {{
            width: 140px;
            font-size: 9pt;
            font-weight: 500;
        }}
        .health-bar-bg {{
            flex: 1;
            height: 14px;
            background: #eee;
            border-radius: 7px;
            overflow: hidden;
        }}
        .health-bar-fill {{
            height: 100%;
            border-radius: 7px;
            transition: width 0.3s;
        }}
        .health-score {{
            width: 50px;
            font-size: 9pt;
            font-weight: 700;
            text-align: right;
        }}
        .health-label {{ width: 60px; text-align: center; }}

        /* AI descriptions */
        .ai-desc {{
            background: #f0f0ff;
            border-left: 4px solid #6c5ce7;
            padding: 8px 12px;
            margin: 8px 0;
            font-style: italic;
            font-size: 9pt;
            color: #4a4a6a;
            border-radius: 0 6px 6px 0;
        }}

        /* Table detail blocks */
        .table-detail {{
            margin: 15px 0;
            padding: 12px;
            border: 1px solid #eee;
            border-radius: 8px;
            page-break-inside: avoid;
        }}
        .table-meta {{
            font-size: 9pt;
            color: #636e72;
            margin-bottom: 8px;
        }}

        /* ER Diagram */
        .er-diagram {{
            text-align: center;
            margin: 15px 0;
            padding: 10px;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid #eee;
        }}
        .er-diagram svg {{
            max-width: 100%;
            height: auto;
        }}

        /* Contracts */
        .contract-block {{
            margin: 10px 0;
            page-break-inside: avoid;
        }}
        pre {{
            background: #2d3436;
            color: #dfe6e9;
            padding: 12px;
            border-radius: 6px;
            font-size: 7pt;
            line-height: 1.4;
            overflow-wrap: break-word;
            white-space: pre-wrap;
        }}
        code {{ font-family: 'Consolas', monospace; }}

        /* Direction arrows */
        .direction {{
            font-weight: 700;
            color: #6c5ce7;
            font-size: 12pt;
        }}

        .muted {{
            color: #b2bec3;
            font-style: italic;
        }}

        /* Appendix */
        .appendix-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin: 10px 0;
        }}
        .appendix-item {{
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 9pt;
        }}
        .appendix-item strong {{ color: #6c5ce7; }}
    </style>
    </head>
    <body>

    <!-- COVER PAGE -->
    <div class="cover">
        <h1>Database Intelligence Report</h1>
        <div class="subtitle">
            Schema Analysis &bull; Data Quality &bull;
            Causal Discovery &bull; Data Contracts
        </div>
        <div class="cover-box">
            {ai_text.get('schema', 'Analysis complete.')}
        </div>
        <div class="date">
            Generated: {datetime.now():%Y-%m-%d %H:%M} |
            DB Intelligence Agent v2.0
        </div>
    </div>

    <!-- OVERVIEW -->
    <h2>1. Database Overview</h2>
    <div class="metrics">{metrics_html}</div>

    <table>
        <thead>
            <tr>
                <th>Table</th><th>Rows</th>
                <th>Columns</th><th>Primary Keys</th>
                <th>FKs</th><th>Completeness</th>
                <th>Health</th>
            </tr>
        </thead>
        <tbody>{schema_rows}</tbody>
    </table>

    <!-- HEALTH SCORES -->
    <h2>2. Health Scores</h2>
    {health_bars}

    <h3>Score Breakdown</h3>
    <table>
        <thead>
            <tr>
                <th>Table</th><th>Score</th>
                <th>Nulls</th><th>Complete</th>
                <th>PK Uniq</th><th>Recommendation</th>
            </tr>
        </thead>
        <tbody>{health_breakdown}</tbody>
    </table>

    <!-- ER DIAGRAM -->
    <h2>3. Entity Relationships</h2>
    {er_section}

    {'<h3>Foreign Key Details</h3><table><thead><tr>'
     '<th>From</th><th>FK Column</th>'
     '<th>To</th><th>PK Column</th>'
     '</tr></thead><tbody>'
     + fk_rows +
     '</tbody></table>'
     if edges else ''}

    <!-- TABLE DETAILS -->
    <h2>4. Table Details</h2>
    {table_details}

    <!-- CAUSAL -->
    <h2>5. Causal Intelligence</h2>
    {'<div class="ai-desc">'
     + ai_text.get("causal", "")
     + '</div>' if ai_text.get("causal") else ''}

    {'<h3>Discovered Relationships ('
     + str(len(causal))
     + ' found)</h3>'
     '<table><thead><tr>'
     '<th>From</th><th>Dir</th><th>To</th>'
     '<th>Method</th><th>Strength</th>'
     '<th>P-value</th><th>Insight</th>'
     '</tr></thead><tbody>'
     + causal_rows +
     '</tbody></table>'
     if causal else
     '<p class="muted">No significant relationships.</p>'}

    <!-- CONTRACTS -->
    <h2>6. Data Contracts (YAML)</h2>
    <p>{len(contracts)} contracts generated with
       column-level quality SLAs.</p>
    {contract_sections}

    <!-- APPENDIX -->
    <h2>7. Appendix</h2>
    <div class="appendix-grid">
        <div class="appendix-item">
            <strong>Generated:</strong>
            {datetime.now():%Y-%m-%d %H:%M:%S}
        </div>
        <div class="appendix-item">
            <strong>Tables:</strong>
            {schema['total_tables']}
        </div>
        <div class="appendix-item">
            <strong>Columns:</strong>
            {schema['total_columns']}
        </div>
        <div class="appendix-item">
            <strong>Rows:</strong>
            {schema['total_rows']:,}
        </div>
        <div class="appendix-item">
            <strong>Causal Pairs:</strong>
            {len(causal)}
        </div>
        <div class="appendix-item">
            <strong>Contracts:</strong>
            {len(contracts)}
        </div>
        <div class="appendix-item">
            <strong>Avg Health:</strong>
            {avg_h:.1f}/100
        </div>
        <div class="appendix-item">
            <strong>Engine:</strong>
            DB Intelligence Agent
        </div>
    </div>

    </body>
    </html>
    """

    pdf_bytes = HTML(string=html_content).write_pdf()
    return pdf_bytes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MERMAID ER DIAGRAM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_mermaid_er(schema) -> str:
    """Build Mermaid erDiagram code from schema."""
    lines = ["erDiagram"]
    tables = schema["tables"]

    for tn, t in tables.items():
        safe_name = tn.replace(" ", "_").replace("-", "_")
        for col in t["columns"]:
            col_type = (
                col["type"]
                .split("(")[0]
                .replace(" ", "")
                .upper()[:15]
            )
            if not col_type:
                col_type = "TEXT"
            markers = ""
            if col["is_pk"]:
                markers = "PK"
            elif col["is_fk"]:
                markers = "FK"
            safe_col = (
                col["name"]
                .replace(" ", "_")
                .replace("-", "_")
            )
            lines.append(
                f"    {safe_name} {{"
                f"\n        {col_type} {safe_col}"
                f' "{markers}"'
                f"\n    }}"
            )

    # Relationships
    for tn, t in tables.items():
        safe_from = tn.replace(" ", "_").replace("-", "_")
        for fk in t["foreign_keys"]:
            ref = fk.get("referred_table", "")
            if not ref or ref not in tables:
                continue
            safe_to = (
                ref.replace(" ", "_").replace("-", "_")
            )
            fc = (
                fk["constrained_columns"][0]
                if fk["constrained_columns"]
                else "id"
            )
            lines.append(
                f'    {safe_from} }}o--|| {safe_to}'
                f' : "{fc}"'
            )

    return "\n".join(lines)


def build_mermaid_er_clean(schema) -> str:
    """Build clean Mermaid ER code with entity blocks."""
    lines = ["erDiagram"]
    tables = schema["tables"]

    # Entity definitions
    for tn, t in tables.items():
        safe = tn.replace(" ", "_").replace("-", "_")
        lines.append(f"    {safe} {{")
        for col in t["columns"][:15]:
            ctype = (
                col["type"]
                .split("(")[0]
                .strip()
                .upper()[:12]
            )
            if not ctype:
                ctype = "TEXT"
            cname = (
                col["name"]
                .replace(" ", "_")
                .replace("-", "_")
            )
            marker = ""
            if col["is_pk"]:
                marker = "PK"
            elif col["is_fk"]:
                marker = "FK"
            if marker:
                lines.append(
                    f'        {ctype} {cname} "{marker}"'
                )
            else:
                lines.append(f"        {ctype} {cname}")
        if len(t["columns"]) > 15:
            lines.append(
                f"        TEXT _more_"
                f' "{len(t["columns"])-15} more"'
            )
        lines.append("    }")

    # Relationships
    for tn, t in tables.items():
        sf = tn.replace(" ", "_").replace("-", "_")
        for fk in t["foreign_keys"]:
            ref = fk.get("referred_table", "")
            if not ref or ref not in tables:
                continue
            st_name = (
                ref.replace(" ", "_").replace("-", "_")
            )
            fc = (
                fk["constrained_columns"][0]
                if fk["constrained_columns"]
                else "id"
            )
            lines.append(
                f'    {sf} }}o--|| {st_name} : "{fc}"'
            )

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THEME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="DB Intelligence Agent",
    layout="wide",
    page_icon="🧠",
)

COLORS = {
    "bg": "#0a0a0f", "card": "#12121a",
    "accent": "#6c5ce7", "accent2": "#00cec9",
    "success": "#00b894", "warning": "#fdcb6e",
    "danger": "#e17055", "text": "#dfe6e9",
    "muted": "#636e72",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp {{
    background: linear-gradient(
        135deg, {COLORS['bg']} 0%,
        #0d0d1a 50%, #111128 100%
    );
    font-family: 'Inter', sans-serif;
}}
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding: 1rem 2rem;}}
.glass-card {{
    background: rgba(18,18,26,0.8);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(108,92,231,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}}
.glass-card:hover {{
    border-color: rgba(108,92,231,0.5);
    box-shadow: 0 0 30px rgba(108,92,231,0.15);
    transform: translateY(-2px);
}}
.metric-card {{
    background: linear-gradient(
        135deg, rgba(108,92,231,0.15),
        rgba(0,206,201,0.1)
    );
    border: 1px solid rgba(108,92,231,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}}
.metric-value {{
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(
        135deg, {COLORS['accent']}, {COLORS['accent2']}
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.metric-label {{
    font-size: 0.85rem; color: {COLORS['muted']};
    text-transform: uppercase;
    letter-spacing: 1px; margin-top: 4px;
}}
.hero {{
    text-align: center; padding: 3rem 1rem;
    background: radial-gradient(
        ellipse at center,
        rgba(108,92,231,0.12) 0%, transparent 70%
    );
}}
.hero h1 {{
    font-size: 3rem; font-weight: 700;
    background: linear-gradient(
        135deg, #fff 0%,
        {COLORS['accent']} 50%,
        {COLORS['accent2']} 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.hero p {{
    color: {COLORS['muted']};
    font-size: 1.1rem;
    max-width: 600px; margin: 0 auto;
}}
.chat-msg {{
    padding: 1rem; border-radius: 12px;
    margin: 0.5rem 0; max-width: 85%;
}}
.chat-user {{
    background: linear-gradient(
        135deg, rgba(108,92,231,0.3),
        rgba(108,92,231,0.1)
    );
    border: 1px solid rgba(108,92,231,0.3);
    margin-left: auto;
}}
.chat-ai {{
    background: rgba(18,18,26,0.8);
    border: 1px solid rgba(0,206,201,0.2);
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px; background: rgba(18,18,26,0.5);
    border-radius: 12px; padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {COLORS['muted']}; font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(
        135deg, {COLORS['accent']}, {COLORS['accent2']}
    ) !important;
    color: white !important;
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
::-webkit-scrollbar-thumb {{
    background: {COLORS['accent']}; border-radius: 3px;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(
        180deg, #0d0d1a 0%, #12121a 100%
    );
    border-right: 1px solid rgba(108,92,231,0.15);
}}
.stButton > button {{
    background: linear-gradient(
        135deg, {COLORS['accent']}, {COLORS['accent2']}
    ) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108,92,231,0.4) !important;
}}
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI PROVIDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AI:
    PROVIDERS = {
        "anthropic": {
            "name": "Claude", "icon": "🟠",
            "models": [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
            ],
            "key": "ANTHROPIC_API_KEY",
        },
        "openai": {
            "name": "GPT-4o", "icon": "🟢",
            "models": ["gpt-4o", "gpt-4o-mini"],
            "key": "OPENAI_API_KEY",
        },
        "gemini": {
            "name": "Gemini", "icon": "🔵",
            "models": ["gemini-2.0-flash", "gemini-1.5-pro"],
            "key": "GOOGLE_API_KEY",
        },
        "groq": {
            "name": "Groq", "icon": "🟣",
            "models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ],
            "key": "GROQ_API_KEY",
        },
    }

    def __init__(self, provider=None, model=None):
        self.provider = provider or os.getenv(
            "DEFAULT_AI_PROVIDER", "groq"
        )
        self.model = model or os.getenv(
            "DEFAULT_AI_MODEL",
            self.PROVIDERS[self.provider]["models"][0],
        )
        self.client = self._init()

    def _init(self):
        api_key = os.getenv(
            self.PROVIDERS[self.provider]["key"]
        )
        if not api_key:
            return None
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=api_key)
        elif self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model)
        elif self.provider == "groq":
            from groq import Groq
            return Groq(api_key=api_key)

    def ask(self, prompt: str, max_tokens=1024) -> str:
        if not self.client:
            return (
                f"No API key for {self.provider}. "
                f"Set {self.PROVIDERS[self.provider]['key']}"
                " in .env"
            )
        try:
            if self.provider == "anthropic":
                r = self.client.messages.create(
                    model=self.model, max_tokens=max_tokens,
                    messages=[{
                        "role": "user", "content": prompt
                    }],
                )
                return r.content[0].text
            elif self.provider in ("openai", "groq"):
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=max_tokens,
                    messages=[{
                        "role": "user", "content": prompt
                    }],
                )
                return r.choices[0].message.content
            elif self.provider == "gemini":
                return self.client.generate_content(
                    prompt
                ).text
        except Exception as e:
            return f"AI Error ({self.provider}): {e}"

    @classmethod
    def available(cls):
        return {
            k: v for k, v in cls.PROVIDERS.items()
            if os.getenv(v["key"])
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATABASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_sqlite(f):
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".db"
    )
    tmp.write(f.getbuffer())
    tmp.close()
    return create_engine(f"sqlite:///{tmp.name}")


def load_csvs(files):
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".db"
    )
    tmp.close()
    eng = create_engine(f"sqlite:///{tmp.name}")
    names = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            name = (
                os.path.splitext(f.name)[0]
                .lower().replace(" ", "_").replace("-", "_")
            )
            df.to_sql(
                name, eng,
                if_exists="replace", index=False,
            )
            names.append(name)
        except Exception:
            pass
    return eng, names


def test_db(eng):
    try:
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCHEMA + QUALITY + CAUSAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_schema(eng):
    insp = inspect(eng)
    tables = {}
    for tbl in insp.get_table_names():
        try:
            cols = insp.get_columns(tbl)
            pk = insp.get_pk_constraint(tbl).get(
                "constrained_columns", []
            )
            fks = insp.get_foreign_keys(tbl)
            idxs = insp.get_indexes(tbl)
            with eng.connect() as c:
                rows = c.execute(
                    text(f'SELECT COUNT(*) FROM "{tbl}"')
                ).scalar() or 0
        except Exception:
            continue
        fk_map, fk_list = {}, []
        for fk in fks:
            for cc, rc in zip(
                fk.get("constrained_columns", []),
                fk.get("referred_columns", []),
            ):
                fk_map[cc] = {
                    "ref_table": fk.get(
                        "referred_table", ""
                    ),
                    "ref_col": rc,
                }
            fk_list.append({
                "constrained_columns": fk.get(
                    "constrained_columns", []
                ),
                "referred_table": fk.get(
                    "referred_table", ""
                ),
                "referred_columns": fk.get(
                    "referred_columns", []
                ),
            })
        columns = []
        for col in cols:
            n = col["name"]
            fi = fk_map.get(n, {})
            columns.append({
                "name": n,
                "type": str(col.get("type", "TEXT")),
                "nullable": bool(col.get("nullable", True)),
                "is_pk": n in pk,
                "is_fk": n in fk_map,
                "ref_table": fi.get("ref_table", ""),
                "ref_col": fi.get("ref_col", ""),
            })
        tables[tbl] = {
            "name": tbl, "row_count": rows,
            "columns": columns, "primary_keys": pk,
            "foreign_keys": fk_list,
            "index_count": len(idxs),
        }
    # Heuristic FK
    if not sum(
        len(t["foreign_keys"]) for t in tables.values()
    ):
        tnames = list(tables.keys())
        for tn, t in tables.items():
            for col in t["columns"]:
                if (
                    col["name"].endswith("_id")
                    and not col["is_pk"]
                ):
                    base = col["name"][:-3]
                    for cand in [
                        base, base + "s", base + "es"
                    ]:
                        if cand in tnames and cand != tn:
                            rc_list = [
                                c["name"]
                                for c in tables[cand]["columns"]
                            ]
                            rc = (
                                col["name"]
                                if col["name"] in rc_list
                                else (
                                    "id"
                                    if "id" in rc_list
                                    else None
                                )
                            )
                            if rc:
                                col["is_fk"] = True
                                col["ref_table"] = cand
                                col["ref_col"] = rc
                                t["foreign_keys"].append({
                                    "constrained_columns": [
                                        col["name"]
                                    ],
                                    "referred_table": cand,
                                    "referred_columns": [rc],
                                })
                                break
    return {
        "tables": tables,
        "total_tables": len(tables),
        "total_columns": sum(
            len(t["columns"]) for t in tables.values()
        ),
        "total_rows": sum(
            t["row_count"] for t in tables.values()
        ),
    }


def sample_tables(eng, schema, n=5000):
    out = {}
    for tbl in schema["tables"]:
        try:
            with eng.connect() as c:
                out[tbl] = pd.read_sql(
                    text(
                        f'SELECT * FROM "{tbl}" LIMIT :n'
                    ),
                    c, params={"n": n},
                )
        except Exception:
            out[tbl] = pd.DataFrame()
    return out


def _sf(v):
    try:
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None
    except Exception:
        return None


def profile_table(df):
    if df is None or df.empty:
        return {
            "sampled_rows": 0, "columns": {},
            "overall_completeness": 0,
        }
    cols = {}
    for c in df.columns:
        s = df[c]
        total = len(s)
        nulls = int(s.isna().sum())
        nr = round(nulls / total, 6) if total else 0
        uniq = int(s.nunique(dropna=True))
        p = {
            "dtype": str(s.dtype),
            "null_count": nulls,
            "null_rate": nr,
            "completeness": round(1 - nr, 6),
            "unique_count": uniq,
            "unique_rate": round(
                uniq / max(total - nulls, 1), 6
            ),
            "col_kind": "unknown",
        }
        clean = s.dropna()
        if clean.empty:
            cols[c] = p
            continue
        if pd.api.types.is_numeric_dtype(s):
            p.update({
                "col_kind": "numeric",
                "min_value": _sf(clean.min()),
                "max_value": _sf(clean.max()),
                "mean_value": _sf(clean.mean()),
                "std_value": _sf(clean.std()),
            })
        else:
            vc = (
                clean.astype(str).str[:80]
                .value_counts().head(10)
            )
            p["top_values"] = [
                {"value": str(k), "count": int(v)}
                for k, v in vc.items()
            ]
            p["col_kind"] = "categorical"
        cols[c] = p
    vals = [v["completeness"] for v in cols.values()]
    return {
        "sampled_rows": len(df), "columns": cols,
        "overall_completeness": (
            round(float(np.mean(vals)), 4) if vals else 0
        ),
        "worst_columns": sorted(
            cols,
            key=lambda x: cols[x]["null_rate"],
            reverse=True,
        )[:3],
    }


def profile_all(schema, samples):
    return {t: profile_table(d) for t, d in samples.items()}


def find_causal(
    samples, mi_min=0.05, chi2_p=0.05,
    max_cols=8, max_cats=25,
):
    rels = []
    for tbl, df in samples.items():
        if df is None or df.empty or len(df) < 50:
            continue
        num = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()[:max_cols]
        cat = [
            c for c in df.select_dtypes(
                include=["object", "category"]
            ).columns
            if df[c].nunique() <= max_cats
        ][:max_cols]

        if len(num) >= 2:
            for target in num[:6]:
                feats = [c for c in num if c != target][:5]
                if not feats:
                    continue
                try:
                    scores = mutual_info_regression(
                        df[feats].fillna(0).values,
                        df[target].fillna(0).values,
                        random_state=42,
                    )
                    for j, sc in enumerate(scores):
                        if sc >= mi_min:
                            lv = (
                                "strongly" if sc > 0.3
                                else "moderately"
                                if sc > 0.1 else "weakly"
                            )
                            rels.append({
                                "from": f"{tbl}.{feats[j]}",
                                "to": f"{tbl}.{target}",
                                "method": "mutual_info",
                                "strength": round(
                                    float(sc), 4
                                ),
                                "p_value": round(
                                    max(0.001, 1 - float(sc)),
                                    4,
                                ),
                                "direction": "->",
                                "insight": (
                                    f"'{feats[j]}' {lv} "
                                    f"influences '{target}'"
                                ),
                            })
                except Exception:
                    pass

        if len(cat) >= 2:
            for i in range(min(len(cat), 6)):
                for j in range(i + 1, min(len(cat), 6)):
                    try:
                        ct = pd.crosstab(
                            df[cat[i]].fillna("_")
                            .astype(str).str[:50],
                            df[cat[j]].fillna("_")
                            .astype(str).str[:50],
                        )
                        if (
                            ct.shape[0] < 2
                            or ct.shape[1] < 2
                        ):
                            continue
                        chi2, p, _, _ = chi2_contingency(ct)
                        n = ct.values.sum()
                        md = min(ct.shape) - 1
                        if md <= 0:
                            continue
                        cv = float(np.sqrt(chi2 / (n * md)))
                        if p < chi2_p and cv >= 0.1:
                            lv = (
                                "strongly" if cv > 0.3
                                else "moderately"
                                if cv > 0.1 else "weakly"
                            )
                            rels.append({
                                "from": f"{tbl}.{cat[i]}",
                                "to": f"{tbl}.{cat[j]}",
                                "method": "chi_squared",
                                "strength": round(cv, 4),
                                "p_value": round(
                                    float(p), 4
                                ),
                                "direction": "<->",
                                "insight": (
                                    f"'{cat[i]}' & "
                                    f"'{cat[j]}' {lv} "
                                    "associated"
                                ),
                            })
                    except Exception:
                        pass

        if num and cat:
            for cc in cat[:4]:
                try:
                    y = pd.factorize(
                        df[cc].fillna("_").astype(str)
                    )[0]
                    scores = mutual_info_classif(
                        df[num[:5]].fillna(0).values,
                        y, random_state=42,
                    )
                    for j, sc in enumerate(scores):
                        if sc >= mi_min:
                            lv = (
                                "strongly" if sc > 0.3
                                else "moderately"
                                if sc > 0.1 else "weakly"
                            )
                            rels.append({
                                "from": f"{tbl}.{num[j]}",
                                "to": f"{tbl}.{cc}",
                                "method": "mi_classif",
                                "strength": round(
                                    float(sc), 4
                                ),
                                "p_value": round(
                                    max(0.001, 1 - float(sc)),
                                    4,
                                ),
                                "direction": "->",
                                "insight": (
                                    f"'{num[j]}' {lv} "
                                    f"predicts '{cc}'"
                                ),
                            })
                except Exception:
                    pass

    rels.sort(key=lambda r: r["strength"], reverse=True)
    return rels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEALTH + CONTRACTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_health(schema, quality):
    scores = {}
    for tbl, q in quality.items():
        cols = q.get("columns", {})
        if not cols:
            continue
        avg_null = float(np.mean(
            [v.get("null_rate", 0) for v in cols.values()]
        ))
        null_s = max(0, 100 * (1 - avg_null * 2))
        comp_s = float(
            q.get("overall_completeness", 0.8)
        ) * 100
        pk_cols = schema["tables"].get(
            tbl, {}
        ).get("primary_keys", [])
        uniq_s = 100.0
        for pk in pk_cols:
            if pk in cols:
                uniq_s = min(
                    uniq_s,
                    cols[pk].get("unique_rate", 1) * 100,
                )
        score = round(min(100, max(
            0,
            null_s * 0.35 + comp_s * 0.35 + uniq_s * 0.30
        )), 1)
        label = (
            "Healthy" if score >= 75
            else "At Risk" if score >= 50
            else "Critical"
        )
        icon = (
            "✅" if score >= 75
            else "⚠️" if score >= 50
            else "🚨"
        )
        worst = q.get("worst_columns", [])
        rec = (
            f"'{tbl}' is healthy." if score >= 75
            else f"Fix nulls in {', '.join(worst[:2])}."
            if score >= 50
            else f"CRITICAL: Audit '{tbl}' now."
        )
        scores[tbl] = {
            "table": tbl, "score": score,
            "label": label, "icon": icon,
            "breakdown": {
                "Nulls": round(null_s, 1),
                "Completeness": round(comp_s, 1),
                "PK_Uniqueness": round(uniq_s, 1),
            },
            "recommendation": rec,
        }
    return scores


def gen_contracts(schema, quality):
    contracts = {}
    for tbl, q in quality.items():
        cols = q.get("columns", {})
        pk = schema["tables"].get(
            tbl, {}
        ).get("primary_keys", [])
        contract = {
            "version": "1.0",
            "generated": datetime.now().isoformat(),
            "table": tbl,
            "sla": {
                "min_completeness": round(
                    q.get("overall_completeness", 0.8), 3
                ),
            },
            "columns": {},
        }
        for cn, cd in cols.items():
            cc = {
                "dtype": cd.get("dtype", ""),
                "max_null_rate": round(min(
                    0.99,
                    cd.get("null_rate", 0) * 1.1 + 0.005
                ), 4),
            }
            if cn in pk:
                cc.update({
                    "unique": True, "not_null": True,
                    "max_null_rate": 0.0,
                })
            if cd.get("min_value") is not None:
                cc["range"] = [
                    cd["min_value"], cd["max_value"]
                ]
            contract["columns"][cn] = cc
        contracts[tbl] = yaml.dump(
            contract,
            default_flow_style=False,
            sort_keys=False,
        )
    return contracts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ER VIZ (Plotly fallback + Mermaid)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_er(schema):
    G = nx.DiGraph()
    edges = []
    for n, t in schema["tables"].items():
        G.add_node(
            n, row_count=t["row_count"],
            col_count=len(t["columns"]),
        )
    for tn, t in schema["tables"].items():
        for fk in t["foreign_keys"]:
            ref = fk.get("referred_table", "")
            if ref and ref in schema["tables"]:
                fc = (
                    fk["constrained_columns"][0]
                    if fk["constrained_columns"] else ""
                )
                tc = (
                    fk["referred_columns"][0]
                    if fk["referred_columns"] else ""
                )
                G.add_edge(
                    tn, ref,
                    from_col=fc, to_col=tc,
                )
                edges.append({
                    "from": tn, "fk_col": fc,
                    "to": ref, "pk_col": tc,
                })
    return G, edges


def render_er_plotly(G, schema, health):
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No tables", showarrow=False
        )
        return fig
    pos = nx.spring_layout(
        G, seed=42, k=4.0, iterations=80
    )
    fig = go.Figure()
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=2.5, color="rgba(108,92,231,0.6)"),
            hoverinfo="none", showlegend=False,
        ))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_trace(go.Scatter(
            x=[mx], y=[my], mode="text",
            text=[
                f"{data.get('from_col', '')} -> "
                f"{data.get('to_col', '')}"
            ],
            textfont=dict(size=9, color="#a29bfe"),
            hoverinfo="none", showlegend=False,
        ))
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3,
            arrowsize=1.5, arrowwidth=2,
            arrowcolor="rgba(108,92,231,0.7)",
        )
    for node in G.nodes():
        x, y = pos[node]
        t = schema["tables"][node]
        h = health.get(node, {})
        color = (
            "#00b894" if h.get("score", 0) >= 75
            else "#fdcb6e" if h.get("score", 0) >= 50
            else "#e17055"
        )
        cols = [c["name"] for c in t["columns"]]
        pk_cols = t["primary_keys"]
        col_text = "<br>".join([
            f"{'🔑 ' if c in pk_cols else '  '}{c}"
            for c in cols[:12]
        ])
        hover = (
            f"<b>{node}</b><br>"
            f"Rows: {t['row_count']:,}<br>"
            f"Health: {h.get('icon', '')} "
            f"{h.get('score', '?')}/100<br>"
            f"<hr>{col_text}"
        )
        deg = G.degree(node)
        size = max(35, min(70, deg * 15 + 35))
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(
                size=size, color=color,
                line=dict(width=3, color="white"),
                symbol="square", opacity=0.9,
            ),
            text=[f"{node}\n({t['row_count']:,})"],
            textposition="top center",
            textfont=dict(size=11, color="white"),
            hovertext=[hover], hoverinfo="text",
            showlegend=False,
        ))
    fig.update_layout(
        height=700,
        plot_bgcolor="#0a0a0f",
        paper_bgcolor="#0a0a0f",
        font_color="white",
        xaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False,
        ),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_md(schema, quality, health, ai_text):
    lines = [
        f"# Data Dictionary\n*{datetime.now():%Y-%m-%d %H:%M}*\n",
        (
            f"**{schema['total_tables']} tables | "
            f"{schema['total_columns']} columns | "
            f"{schema['total_rows']:,} rows**\n"
        ),
    ]
    if ai_text.get("schema"):
        lines.append(f"> {ai_text['schema']}\n")
    lines += [
        "| Table | Rows | Health |", "|-|-|-|"
    ]
    for n, t in schema["tables"].items():
        h = health.get(n, {})
        lines.append(
            f"| `{n}` | {t['row_count']:,} | "
            f"{h.get('icon', '')} {h.get('score', '-')} |"
        )
    for n, t in schema["tables"].items():
        h = health.get(n, {})
        lines += [
            f"\n## `{n}` {h.get('icon', '')}\n",
            (
                f"Rows: {t['row_count']:,} | "
                f"PKs: {', '.join(t['primary_keys']) or 'None'}\n"
            ),
            "| Column | Type | PK | FK | Null% |",
            "|-|-|-|-|-|",
        ]
        qc = quality.get(n, {}).get("columns", {})
        for col in t["columns"]:
            nr = qc.get(
                col["name"], {}
            ).get("null_rate", 0)
            fk = (
                f"->{col['ref_table']}.{col['ref_col']}"
                if col["is_fk"] else ""
            )
            lines.append(
                f"| `{col['name']}` | `{col['type']}` | "
                f"{'Y' if col['is_pk'] else ''} | "
                f"{fk} | {nr:.1%} |"
            )
    return "\n".join(lines)


def build_zip(contracts, json_str, md, pdf_bytes=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.json", json_str)
        z.writestr("dictionary.md", md)
        if pdf_bytes:
            z.writestr("report.pdf", pdf_bytes)
        for t, y in contracts.items():
            z.writestr(f"contracts/{t}.yaml", y)
    buf.seek(0)
    return buf.read()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_analysis(engine, provider, model, n_rows):
    bar = st.progress(0)
    status = st.empty()

    status.info("🔍 Extracting schema...")
    bar.progress(10)
    schema = extract_schema(engine)

    status.info(f"📊 Sampling {n_rows:,} rows...")
    bar.progress(25)
    samples = sample_tables(engine, schema, n_rows)

    status.info("🔬 Profiling quality...")
    bar.progress(40)
    quality = profile_all(schema, samples)

    status.info("🔗 Building ER diagram...")
    bar.progress(50)
    G, edges = build_er(schema)
    mermaid_code = build_mermaid_er_clean(schema)

    status.info("⭐ Causal discovery...")
    bar.progress(60)
    causal = find_causal(samples)

    status.info("🏥 Health scores...")
    bar.progress(70)
    health = compute_health(schema, quality)

    status.info("📄 Contracts...")
    bar.progress(78)
    contracts = gen_contracts(schema, quality)

    status.info("🤖 AI analysis...")
    bar.progress(84)
    ai = AI(provider, model)
    ai_text = {}
    tbl_info = {
        n: {
            "rows": t["row_count"],
            "cols": len(t["columns"]),
        }
        for n, t in schema["tables"].items()
    }
    ai_text["schema"] = ai.ask(
        "Senior data architect. 3 sentences: "
        "business domain, key tables, patterns.\n"
        f"Tables: {json.dumps(tbl_info)}\n"
        f"Relationships: {len(edges)} FKs", 400
    )
    ai_text["causal"] = (
        ai.ask(
            "3 sentences on business meaning:\n"
            f"{json.dumps(causal[:8])}", 300
        )
        if causal
        else "No significant causal relationships."
    )
    ai_text["tables"] = {}
    for tn, t in schema["tables"].items():
        col_list = [
            {"name": c["name"], "type": c["type"]}
            for c in t["columns"][:15]
        ]
        ai_text["tables"][tn] = ai.ask(
            f"Describe '{tn}' ({t['row_count']:,} rows)"
            " in 2 sentences. "
            f"Columns: {json.dumps(col_list)}", 300
        )

    status.info("📕 Building PDF (WeasyPrint)...")
    bar.progress(90)
    try:
        pdf_bytes = build_pdf_weasy(
            schema, quality, health, causal,
            edges, ai_text, contracts,
        )
    except Exception as e:
        st.warning(f"PDF generation issue: {e}")
        pdf_bytes = None

    status.info("📦 Packaging...")
    bar.progress(95)
    er_fig = render_er_plotly(G, schema, health)
    md = build_md(schema, quality, health, ai_text)
    jr = json.dumps({
        "schema": schema, "quality": quality,
        "causal": causal, "health": health,
        "er_edges": edges,
    }, indent=2, default=str)
    zb = build_zip(contracts, jr, md, pdf_bytes)

    bar.progress(100)
    status.success("✅ Complete!")

    return {
        "schema": schema, "quality": quality,
        "G": G, "edges": edges,
        "causal": causal, "health": health,
        "contracts": contracts, "ai_text": ai_text,
        "md": md, "json_report": jr, "zip": zb,
        "samples": samples, "pdf_bytes": pdf_bytes,
        "er_fig": er_fig, "mermaid": mermaid_code,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHATBOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_chatbot(R):
    st.markdown("### 💬 AI Data Assistant")
    st.caption("Ask about your data using actual analysis.")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    with st.expander("💡 Examples", expanded=False):
        for i, q in enumerate([
            "Worst quality tables?",
            "Most null columns?",
            "Causal findings?",
            "Health summary?",
            "What to fix first?",
            "Table relationships?",
        ]):
            if st.button(q, key=f"ex_{i}",
                         use_container_width=True):
                st.session_state.chat_q = q

    for msg in st.session_state.chat:
        cls = (
            "chat-user" if msg["role"] == "user"
            else "chat-ai"
        )
        who = "You" if msg["role"] == "user" else "AI"
        st.markdown(
            f'<div class="chat-msg {cls}">'
            f'<b>{who}:</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    default = st.session_state.pop("chat_q", "")
    inp = st.chat_input("Ask about your data...")
    if default:
        inp = default
    if inp:
        st.session_state.chat.append({
            "role": "user", "content": inp
        })
        schema = R["schema"]
        tbl_sums = [
            f"- {tn}: {t['row_count']:,} rows, "
            f"health={R['health'].get(tn, {}).get('score', '?')}"
            for tn, t in schema["tables"].items()
        ]
        sample_ctx = ""
        for tn, df in R.get("samples", {}).items():
            if not df.empty:
                sample_ctx += (
                    f"\n{tn}:\n{df.head(3).to_string()}\n"
                )
        ctx = (
            f"Expert analyst. Use ONLY this data:\n"
            f"DB: {schema['total_tables']} tables, "
            f"{schema['total_rows']:,} rows\n"
            f"TABLES:\n{chr(10).join(tbl_sums)}\n"
            f"FKs: {json.dumps(R['edges'][:15])}\n"
            f"CAUSAL: {json.dumps(R['causal'][:10])}\n"
            f"SAMPLES:\n{sample_ctx[:3000]}\n"
            f"Be specific."
        )
        prov = st.session_state.get(
            "ai_provider",
            os.getenv("DEFAULT_AI_PROVIDER", "groq"),
        )
        mod = st.session_state.get(
            "ai_model",
            os.getenv(
                "DEFAULT_AI_MODEL",
                "llama-3.3-70b-versatile"
            ),
        )
        with st.spinner("Thinking..."):
            resp = AI(prov, mod).ask(
                f"{ctx}\n\nQ: {inp}", 1500
            )
        st.session_state.chat.append({
            "role": "ai", "content": resp
        })
        st.rerun()
    if st.session_state.chat:
        if st.button("Clear chat",
                     use_container_width=True):
            st.session_state.chat = []
            st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2 style="background: linear-gradient(
            135deg, #6c5ce7, #00cec9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;">
            🧠 DB Intelligence
        </h2>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 🤖 AI Engine")
    avail = AI.available()
    if avail:
        labels = {
            f"{v['icon']} {v['name']}": k
            for k, v in avail.items()
        }
        dp = os.getenv("DEFAULT_AI_PROVIDER", "groq")
        dl = next(
            (l for l, k in labels.items() if k == dp),
            list(labels.keys())[0],
        )
        sl = st.selectbox(
            "Provider", list(labels.keys()),
            index=list(labels.keys()).index(dl),
            label_visibility="collapsed",
        )
        sp = labels[sl]
        models = AI.PROVIDERS[sp]["models"]
        dm = os.getenv("DEFAULT_AI_MODEL", models[0])
        sm = st.selectbox(
            "Model", models,
            index=models.index(dm) if dm in models else 0,
            label_visibility="collapsed",
        )
        st.session_state["ai_provider"] = sp
        st.session_state["ai_model"] = sm
        st.success(f"✅ {sl} / `{sm}`")
    else:
        st.error("No API keys in .env!")
    st.divider()

    st.markdown("#### 📂 Data")
    mode = st.radio(
        "Source",
        ["SQLite", "CSV Files", "DB URL"],
        label_visibility="collapsed",
    )
    engine = None
    ready = False

    if mode == "SQLite":
        f = st.file_uploader(
            ".db", type=["db", "sqlite", "sqlite3"],
            label_visibility="collapsed",
        )
        if f:
            try:
                engine = load_sqlite(f)
                if test_db(engine):
                    st.success(f"✅ {f.name}")
                    ready = True
            except Exception as e:
                st.error(str(e))
    elif mode == "CSV Files":
        files = st.file_uploader(
            "CSVs", type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if files:
            try:
                engine, names = load_csvs(files)
                if test_db(engine) and names:
                    st.success(f"✅ {len(names)} tables")
                    ready = True
            except Exception as e:
                st.error(str(e))
    else:
        url = st.text_input(
            "URL",
            placeholder="postgresql://user:pass@host/db",
        )
        if url:
            try:
                engine = create_engine(url)
                if test_db(engine):
                    st.success("✅ Connected")
                    ready = True
            except Exception as e:
                st.error(str(e))

    st.divider()
    n_rows = st.slider("Sample rows", 500, 50000, 5000, 500)
    run = st.button(
        "🚀 Analyze", type="primary",
        use_container_width=True, disabled=not ready,
    )
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if run and engine and ready:
    try:
        prov = st.session_state.get(
            "ai_provider",
            os.getenv("DEFAULT_AI_PROVIDER", "groq"),
        )
        mod = st.session_state.get(
            "ai_model",
            os.getenv(
                "DEFAULT_AI_MODEL",
                "llama-3.3-70b-versatile",
            ),
        )
        st.session_state["R"] = run_analysis(
            engine, prov, mod, n_rows
        )
    except Exception as e:
        st.error(f"Failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WELCOME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "R" not in st.session_state:
    st.markdown("""
    <div class="hero">
        <h1>🧠 DB Intelligence Agent</h1>
        <p>Upload a database → Get AI-powered analysis,
        Mermaid ER diagrams, causal discovery,
        and professional PDF reports.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (ic, ti, de) in zip(
        [c1, c2, c3, c4],
        [
            ("📋", "Schema", "Tables, PKs, FKs auto-detected"),
            ("🔗", "Mermaid ER", "Interactive ER diagrams"),
            ("⭐", "Causal AI", "MI & Chi² discovery"),
            ("📕", "PDF Report", "WeasyPrint professional PDF"),
        ],
    ):
        col.markdown(
            f"""<div class="glass-card"
                 style="text-align:center; min-height:180px;">
                <div style="font-size:2.5rem;">{ic}</div>
                <h3 style="color:white;">{ti}</h3>
                <p style="color:{COLORS['muted']};
                   font-size:0.85rem;">{de}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R = st.session_state["R"]
S, Q, H, C = (
    R["schema"], R["quality"],
    R["health"], R["causal"],
)

st.markdown("<br>", unsafe_allow_html=True)
mcols = st.columns(6)
avg_h = (
    f"{np.mean([h['score'] for h in H.values()]):.0f}/100"
    if H else "—"
)
for col, (ic, lb, vl) in zip(mcols, [
    ("📦", "Tables", S["total_tables"]),
    ("🔢", "Columns", S["total_columns"]),
    ("📊", "Rows", f"{S['total_rows']:,}"),
    ("🔗", "Relations", len(R["edges"])),
    ("⭐", "Causal", len(C)),
    ("💊", "Health", avg_h),
]):
    col.markdown(
        f"""<div class="metric-card">
            <div style="font-size:1.5rem;">{ic}</div>
            <div class="metric-value">{vl}</div>
            <div class="metric-label">{lb}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""<div class="glass-card">
        <h4 style="color:{COLORS['accent']};">
            🤖 AI Summary
        </h4>
        <p style="color:{COLORS['text']};">
            {R['ai_text'].get('schema', '')}
        </p>
    </div>""",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# TABS
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📋 Schema", "🔗 ER Diagram", "💊 Health",
    "⭐ Causal", "💬 AI Chat", "📖 Dictionary",
    "⬇️ Downloads",
])

# ═══ TAB 1 — SCHEMA ═══
with t1:
    st.subheader("Schema Overview")
    rows = []
    for n, t in S["tables"].items():
        h = H.get(n, {})
        q = Q.get(n, {})
        rows.append({
            "Table": n,
            "Rows": f"{t['row_count']:,}",
            "Cols": len(t["columns"]),
            "PKs": ", ".join(t["primary_keys"]) or "—",
            "FKs": len(t["foreign_keys"]),
            "Complete": (
                f"{q.get('overall_completeness', 0):.1%}"
            ),
            "Health": (
                f"{h.get('icon', '')} "
                f"{h.get('label', '')} "
                f"({h.get('score', '—')})"
            ),
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True, hide_index=True,
    )
    for n, t in S["tables"].items():
        h = H.get(n, {})
        with st.expander(
            f"**{n}** · {t['row_count']:,} rows · "
            f"{h.get('icon', '')} {h.get('score', '')}"
        ):
            ai_d = R["ai_text"].get("tables", {}).get(n, "")
            if ai_d:
                st.info(ai_d)
            if h.get("recommendation"):
                st.warning(h["recommendation"])
            qc = Q.get(n, {}).get("columns", {})
            cr = [{
                "Column": c["name"],
                "Type": c["type"],
                "PK": "🔑" if c["is_pk"] else "",
                "FK": (
                    f"→{c['ref_table']}.{c['ref_col']}"
                    if c["is_fk"] else ""
                ),
                "Null%": (
                    f"{qc.get(c['name'], {}).get('null_rate', 0):.1%}"
                ),
                "Kind": qc.get(
                    c["name"], {}
                ).get("col_kind", "—"),
            } for c in t["columns"]]
            st.dataframe(
                pd.DataFrame(cr),
                use_container_width=True,
                hide_index=True,
            )


# ═══ TAB 2 — ER DIAGRAM (MERMAID + PLOTLY) ═══
with t2:
    st.subheader("🔗 Entity Relationship Diagram")

    er_view = st.radio(
        "View",
        ["Mermaid (Code)", "Interactive (Plotly)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if er_view == "Mermaid (Code)":
        st.caption(
            "📋 Mermaid ER Diagram — "
            "Copy & paste into mermaid.live or "
            "any Mermaid renderer"
        )

        mermaid_code = R.get("mermaid", "")

        # Render using streamlit's built-in mermaid
        # (st >= 1.33)
        try:
            st.markdown(
                f"```mermaid\n{mermaid_code}\n```"
            )
        except Exception:
            st.code(mermaid_code, language="text")

        # Copyable code
        with st.expander(
            "📋 Copy Mermaid Code", expanded=False
        ):
            st.code(mermaid_code, language="text")
            st.markdown(
                "[Open in Mermaid Live Editor →]"
                "(https://mermaid.live)"
            )

    else:
        st.caption(
            "Interactive Plotly diagram — "
            "hover for details"
        )
        st.plotly_chart(
            R["er_fig"], use_container_width=True
        )

    c1, c2 = st.columns(2)
    with c1:
        if R["edges"]:
            st.subheader("FK Relationships")
            st.dataframe(
                pd.DataFrame(R["edges"]),
                use_container_width=True,
                hide_index=True,
            )
    with c2:
        if R["G"].nodes():
            st.subheader("Connectivity")
            cent = nx.degree_centrality(R["G"])
            hub_df = pd.DataFrame([
                {"Table": t, "Centrality": round(c, 3)}
                for t, c in sorted(
                    cent.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ])
            fig = px.bar(
                hub_df, x="Centrality", y="Table",
                orientation="h", color="Centrality",
                color_continuous_scale=[
                    "#6c5ce7", "#00cec9"
                ],
                template="plotly_dark",
            )
            fig.update_layout(
                height=max(250, len(hub_df) * 40),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#0a0a0f",
                paper_bgcolor="#0a0a0f",
            )
            st.plotly_chart(
                fig, use_container_width=True
            )


# ═══ TAB 3 — HEALTH ═══
with t3:
    st.subheader("Health Scores")
    if H:
        hdf = pd.DataFrame(H.values()).sort_values("score")
        fig = px.bar(
            hdf, x="score", y="table",
            orientation="h", color="label",
            text="score",
            color_discrete_map={
                "Healthy": "#00b894",
                "At Risk": "#fdcb6e",
                "Critical": "#e17055",
            },
            template="plotly_dark",
        )
        fig.update_traces(
            texttemplate="%{text:.0f}",
            textposition="outside",
        )
        fig.update_layout(
            height=max(250, len(H) * 50),
            margin=dict(l=10, r=80, t=10, b=10),
            plot_bgcolor="#0a0a0f",
            paper_bgcolor="#0a0a0f",
        )
        st.plotly_chart(fig, use_container_width=True)
        for h in sorted(
            H.values(), key=lambda x: x["score"]
        ):
            with st.expander(
                f"{h['icon']} {h['table']} — "
                f"{h['score']}/100"
            ):
                bc = st.columns(len(h["breakdown"]))
                for c, (k, v) in zip(
                    bc, h["breakdown"].items()
                ):
                    c.metric(k, f"{v:.0f}/100")
                st.warning(h["recommendation"])

    st.divider()
    st.subheader("Column Inspector")
    sel = st.selectbox("Table", list(Q.keys()), key="qt")
    if sel and sel in Q:
        tq = Q[sel]
        cr = [{
            "Column": cn,
            "Kind": cd.get("col_kind", "—"),
            "Nulls": cd.get("null_count", 0),
            "Null%": f"{cd.get('null_rate', 0):.2%}",
            "Min": cd.get("min_value"),
            "Max": cd.get("max_value"),
            "Mean": cd.get("mean_value"),
        } for cn, cd in tq.get("columns", {}).items()]
        st.dataframe(
            pd.DataFrame(cr),
            use_container_width=True, hide_index=True,
        )
        la, lb = st.columns(2)
        with la:
            nd = [
                {
                    "Col": r["Column"],
                    "Null": float(r["Null%"].strip("%")),
                }
                for r in cr
                if float(r["Null%"].strip("%")) > 0
            ]
            if nd:
                fig = px.bar(
                    pd.DataFrame(nd).sort_values("Null"),
                    x="Null", y="Col", orientation="h",
                    color="Null",
                    color_continuous_scale="Reds",
                    template="plotly_dark",
                )
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="#0a0a0f",
                    paper_bgcolor="#0a0a0f",
                )
                st.plotly_chart(
                    fig, use_container_width=True
                )
            else:
                st.success("Zero nulls!")
        with lb:
            cats = [
                cn for cn, cd
                in tq.get("columns", {}).items()
                if cd.get("col_kind") == "categorical"
            ]
            if cats:
                sc = st.selectbox(
                    "Category", cats, key="tv"
                )
                tv = tq["columns"].get(
                    sc, {}
                ).get("top_values", [])
                if tv:
                    fig = px.bar(
                        pd.DataFrame(tv).head(10),
                        x="count", y="value",
                        orientation="h", color="count",
                        color_continuous_scale=[
                            "#6c5ce7", "#00cec9"
                        ],
                        template="plotly_dark",
                    )
                    fig.update_layout(
                        height=350,
                        margin=dict(
                            l=10, r=10, t=10, b=10
                        ),
                        plot_bgcolor="#0a0a0f",
                        paper_bgcolor="#0a0a0f",
                    )
                    st.plotly_chart(
                        fig, use_container_width=True
                    )


# ═══ TAB 4 — CAUSAL ═══
with t4:
    st.markdown(
        f"""<div class="glass-card">
            <h4 style="color:{COLORS['accent']};">
                ⭐ Causal Intelligence
            </h4>
            <p style="color:{COLORS['text']};">
                MI & Chi-Squared directional discovery.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.info(R["ai_text"].get("causal", ""))
    if C:
        cdf = pd.DataFrame(C)
        top = cdf.head(20).copy()
        top["label"] = (
            top["from"] + " " + top["direction"]
            + " " + top["to"]
        )
        fig = px.bar(
            top, x="strength", y="label",
            color="method", orientation="h",
            template="plotly_dark",
            color_discrete_map={
                "mutual_info": "#6c5ce7",
                "chi_squared": "#00cec9",
                "mi_classif": "#fdcb6e",
            },
            hover_data=["p_value", "insight"],
        )
        fig.update_layout(
            height=max(350, len(top) * 30),
            margin=dict(l=10, r=30, t=10, b=10),
            plot_bgcolor="#0a0a0f",
            paper_bgcolor="#0a0a0f",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            cdf[[
                "from", "direction", "to",
                "method", "strength", "p_value",
                "insight",
            ]],
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("Data Contracts")
    for tn, ys in R["contracts"].items():
        h = H.get(tn, {})
        with st.expander(f"{tn} {h.get('icon', '')}"):
            c1, c2 = st.columns([6, 1])
            c1.code(ys, language="yaml")
            c2.download_button(
                "⬇️", ys, f"{tn}.yaml",
                "text/yaml", key=f"c_{tn}",
                use_container_width=True,
            )


# ═══ TAB 5 — CHAT ═══
with t5:
    render_chatbot(R)


# ═══ TAB 6 — DICTIONARY ═══
with t6:
    st.subheader("Data Dictionary")
    st.markdown(R["md"])
    st.download_button(
        "⬇️ .md", R["md"],
        "dictionary.md", "text/markdown",
    )


# ═══ TAB 7 — DOWNLOADS ═══
with t7:
    st.subheader("⬇️ Downloads")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.markdown(
            f"""<div class="glass-card"
                 style="text-align:center;">
                <div style="font-size:2rem;">📦</div>
                <h4 style="color:white;">ZIP</h4>
                <p style="color:{COLORS['muted']};">
                    All reports
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.download_button(
            "📦 Everything",
            R["zip"],
            f"db_{datetime.now():%Y%m%d_%H%M}.zip",
            "application/zip",
            type="primary",
            use_container_width=True,
        )

    with d2:
        st.markdown(
            f"""<div class="glass-card"
                 style="text-align:center;">
                <div style="font-size:2rem;">📕</div>
                <h4 style="color:white;">PDF</h4>
                <p style="color:{COLORS['muted']};">
                    WeasyPrint report
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        if R.get("pdf_bytes"):
            st.download_button(
                "📕 PDF Report",
                R["pdf_bytes"],
                f"report_{datetime.now():%Y%m%d}.pdf",
                "application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("PDF unavailable")

    with d3:
        st.markdown(
            f"""<div class="glass-card"
                 style="text-align:center;">
                <div style="font-size:2rem;">📖</div>
                <h4 style="color:white;">Dictionary</h4>
            </div>""",
            unsafe_allow_html=True,
        )
        st.download_button(
            "📖 Markdown",
            R["md"], "dictionary.md",
            "text/markdown",
            use_container_width=True,
        )

    with d4:
        st.markdown(
            f"""<div class="glass-card"
                 style="text-align:center;">
                <div style="font-size:2rem;">📄</div>
                <h4 style="color:white;">JSON</h4>
            </div>""",
            unsafe_allow_html=True,
        )
        st.download_button(
            "📄 JSON Report",
            R["json_report"], "report.json",
            "application/json",
            use_container_width=True,
        )

    # PDF Preview
    if R.get("pdf_bytes"):
        st.divider()
        st.subheader("📕 PDF Preview")
        b64 = base64.b64encode(
            R["pdf_bytes"]
        ).decode()
        st.markdown(
            f"""<iframe
                src="data:application/pdf;base64,{b64}"
                width="100%" height="600"
                style="border:1px solid
                rgba(108,92,231,0.3);
                border-radius:12px;">
            </iframe>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Contracts")
    cc = st.columns(min(4, max(1, len(R["contracts"]))))
    for i, (tn, ys) in enumerate(R["contracts"].items()):
        h = H.get(tn, {})
        cc[i % len(cc)].download_button(
            f"{h.get('icon', '')} {tn}",
            ys, f"{tn}.yaml", "text/yaml",
            key=f"dl_{tn}",
            use_container_width=True,
        )
