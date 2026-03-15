# Run: streamlit run dashboard.py
# Upload SQLite .db file OR multiple CSV files → get full analysis

from __future__ import annotations
import io, json, os, sys, tempfile, warnings, zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# Import the multi_ai module
from multi_ai import MultiAIProvider

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sqlalchemy import create_engine, inspect, text

# ─────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DB Intelligence Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# SECTION 1 — CONNECT TO DATABASE
# ─────────────────────────────────────────────────────────────
def connect_sqlite(uploaded_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return create_engine(f"sqlite:///{tmp.name}"), tmp.name

def connect_csvs(csv_files):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    engine = create_engine(f"sqlite:///{tmp.name}")
    loaded = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            tbl = os.path.splitext(f.name)[0].lower().replace(" ","_").replace("-","_")
            df.to_sql(tbl, engine, if_exists="replace", index=False)
            loaded.append(tbl)
        except Exception as e:
            st.warning(f"Could not load {f.name}: {e}")
    return engine, tmp.name, loaded

def connect_url(url: str):
    return create_engine(url), url

def test_engine(engine) -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────
# SECTION 2 — SCHEMA EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_schema(engine) -> Dict:
    insp = inspect(engine)
    tables = {}
    for tbl in insp.get_table_names():
        try:
            cols_raw = insp.get_columns(tbl)
            pk_info  = insp.get_pk_constraint(tbl)
            fks_raw  = insp.get_foreign_keys(tbl)
            indexes  = insp.get_indexes(tbl)
        except Exception:
            continue

        pk_cols = pk_info.get("constrained_columns", [])
        fk_map  = {}
        fk_list = []
        for fk in fks_raw:
            for col, rc in zip(fk.get("constrained_columns",[]), fk.get("referred_columns",[])):
                fk_map[col] = {"ref_table": fk.get("referred_table",""), "ref_col": rc}
            fk_list.append({
                "constrained_columns": fk.get("constrained_columns",[]),
                "referred_table":      fk.get("referred_table",""),
                "referred_columns":    fk.get("referred_columns",[]),
            })

        try:
            with engine.connect() as c:
                row_count = c.execute(text(f'SELECT COUNT(*) FROM "{tbl}"')).scalar() or 0
        except Exception:
            row_count = 0

        columns = []
        for col in cols_raw:
            n = col["name"]
            fi = fk_map.get(n, {})
            columns.append({
                "name":      n,
                "type":      str(col.get("type","TEXT")),
                "nullable":  bool(col.get("nullable", True)),
                "default":   str(col.get("default","")) or None,
                "is_pk":     n in pk_cols,
                "is_fk":     n in fk_map,
                "ref_table": fi.get("ref_table",""),
                "ref_col":   fi.get("ref_col",""),
            })

        tables[tbl] = {
            "name":         tbl,
            "row_count":    row_count,
            "columns":      columns,
            "primary_keys": pk_cols,
            "foreign_keys": fk_list,
            "index_count":  len(indexes),
        }

    # Heuristic FK detection for CSV-sourced databases
    total_fks = sum(len(t["foreign_keys"]) for t in tables.values())
    if total_fks == 0:
        tables = _heuristic_fk(tables)

    return {
        "tables":        tables,
        "total_tables":  len(tables),
        "total_columns": sum(len(t["columns"]) for t in tables.values()),
        "total_rows":    sum(t["row_count"] for t in tables.values()),
    }

def _heuristic_fk(tables: Dict) -> Dict:
    """Detect FK relationships from column naming patterns (e.g. customer_id → customers)."""
    tbl_names = list(tables.keys())
    for tbl_name, tbl in tables.items():
        for col in tbl["columns"]:
            n = col["name"]
            if not n.endswith("_id") or col["is_pk"]:
                continue
            base = n[:-3]
            for candidate in [base, base+"s", base+"es"]:
                if candidate in tbl_names and candidate != tbl_name:
                    ref_cols = [c["name"] for c in tables[candidate]["columns"]]
                    rc = n if n in ref_cols else ("id" if "id" in ref_cols else None)
                    if rc:
                        col["is_fk"] = True
                        col["ref_table"] = candidate
                        col["ref_col"] = rc
                        tables[tbl_name]["foreign_keys"].append({
                            "constrained_columns": [n],
                            "referred_table":      candidate,
                            "referred_columns":    [rc],
                        })
                        break
    return tables

# ─────────────────────────────────────────────────────────────
# SECTION 3 — DATA SAMPLING
# ─────────────────────────────────────────────────────────────
def sample_all(engine, schema: Dict, n: int) -> Dict[str, pd.DataFrame]:
    samples = {}
    for tbl in schema["tables"]:
        try:
            with engine.connect() as c:
                samples[tbl] = pd.read_sql(
                    text(f'SELECT * FROM "{tbl}" LIMIT :n'), c, params={"n": n})
        except Exception:
            samples[tbl] = pd.DataFrame()
    return samples

# ─────────────────────────────────────────────────────────────
# SECTION 4 — QUALITY PROFILING
# ─────────────────────────────────────────────────────────────
def profile_quality(schema: Dict, samples: Dict) -> Dict:
    result = {}
    for tbl, df in samples.items():
        if df is None or df.empty:
            result[tbl] = {"table_name": tbl, "sampled_rows": 0, "column_count": 0,
                           "overall_completeness": 0.0, "worst_columns": [], "columns": {}}
            continue

        col_profiles = {}
        for col in df.columns:
            try:
                col_profiles[col] = _profile_col(df[col])
            except Exception:
                col_profiles[col] = {"dtype": "unknown", "null_rate": 0.0,
                                     "completeness": 1.0, "col_kind": "error",
                                     "null_count": 0, "unique_count": 0,
                                     "unique_rate": 0.0, "top_values": [],
                                     "min_value": None, "max_value": None,
                                     "mean_value": None, "std_value": None,
                                     "freshness_days": None, "is_fresh": None,
                                     "total_count": len(df)}

        vals    = [v["completeness"] for v in col_profiles.values()]
        overall = round(float(np.mean(vals)), 4) if vals else 0.0
        worst   = sorted(col_profiles.items(),
                         key=lambda x: x[1].get("null_rate", 0), reverse=True)

        result[tbl] = {
            "table_name":           tbl,
            "sampled_rows":         len(df),
            "column_count":         len(df.columns),
            "overall_completeness": overall,
            "worst_columns":        [w[0] for w in worst[:3]
                                     if w[1].get("null_rate", 0) > 0],
            "columns":              col_profiles,
        }
    return result

def _profile_col(s: pd.Series) -> Dict:
    base = {
        "dtype": str(s.dtype), "total_count": len(s), "null_count": 0,
        "null_rate": 0.0, "completeness": 1.0, "unique_count": 0,
        "unique_rate": 0.0, "min_value": None, "max_value": None,
        "mean_value": None, "std_value": None, "top_values": [],
        "freshness_days": None, "is_fresh": None, "col_kind": "unknown"
    }
    try:
        total      = len(s)
        null_count = int(s.isna().sum())
        null_rate  = round(null_count / total, 6) if total else 0.0
        completeness  = round(1.0 - null_rate, 6)
        unique_count  = int(s.nunique(dropna=True))
        unique_rate   = round(unique_count / max(total - null_count, 1), 6)

        p = {**base, "null_count": null_count, "null_rate": null_rate,
             "completeness": completeness, "unique_count": unique_count,
             "unique_rate": unique_rate}

        clean = s.dropna()
        if clean.empty:
            return p

        if pd.api.types.is_numeric_dtype(s):
            p.update({"col_kind": "numeric",
                      "min_value": _sf(clean.min()), "max_value": _sf(clean.max()),
                      "mean_value": _sf(clean.mean()), "std_value": _sf(clean.std())})

        elif pd.api.types.is_datetime64_any_dtype(s):
            p.update(_dt_stats(clean))
            p["col_kind"] = "datetime"

        else:
            # Safe string conversion — handles bytes and binary data
            def _safe_str(v):
                try:
                    return str(v)
                except Exception:
                    return None

            str_vals = [_safe_str(v) for v in clean]
            str_vals = [v for v in str_vals if v is not None]

            if not str_vals:
                return p

            if _is_datetime_str(clean):
                try:
                    parsed = pd.to_datetime(clean, errors="coerce")
                    parsed = parsed[parsed.notna()]
                    if not parsed.empty:
                        p.update(_dt_stats(parsed))
                        p["col_kind"] = "datetime_string"
                        return p
                except Exception:
                    pass

            try:
                vc = pd.Series(str_vals).str[:80].value_counts().head(10)
                p["top_values"] = [{"value": str(k), "count": int(v)}
                                   for k, v in vc.items()]
            except Exception:
                p["top_values"] = []

            p["col_kind"] = "categorical"

        return p

    except Exception:
        base["col_kind"] = "error"
        return base

def _dt_stats(s: pd.Series) -> Dict:
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce").dropna()
    if s.empty:
        return {}
    try:
        max_naive = s.max().to_pydatetime().replace(tzinfo=None)
    except Exception:
        max_naive = datetime.now()
    days = (datetime.now() - max_naive).days
    return {"min_value":str(s.min()),"max_value":str(s.max()),
            "freshness_days":days,"is_fresh":days <= 30}

def _is_datetime_str(s: pd.Series) -> bool:
    try:
        sample = s.dropna().head(20)
        str_vals = []
        for v in sample:
            try:
                str_vals.append(str(v))
            except Exception:
                return False
        if not str_vals:
            return False
        hits = sum(
            1 for v in str_vals
            if isinstance(v, str)
            and any(p in v for p in ["-", "/", "T", ":"])
            and len(v) >= 8
        )
        return hits > len(str_vals) * 0.6
    except Exception:
        return False

def _sf(val) -> Optional[float]:
    try:
        f = float(val)
        return round(f, 4) if np.isfinite(f) else None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# SECTION 5 — ER MAP
# ─────────────────────────────────────────────────────────────
def build_er(schema: Dict):
    G = nx.DiGraph()
    tables = schema["tables"]
    for n, t in tables.items():
        G.add_node(n, row_count=t["row_count"], col_count=len(t["columns"]))
    edges = []
    for tbl_name, tbl in tables.items():
        for fk in tbl["foreign_keys"]:
            ref = fk.get("referred_table","")
            if not ref or ref not in tables:
                continue
            fc = fk["constrained_columns"][0] if fk["constrained_columns"] else ""
            tc = fk["referred_columns"][0]    if fk["referred_columns"]    else ""
            G.add_edge(tbl_name, ref, from_col=fc, to_col=tc)
            edges.append({"From Table":tbl_name,"FK Column":fc,
                          "To Table":ref,"PK Column":tc,"Cardinality":"many-to-one"})
    centrality = nx.degree_centrality(G) if G.nodes else {}
    hubs = sorted(centrality.items(), key=lambda x:x[1], reverse=True)
    return G, {
        "nodes":[{"table":n,"row_count":G.nodes[n].get("row_count",0),
                  "degree":G.degree(n)} for n in G.nodes],
        "edges":edges,
        "hub_tables":[{"table":t,"centrality":round(c,4)} for t,c in hubs],
        "total_relationships":G.number_of_edges(),
    }

def render_er_chart(G: nx.DiGraph) -> go.Figure:
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No tables found", showarrow=False)
        return fig
    pos = nx.spring_layout(G, seed=42, k=3.0)
    fig = go.Figure()
    for u, v, data in G.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        fig.add_trace(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",
            line=dict(width=2,color="#7c6ff7"),hoverinfo="none",showlegend=False))
        fig.add_trace(go.Scatter(x=[(x0+x1)/2],y=[(y0+y1)/2],mode="text",
            text=[f"{data.get('from_col','')}→{data.get('to_col','')}"],
            textfont=dict(size=9,color="#888"),hoverinfo="none",showlegend=False))

    nc = ["#e74c3c" if G.degree(n)>=3 else "#7c6ff7" for n in G.nodes()]
    ns = [max(22,min(50,G.degree(n)*12+22)) for n in G.nodes()]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        marker=dict(size=ns,color=nc,line=dict(width=2,color="white")),
        text=list(G.nodes()), textposition="top center",
        textfont=dict(size=11,color="white"),
        hovertext=[f"<b>{n}</b><br>Rows: {G.nodes[n].get('row_count',0):,}<br>Connections: {G.degree(n)}"
                   for n in G.nodes()],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(height=520,showlegend=False,
        plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",font_color="white",
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        margin=dict(l=20,r=20,t=20,b=20))
    return fig

# ─────────────────────────────────────────────────────────────
# SECTION 6 — ⭐ CAUSAL INTELLIGENCE ENGINE
# ─────────────────────────────────────────────────────────────
class CausalRel:
    def __init__(self,ft,fc,tt,tc,method,strength,pval,direction="→",insight=""):
        self.from_table=ft; self.from_col=fc; self.to_table=tt; self.to_col=tc
        self.method=method; self.strength=strength; self.p_value=pval
        self.direction=direction; self.insight=insight
    def to_dict(self):
        return {"from":f"{self.from_table}.{self.from_col}",
                "to":f"{self.to_table}.{self.to_col}",
                "method":self.method,"strength":round(self.strength,4),
                "p_value":round(self.p_value,4),"direction":self.direction,
                "insight":self.insight}

def discover_causal(samples: Dict, MI_MIN=0.05, CHI2_P=0.05, CRAMERS_MIN=0.10,
                    MAX_COLS=8, MAX_CATS=25) -> List[CausalRel]:
    rels = []
    for tbl, df in samples.items():
        if df is None or df.empty or len(df) < 50:
            continue
        num = df.select_dtypes(include=[np.number]).columns.tolist()[:MAX_COLS]
        cat = [c for c in df.select_dtypes(include=["object","category"]).columns
               if df[c].nunique() <= MAX_CATS][:MAX_COLS]

        # Numeric → Numeric
        if len(num) >= 2:
            for target in num[:6]:
                feats = [c for c in num if c != target][:5]
                if not feats: continue
                try:
                    X = df[feats].fillna(0).values
                    y = df[target].fillna(0).values
                    scores = mutual_info_regression(X, y, random_state=42)
                    for j, sc in enumerate(scores):
                        if sc >= MI_MIN:
                            lv = "strongly" if sc>0.3 else "moderately" if sc>0.1 else "weakly"
                            rels.append(CausalRel(tbl,feats[j],tbl,target,
                                "mutual_information",float(sc),max(0.001,1-float(sc)),"→",
                                f"'{feats[j]}' {lv} influences '{target}'."))
                except Exception:
                    pass

        # Categorical ↔ Categorical
        if len(cat) >= 2:
            for i in range(min(len(cat),6)):
                for j in range(i+1, min(len(cat),6)):
                    a, b = cat[i], cat[j]
                    try:
                        ct = pd.crosstab(df[a].fillna("_null_").astype(str).str[:50],
                                         df[b].fillna("_null_").astype(str).str[:50])
                        if ct.shape[0]<2 or ct.shape[1]<2: continue
                        chi2, p, _, _ = chi2_contingency(ct)
                        n = ct.values.sum()
                        md = min(ct.shape)-1
                        if md <= 0: continue
                        cv = float(np.sqrt(chi2/(n*md)))
                        if p < CHI2_P and cv >= CRAMERS_MIN:
                            lv = "strongly" if cv>0.3 else "moderately" if cv>0.1 else "weakly"
                            rels.append(CausalRel(tbl,a,tbl,b,
                                "chi_squared",cv,float(p),"↔",
                                f"'{a}' and '{b}' are {lv} associated."))
                    except Exception:
                        pass

        # Numeric → Categorical
        if num and cat:
            for cc in cat[:4]:
                try:
                    y = pd.factorize(df[cc].fillna("_null_").astype(str))[0]
                    X = df[num[:5]].fillna(0).values
                    scores = mutual_info_classif(X, y, random_state=42)
                    for j, sc in enumerate(scores):
                        if sc >= MI_MIN:
                            lv = "strongly" if sc>0.3 else "moderately" if sc>0.1 else "weakly"
                            rels.append(CausalRel(tbl,num[j],tbl,cc,
                                "mutual_info_classif",float(sc),max(0.001,1-float(sc)),"→",
                                f"'{num[j]}' {lv} influences category '{cc}'."))
                except Exception:
                    pass

    rels.sort(key=lambda r: r.strength, reverse=True)
    return rels

# ─────────────────────────────────────────────────────────────
# SECTION 7 — HEALTH SCORES
# ─────────────────────────────────────────────────────────────
class HealthScore:
    LEVELS = [(75,"Healthy","✅","#3dd68c"),(50,"At Risk","⚠️","#f5a623"),(0,"Critical","🚨","#f47067")]
    def __init__(self,name,score,bd,rec):
        self.
\<Streaming stoppped because the conversation grew too long for this model\>
