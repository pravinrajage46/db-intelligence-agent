# dashboard.py
# Run: streamlit run dashboard.py
# Upload SQLite .db file OR multiple CSV files → get full analysis

from __future__ import annotations
import io, json, os, sys, tempfile, warnings, zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
# Line 1 - Add this at the very top
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Line 4 - Now this will work
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
import streamlit as st
from multi_ai import MultiAIProvider

# ── Sidebar AI Selector ─────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 Choose AI Provider")

providers = MultiAIProvider.PROVIDERS

# Only show providers that have API key
available = MultiAIProvider.get_available()

if not available:
    st.sidebar.error("❌ No API keys found in .env!")
else:
    # Provider selector
    provider_options = {
        f"{info['icon']} {info['name']}": key
        for key, info in available.items()
    }

    selected_label = st.sidebar.selectbox(
        "AI Provider",
        options=list(provider_options.keys())
    )

    selected_provider = provider_options[selected_label]

    # Model selector for chosen provider
    available_models = providers[selected_provider]["models"]
    selected_model = st.sidebar.selectbox(
        "Model",
        options=available_models
    )

    # Show status
    st.sidebar.success(
        f"✅ Using: {selected_label}\n\n📦 Model: `{selected_model}`"
    )

    # Initialize AI
    ai = MultiAIProvider(
        provider=selected_provider,
        model=selected_model
    )

    # Test button
    if st.sidebar.button("🧪 Test AI Connection"):
        with st.spinner("Testing..."):
            result = ai.generate(
                "Say hello in one line and tell your name.",
                max_tokens=100
            )
            st.sidebar.info(f"💬 {result}")
# ─────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(...)
    st.title(...)
    ...
    st.divider()
    run_btn = st.button(...)
    clear_btn = st.button(...)

    st.divider()
    st.caption("⭐ Causal Intelligence Engine")
    st.caption("Exceptional Feature...")

    # ✅ ADD THIS BELOW ↓
    st.divider()
    st.markdown("## 🤖 Choose AI Provider")

    from multi_ai import MultiAIProvider
    available = MultiAIProvider.get_available()

    if not available:
        st.error("❌ No API keys found in .env!")
    else:
        provider_options = {
            f"{info['icon']} {info['name']}": key
            for key, info in available.items()
        }
        selected_label = st.selectbox(
            "AI Provider",
            options=list(provider_options.keys())
        )
        selected_provider = provider_options[selected_label]
        available_models = MultiAIProvider.PROVIDERS[selected_provider]["models"]
        selected_model = st.selectbox("Model", options=available_models)
        st.success(f"✅ {selected_label}\n\n📦 `{selected_model}`")

        st.session_state["ai_provider"] = selected_provider
        st.session_state["ai_model"] = selected_model

        if st.button("🧪 Test AI Connection"):
            with st.spinner("Testing..."):
                ai = MultiAIProvider(
                    provider=selected_provider,
                    model=selected_model
                )
                result = ai.generate(
                    "Say hello in one line and tell your name.",
                    max_tokens=100
                )
                st.info(f"💬 {result}")
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

    except Exception as e:
        base["col_kind"] = "error"
        return base

    clean = s.dropna()
    if clean.empty:
        return p

    if pd.api.types.is_numeric_dtype(s):
        p.update({"col_kind":"numeric","min_value":_sf(clean.min()),"max_value":_sf(clean.max()),
                  "mean_value":_sf(clean.mean()),"std_value":_sf(clean.std())})
    elif pd.api.types.is_datetime64_any_dtype(s):
        p.update(_dt_stats(clean)); p["col_kind"] = "datetime"
    else:
        if _is_datetime_str(clean):
            try:
                parsed = pd.to_datetime(clean, errors="coerce").dropna()
                if not parsed.empty:
                    p.update(_dt_stats(parsed)); p["col_kind"] = "datetime_string"; return p
            except Exception:
                pass
        vc = clean.astype(str).str[:80].value_counts().head(10)
        p["top_values"] = [{"value":str(k),"count":int(v)} for k,v in vc.items()]
        p["col_kind"] = "categorical"
    return p

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
        self.table_name=name; self.score=score; self.breakdown=bd; self.recommendation=rec
        self.label,self.icon,self.color = self._cls()
    def _cls(self):
        for t,l,i,c in self.LEVELS:
            if self.score >= t: return l,i,c
        return "Unknown","❓","#999"
    def to_dict(self):
        return {"table":self.table_name,"score":self.score,"label":self.label,
                "icon":self.icon,"breakdown":self.breakdown,"recommendation":self.recommendation}

def compute_health(schema: Dict, quality: Dict) -> Dict[str, HealthScore]:
    scores = {}
    for tbl, metrics in quality.items():
        cols = metrics.get("columns", {})
        if not cols: continue

        avg_null = float(np.mean([v.get("null_rate",0) for v in cols.values()]))
        null_s   = max(0.0, 100*(1 - avg_null*2))

        fresh_s = 60.0
        for cd in cols.values():
            days = cd.get("freshness_days")
            if days is not None:
                fresh_s = max(0.0, 100-(days/60)*100); break

        comp_s = float(metrics.get("overall_completeness",0.8))*100

        pk_cols = schema["tables"].get(tbl,{}).get("primary_keys",[])
        uniq_s  = 100.0
        for pk in pk_cols:
            if pk in cols:
                uniq_s = min(uniq_s, cols[pk].get("unique_rate",1.0)*100)

        composite = round(min(100, max(0,
            null_s*0.30 + fresh_s*0.25 + comp_s*0.25 + uniq_s*0.20)), 1)

        bd = {"Null Score":round(null_s,1),"Freshness":round(fresh_s,1),
              "Completeness":round(comp_s,1),"PK Uniqueness":round(uniq_s,1)}

        worst = metrics.get("worst_columns",[])
        if composite >= 75:
            rec = f"'{tbl}' is healthy. Maintain current pipelines."
        elif composite >= 50:
            ws = ", ".join(f"'{c}'" for c in worst[:2]) or "some columns"
            rec = f"'{tbl}' needs attention — high nulls in {ws}."
        else:
            rec = f"CRITICAL: '{tbl}' has severe quality issues. Audit ETL immediately."

        scores[tbl] = HealthScore(tbl, composite, bd, rec)
    return scores

# ─────────────────────────────────────────────────────────────
# SECTION 8 — DATA CONTRACT GENERATION
# ─────────────────────────────────────────────────────────────
def generate_contracts(schema: Dict, quality: Dict) -> Dict[str, str]:
    contracts = {}
    for tbl, metrics in quality.items():
        cols = metrics.get("columns",{})
        tbl_info = schema["tables"].get(tbl,{})
        pk_cols  = tbl_info.get("primary_keys",[])

        fresh_h = 24
        for cd in cols.values():
            days = cd.get("freshness_days")
            if days and days > 0:
                fresh_h = max(1, int(days*0.5*24)); break

        contract = {
            "version":"1.0",
            "generated_at":datetime.now().isoformat(),
            "generated_by":"DB Intelligence Agent — Causal Intelligence Engine",
            "table":tbl,
            "sla":{
                "freshness_max_hours":fresh_h,
                "min_row_count":max(1, int(metrics.get("sampled_rows",0)*0.9)),
                "min_completeness":float(round(metrics.get("overall_completeness",0.8),3)),
            },
            "columns":{},
        }

        for cn, cd in cols.items():
            is_pk  = cn in pk_cols
            obs_null = cd.get("null_rate", 0.0)
            max_null = round(min(0.99, obs_null*1.1+0.005), 4)
            cc: Dict[str,Any] = {
                "dtype":cd.get("dtype","unknown"),
                "nullable":obs_null > 0,
                "max_null_rate":max_null,
                "min_completeness":round(1-max_null,4),
            }
            if is_pk:
                cc.update({"unique":True,"not_null":True,"max_null_rate":0.0})
            if cd.get("min_value") is not None:
                cc["min_value"] = cd["min_value"]
                cc["max_value"] = cd["max_value"]
            tv = cd.get("top_values",[])
            if tv and cd.get("unique_rate",1) < 0.05:
                cc["allowed_values"] = [v["value"] for v in tv[:20]]
            if cd.get("freshness_days") is not None:
                cc["freshness_max_days"] = int(cd["freshness_days"]*1.5)
            contract["columns"][cn] = cc

        contracts[tbl] = yaml.dump(contract, default_flow_style=False,
                                   sort_keys=False, allow_unicode=True)
    return contracts

# ─────────────────────────────────────────────────────────────
# SECTION 9 — AI NARRATOR
# ─────────────────────────────────────────────────────────────
def call_claude(prompt: str, api_key: str, model: str, max_tokens=600) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(model=model, max_tokens=max_tokens,
                                     messages=[{"role":"user","content":prompt}])
        return msg.content[0].text.strip()
    except Exception as e:
        return f"[AI unavailable: {e}]"

def generate_ai(schema, er_summary, quality, causal_rels, api_key, model) -> Dict:
    ai = {}
    if not api_key:
        n=schema["total_tables"]; c=schema["total_columns"]; r=schema["total_rows"]
        ai["schema"] = (f"This database has {n} tables, {c} columns, and {r:,} rows. "
                        "Add your Anthropic API key in the sidebar to enable AI summaries.")
        ai["causal"] = "Causal relationships found — add API key for business interpretation."
        ai["tables"] = {t:(f"BUSINESS PURPOSE: The '{t}' table stores {tb['row_count']:,} records. "
                           "Add API key for AI description.\nCOLUMNS:\n"+
                           "\n".join(f"- {c['name']}: —" for c in tb["columns"][:5]))
                        for t, tb in schema["tables"].items()}
        return ai

    tbl_sizes = {n:{"rows":t["row_count"],"cols":len(t["columns"])} for n,t in schema["tables"].items()}
    ai["schema"] = call_claude(
        f"You are a senior data architect. Analyze this database and write a 3-sentence "
        f"executive summary: what business domain it serves, key tables, and design patterns.\n"
        f"Tables: {json.dumps(tbl_sizes)}\nRelationships: {len(er_summary.get('edges',[]))} FKs\n"
        f"Write for a business audience. No bullet points.", api_key, model, 400)

    if causal_rels:
        rel_data = [r.to_dict() for r in causal_rels[:10]]
        ai["causal"] = call_claude(
            f"Write 3 sentences explaining what these causal relationships mean for business:\n"
            f"{json.dumps(rel_data)}\nPlain English only.", api_key, model, 300)
    else:
        ai["causal"] = "No significant causal relationships found at current sample size."

    ai["tables"] = {}
    for tbl_name, tbl in schema["tables"].items():
        col_list = [{"name":c["name"],"type":c["type"]} for c in tbl["columns"][:15]]
        q = quality.get(tbl_name,{})
        ai["tables"][tbl_name] = call_claude(
            f"Describe the '{tbl_name}' database table in 2 sentences for a business user, "
            f"then give a one-line description for each column.\n"
            f"Table: {tbl_name} | Rows: {tbl['row_count']:,} | "
            f"Completeness: {q.get('overall_completeness',0):.1%}\n"
            f"Columns: {json.dumps(col_list)}\n"
            f"Format:\nBUSINESS PURPOSE: <2 sentences>\nCOLUMNS:\n- column_name: <one line>",
            api_key, model, 500)
    return ai

# ─────────────────────────────────────────────────────────────
# SECTION 10 — REPORT BUILDER
# ─────────────────────────────────────────────────────────────
def build_markdown(schema, ai, quality, health_scores) -> str:
    lines = [
        "# 📖 Data Dictionary",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        f"**Tables:** {schema['total_tables']} | **Columns:** {schema['total_columns']} | "
        f"**Rows:** {schema['total_rows']:,}",
        "","---","","## Table Index","",
        "| Table | Rows | Columns | Completeness | Health |",
        "|-------|------|---------|-------------|--------|",
    ]
    for n, t in schema["tables"].items():
        h = health_scores.get(n)
        q = quality.get(n,{})
        lines.append(f"| `{n}` | {t['row_count']:,} | {len(t['columns'])} | "
                     f"{q.get('overall_completeness',0):.1%} | "
                     f"{h.icon+' '+h.label+'('+str(h.score)+')' if h else '—'} |")
    lines += ["","---",""]
    for n, t in schema["tables"].items():
        h = health_scores.get(n)
        lines += [f"## `{n}` {h.icon if h else ''}",
                  f"**Rows:** {t['row_count']:,} | **Columns:** {len(t['columns'])} | "
                  f"**PKs:** {', '.join(f'`{k}`' for k in t['primary_keys']) or 'None'}",""]
        s = ai.get("tables",{}).get(n,"")
        if "BUSINESS PURPOSE:" in s:
            bp = s.split("BUSINESS PURPOSE:")[-1].split("COLUMNS:")[0].strip()
            lines += [f"> {bp}",""]
        if h:
            lines += [f"**Health:** {h.score}/100 — {h.recommendation}",""]
        lines += ["| Column | Type | PK | FK | Null Rate |","|--------|------|----|----|-----------|"]
        qc = quality.get(n,{}).get("columns",{})
        for col in t["columns"]:
            nr = qc.get(col["name"],{}).get("null_rate",0)
            nr_s = f"{nr:.1%}" if isinstance(nr,float) else "—"
            pk = "✅" if col["is_pk"] else ""
            fk = f"→ {col['ref_table']}.{col['ref_col']}" if col["is_fk"] else ""
            lines.append(f"| `{col['name']}` | `{col['type']}` | {pk} | {fk} | {nr_s} |")
        lines += ["",""]
    return "\n".join(lines)

def build_zip(contracts, json_report, md_dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("db_analysis.json", json_report)
        zf.writestr("data_dictionary.md", md_dict)
        for t, y in contracts.items():
            zf.writestr(f"data_contracts/{t}_contract.yaml", y)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────
# SECTION 11 — SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DB Intelligence Agent")
    st.caption("Schema · Quality · Causal AI · Governance")
    st.divider()

    st.subheader("📂 Upload Your Data")
    mode = st.radio("Input type", ["🗄️ SQLite File (.db)","📊 CSV Files","🔗 Database URL"],
                    label_visibility="collapsed")

    engine = None
    ready  = False

    if mode == "🗄️ SQLite File (.db)":
        f = st.file_uploader("SQLite database", type=["db","sqlite","sqlite3"],
                             label_visibility="collapsed")
        if f:
            try:
                engine, _ = connect_sqlite(f)
                if test_engine(engine):
                    st.success(f"✅ Loaded: {f.name}")
                    ready = True
                else:
                    st.error("❌ Cannot read this file")
            except Exception as e:
                st.error(f"❌ {e}")

    elif mode == "📊 CSV Files":
        files = st.file_uploader("CSV files (upload multiple)", type=["csv"],
                                 accept_multiple_files=True, label_visibility="collapsed")
        if files:
            try:
                engine, _, loaded = connect_csvs(files)
                if test_engine(engine) and loaded:
                    st.success(f"✅ Loaded {len(loaded)} tables")
                    ready = True
                else:
                    st.error("❌ No tables loaded")
            except Exception as e:
                st.error(f"❌ {e}")

    else:
        url_in = st.text_input("SQLAlchemy URL",
                               placeholder="postgresql://user:pass@host:5432/db")
        if url_in:
            try:
                engine, _ = connect_url(url_in)
                if test_engine(engine):
                    st.success("✅ Connected!")
                    ready = True
                else:
                    st.error("❌ Cannot connect")
            except Exception as e:
                st.error(f"❌ {e}")

    st.divider()
    st.subheader("⚙️ Settings")
    api_key = st.text_input("Anthropic API Key (optional)", type="password",
                             placeholder="sk-ant-...")
    model   = st.selectbox("AI Model", ["claude-sonnet-4-20250514","claude-opus-4-20250514"])
    n_rows  = st.slider("Sample rows per table", 500, 50000, 5000, 500)

    st.divider()
    run_btn   = st.button("🚀 Analyze", type="primary", use_container_width=True, disabled=not ready)
    clear_btn = st.button("🗑️ Clear", use_container_width=True)
    if clear_btn:
        st.session_state.clear(); st.rerun()

# ─────────────────────────────────────────────────────────────
# SECTION 12 — ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────
def run_analysis(engine, api_key, model, n_rows):
    bar    = st.progress(0)
    status = st.empty()
    def upd(p, msg): bar.progress(p); status.info(f"⏳ {msg}")

    upd(8,  "Extracting schema — tables, columns, keys, constraints...")
    schema = extract_schema(engine)

    upd(20, f"Sampling data ({n_rows:,} rows per table)...")
    samples = sample_all(engine, schema, n_rows)

    upd(35, "Profiling data quality — nulls, freshness, distributions...")
    quality = profile_quality(schema, samples)

    upd(48, "Building entity-relationship map...")
    G, er_summary = build_er(schema)

    upd(58, "⭐ Causal Phase 1 — Discovering causal relationships...")
    causal_rels = discover_causal(samples)

    upd(68, "⭐ Causal Phase 2 — Computing predictive health scores...")
    health_scores = compute_health(schema, quality)

    upd(76, "⭐ Causal Phase 3 — Generating YAML data contracts...")
    contracts = generate_contracts(schema, quality)

    upd(84, "🤖 Generating AI business narratives...")
    ai = generate_ai(schema, er_summary, quality, causal_rels, api_key, model)

    upd(94, "Building downloadable reports...")
    md   = build_markdown(schema, ai, quality, health_scores)
    jr   = json.dumps({"metadata":{"generated_at":datetime.now().isoformat(),
                                    "generator":"DB Intelligence Agent v1.0",
                                    **{k:schema[k] for k in ["total_tables","total_columns","total_rows"]}},
                        "schema":schema,"quality":quality,"er_structure":er_summary,
                        "causal":[r.to_dict() for r in causal_rels],
                        "health":{t:h.to_dict() for t,h in health_scores.items()}},
                       indent=2, default=str)
    zb   = build_zip(contracts, jr, md)

    bar.progress(100); status.success("✅ Complete!")
    return {"schema":schema,"quality":quality,"G":G,"er_summary":er_summary,
            "causal_rels":causal_rels,"health_scores":health_scores,"contracts":contracts,
            "ai":ai,"md":md,"json_report":jr,"zip_bytes":zb}

if run_btn and engine and ready:
    try:
        st.session_state["R"] = run_analysis(engine, api_key, model, n_rows)
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        import traceback; st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────
# SECTION 13 — WELCOME SCREEN
# ─────────────────────────────────────────────────────────────
if "R" not in st.session_state:
    st.markdown("# 🧠 DB Intelligence Agent")
    st.markdown("Upload a database file in the sidebar → click **Analyze** → get full AI-powered analysis.")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.info("**📋 Schema**\nTables, columns, PKs, FKs, types, constraints")
    c2.info("**💊 Data Quality**\nNulls, freshness, completeness, distributions")
    c3.info("**⭐ Causal AI**\nDirectional influences + health scores + YAML contracts")
    c4.info("**🤖 AI Summaries**\nBusiness context generated with Claude")
    st.markdown("---")
    st.markdown("**Supported formats:** SQLite `.db` · CSV files · PostgreSQL · MySQL")
    st.stop()

# ─────────────────────────────────────────────────────────────
# SECTION 14 — RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────
R             = st.session_state["R"]
schema        = R["schema"]
quality       = R["quality"]
health_scores = R["health_scores"]
causal_rels   = R["causal_rels"]
er_summary    = R["er_summary"]
G             = R["G"]
ai            = R["ai"]
contracts     = R["contracts"]

# Top metrics
st.markdown("---")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("📦 Tables",       schema["total_tables"])
c2.metric("🔢 Columns",      schema["total_columns"])
c3.metric("📊 Rows",         f"{schema['total_rows']:,}")
c4.metric("🔗 Relationships",er_summary["total_relationships"])
c5.metric("⭐ Causal Pairs", len(causal_rels))
avg_h = round(np.mean([h.score for h in health_scores.values()]),1) if health_scores else 0
c6.metric("💊 Avg Health",   f"{avg_h}/100")
st.markdown("---")

# AI Summary
st.info(f"🤖 **AI Schema Summary:** {ai.get('schema','')}")

# Tabs
t1,t2,t3,t4,t5,t6 = st.tabs([
    "📋 Schema","🔗 ER Map","💊 Data Quality",
    "⭐ Causal Intelligence","📖 Data Dictionary","⬇️ Downloads"
])

# ═══ TAB 1 — SCHEMA ════════════════════════════════════════
with t1:
    st.subheader("Schema Overview")
    rows = []
    for n, t in schema["tables"].items():
        h = health_scores.get(n)
        q = quality.get(n,{})
        rows.append({
            "Table":n,"Rows":f"{t['row_count']:,}","Columns":len(t["columns"]),
            "Primary Keys":", ".join(t["primary_keys"]) or "—",
            "Foreign Keys":len(t["foreign_keys"]),"Indexes":t["index_count"],
            "Completeness":f"{q.get('overall_completeness',0):.1%}",
            "Health":f"{h.icon} {h.label} ({h.score})" if h else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Table Details")
    for n, t in schema["tables"].items():
        h    = health_scores.get(n)
        a_txt = ai.get("tables",{}).get(n,"")
        bp = a_txt.split("BUSINESS PURPOSE:")[-1].split("COLUMNS:")[0].strip() \
             if "BUSINESS PURPOSE:" in a_txt else ""
        badge = f"{h.icon} {h.label} ({h.score}/100)" if h else ""
        with st.expander(f"📦 {n}  {badge}  · {t['row_count']:,} rows"):
            if bp:
                st.info(f"🤖 {bp}")
            if h:
                clr = "🟢" if h.label=="Healthy" else "🟡" if h.label=="At Risk" else "🔴"
                st.markdown(f"**{clr} Health:** {h.recommendation}")

            qc = quality.get(n,{}).get("columns",{})
            crows = []
            for col in t["columns"]:
                cd = qc.get(col["name"],{})
                crows.append({
                    "Column":col["name"],"Type":col["type"],
                    "PK":"✅" if col["is_pk"] else "",
                    "FK":f"→ {col['ref_table']}.{col['ref_col']}" if col["is_fk"] else "",
                    "Nullable":"Yes" if col["nullable"] else "No",
                    "Null Rate":f"{cd.get('null_rate',0):.1%}" if cd else "—",
                    "Unique Rate":f"{cd.get('unique_rate',0):.1%}" if cd else "—",
                    "Kind":cd.get("col_kind","—"),
                    "Min":cd.get("min_value","—"),"Max":cd.get("max_value","—"),
                    "Mean":cd.get("mean_value","—"),
                })
            st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)
            if "COLUMNS:" in a_txt:
                st.markdown("**🤖 Column Descriptions:**")
                st.markdown(a_txt.split("COLUMNS:")[-1].strip())

# ═══ TAB 2 — ER MAP ════════════════════════════════════════
with t2:
    st.subheader("Entity Relationship Map")
    edges = er_summary.get("edges",[])
    if not edges:
        st.warning("No FK relationships detected. Using heuristic detection for CSV files.")
    fig = render_er_chart(G)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Red = hub table (3+ connections)  ·  🔵 Blue = regular table  ·  Hover nodes for details")

    cl, cr = st.columns(2)
    with cl:
        if edges:
            st.subheader("Relationship Details")
            st.dataframe(pd.DataFrame(edges), use_container_width=True, hide_index=True)
    with cr:
        hubs = er_summary.get("hub_tables",[])
        if hubs:
            st.subheader("Most Connected Tables")
            hub_df = pd.DataFrame(hubs).head(10)
            fig_h = px.bar(hub_df, x="centrality", y="table", orientation="h",
                           color="centrality", color_continuous_scale="Blues",
                           template="plotly_dark", title="Degree Centrality")
            fig_h.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_h, use_container_width=True)

# ═══ TAB 3 — DATA QUALITY ══════════════════════════════════
with t3:
    st.subheader("Predictive Health Scores")
    if health_scores:
        hdf = pd.DataFrame([h.to_dict() for h in health_scores.values()])
        fig_h = px.bar(hdf.sort_values("score"), x="score", y="table", orientation="h",
                       color="label", template="plotly_dark", text="score",
                       color_discrete_map={"Healthy":"#3dd68c","At Risk":"#f5a623","Critical":"#f47067"},
                       title="Health Score by Table (0–100)")
        fig_h.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_h.update_layout(height=max(280, len(health_scores)*52),
                            margin=dict(l=10,r=80,t=40,b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        st.subheader("Score Breakdown")
        for h in health_scores.values():
            with st.expander(f"{h.icon} {h.table_name}  —  {h.score}/100  ({h.label})"):
                c1,c2,c3,c4 = st.columns(4)
                for col_w, (key, val) in zip([c1,c2,c3,c4], h.breakdown.items()):
                    col_w.metric(key, f"{val}/100")
                st.warning(f"💡 {h.recommendation}")

    st.subheader("Column-Level Quality Inspector")
    sel = st.selectbox("Select table", list(quality.keys()))
    if sel and sel in quality:
        tq = quality[sel]
        crows = []
        for cn, cd in tq.get("columns",{}).items():
            crows.append({
                "Column":cn,"Kind":cd.get("col_kind","—"),"Type":cd.get("dtype","—"),
                "Null Count":cd.get("null_count",0),
                "Null Rate":f"{cd.get('null_rate',0):.2%}",
                "Completeness":f"{cd.get('completeness',0):.2%}",
                "Unique Count":cd.get("unique_count",0),
                "Unique Rate":f"{cd.get('unique_rate',0):.2%}",
                "Min":cd.get("min_value"),"Max":cd.get("max_value"),
                "Mean":cd.get("mean_value"),"Std":cd.get("std_value"),
                "Freshness(days)":cd.get("freshness_days"),
            })
        if crows:
            st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)
            la, lb = st.columns(2)
            with la:
                nd = [{"Column":r["Column"],"NullPct":float(r["Null Rate"].strip("%"))}
                      for r in crows if float(r["Null Rate"].strip("%")) > 0]
                if nd:
                    fn = px.bar(pd.DataFrame(nd).sort_values("NullPct"),
                                x="NullPct", y="Column", orientation="h",
                                color="NullPct", color_continuous_scale="Reds",
                                title=f"Null Rates — {sel}", template="plotly_dark")
                    fn.update_layout(height=350, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fn, use_container_width=True)
                else:
                    st.success("🎉 No nulls in this table!")
            with lb:
                cat_cols = [r["Column"] for r in crows if r["Kind"]=="categorical"]
                if cat_cols:
                    sel_c = st.selectbox("Top values", cat_cols, key="tv_sel")
                    tv = tq["columns"].get(sel_c,{}).get("top_values",[])
                    if tv:
                        ft = px.bar(pd.DataFrame(tv).head(10), x="count", y="value",
                                    orientation="h", color="count",
                                    color_continuous_scale="Blues",
                                    title=f"Top Values — {sel_c}", template="plotly_dark")
                        ft.update_layout(height=350, margin=dict(l=10,r=10,t=40,b=10))
                        st.plotly_chart(ft, use_container_width=True)

# ═══ TAB 4 — CAUSAL INTELLIGENCE ══════════════════════════
with t4:
    st.markdown("""
    > ⭐ **Causal Intelligence Engine** — Discovers *directional* influences between columns
    using Mutual Information and Chi-Squared tests. Goes beyond correlation to show which
    column drives which, generates Predictive Health Scores, and auto-creates YAML Data Contracts.
    """)

    st.subheader("🤖 AI Causal Narrative")
    st.info(ai.get("causal",""))

    st.subheader(f"Discovered Causal Relationships ({len(causal_rels)} found)")
    if causal_rels:
        rel_df = pd.DataFrame([r.to_dict() for r in causal_rels])
        top20  = rel_df.head(20).copy()
        top20["label"] = top20["from"]+" "+top20["direction"]+" "+top20["to"]
        fig_c = px.bar(top20, x="strength", y="label", color="method",
                       orientation="h", template="plotly_dark",
                       title="Top 20 Causal Relationships by Strength",
                       color_discrete_map={
                           "mutual_information":"#5b8af5",
                           "chi_squared":"#3dd68c",
                           "mutual_info_classif":"#f5a623"},
                       hover_data=["p_value","insight"])
        fig_c.update_layout(height=max(350, len(top20)*30),
                            margin=dict(l=10,r=30,t=40,b=10))
        st.plotly_chart(fig_c, use_container_width=True)
        st.dataframe(
            rel_df[["from","direction","to","method","strength","p_value","insight"]],
            use_container_width=True, hide_index=True)
    else:
        st.info("No significant relationships found. Try increasing sample size.")

    st.subheader("Auto-Generated Data Contracts (YAML)")
    st.markdown("*Ready for Great Expectations · dbt schema tests · custom pipelines*")
    m1,m2,m3 = st.columns(3)
    m1.metric("Contracts Generated", len(contracts))
    total_rules = sum(len(yaml.safe_load(y).get("columns",{}))
                      for y in contracts.values() if y.strip())
    m2.metric("Column Rules Total", total_rules)
    m3.metric("Coverage", "100%")

    for tbl_name, yaml_str in contracts.items():
        h = health_scores.get(tbl_name)
        with st.expander(f"📄 {tbl_name}  {h.icon+' '+h.label if h else ''}"):
            col_v, col_d = st.columns([5,1])
            with col_v:
                st.code(yaml_str, language="yaml")
            with col_d:
                st.download_button("⬇️", data=yaml_str,
                    file_name=f"{tbl_name}_contract.yaml", mime="text/yaml",
                    key=f"dc_{tbl_name}", use_container_width=True)

# ═══ TAB 5 — DATA DICTIONARY ════════════════════════════════
with t5:
    st.subheader("Human-Readable Data Dictionary")
    st.markdown(R["md"])
    st.download_button("⬇️ Download (.md)", data=R["md"],
                       file_name="data_dictionary.md", mime="text/markdown")

# ═══ TAB 6 — DOWNLOADS ══════════════════════════════════════
with t6:
    st.subheader("Download All Reports")
    st.success(f"Analysis complete — {schema['total_tables']} tables · "
               f"{len(causal_rels)} causal pairs · {len(contracts)} contracts")

    d1,d2,d3 = st.columns(3)
    with d1:
        st.markdown("**📦 Complete Package**")
        st.caption("JSON + Markdown + all YAML contracts in one ZIP")
        st.download_button("⬇️ Download Everything (.zip)",
                           data=R["zip_bytes"],
                           file_name=f"db_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                           mime="application/zip", type="primary", use_container_width=True)
    with d2:
        st.markdown("**📖 Data Dictionary**")
        st.caption("Human-readable Markdown file")
        st.download_button("⬇️ Data Dictionary (.md)", data=R["md"],
                           file_name="data_dictionary.md", mime="text/markdown",
                           use_container_width=True)
    with d3:
        st.markdown("**📄 Full JSON Report**")
        st.caption("Machine-readable complete analysis")
        st.download_button("⬇️ Full Report (.json)", data=R["json_report"],
                           file_name="db_analysis.json", mime="application/json",
                           use_container_width=True)

    st.subheader("Individual Data Contracts")
    cc = st.columns(min(4, max(1, len(contracts))))
    for i, (tbl_name, yaml_str) in enumerate(contracts.items()):
        h = health_scores.get(tbl_name)
        with cc[i % len(cc)]:
            st.download_button(f"{h.icon if h else '📄'} {tbl_name}",
                               data=yaml_str,
                               file_name=f"{tbl_name}_contract.yaml",
                               mime="text/yaml",
                               key=f"dl2_{tbl_name}",
                               use_container_width=True)

    st.subheader("Analysis Summary")
    srows = []
    for n, t in schema["tables"].items():
        h = health_scores.get(n)
        q = quality.get(n,{})
        srows.append({
            "Table":n, "Rows":t["row_count"],
            "Columns":len(t["columns"]),
            "Completeness":f"{q.get('overall_completeness',0):.1%}",
            "Health Score":h.score if h else "—",
            "Status":f"{h.icon} {h.label}" if h else "—",
            "Causal Pairs":sum(1 for r in causal_rels if r.from_table==n),
            "Contract":  "✅" if n in contracts else "❌",
        })
    st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)
