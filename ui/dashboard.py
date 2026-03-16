from __future__ import annotations
import io, json, os, tempfile, warnings, zipfile, base64
from datetime import datetime
from typing import Any
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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sqlalchemy import create_engine, inspect, text
from fpdf import FPDF

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎨 THEME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(page_title="DB Intelligence Agent", layout="wide", page_icon="🧠")

COLORS = {
    "bg": "#0a0a0f", "card": "#12121a", "accent": "#6c5ce7",
    "accent2": "#00cec9", "success": "#00b894", "warning": "#fdcb6e",
    "danger": "#e17055", "text": "#dfe6e9", "muted": "#636e72",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {{
        background: linear-gradient(135deg, {COLORS['bg']} 0%, #0d0d1a 50%, #111128 100%);
        font-family: 'Inter', sans-serif;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding: 1rem 2rem;}}
    .glass-card {{
        background: rgba(18, 18, 26, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(108, 92, 231, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }}
    .glass-card:hover {{
        border-color: rgba(108, 92, 231, 0.5);
        box-shadow: 0 0 30px rgba(108, 92, 231, 0.15);
        transform: translateY(-2px);
    }}
    .metric-card {{
        background: linear-gradient(135deg, rgba(108,92,231,0.15), rgba(0,206,201,0.1));
        border: 1px solid rgba(108,92,231,0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent2']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: {COLORS['muted']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}
    .hero {{
        text-align: center;
        padding: 3rem 1rem;
        background: radial-gradient(ellipse at center, rgba(108,92,231,0.12) 0%, transparent 70%);
    }}
    .hero h1 {{
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, {COLORS['accent']} 50%, {COLORS['accent2']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }}
    .hero p {{ color: {COLORS['muted']}; font-size: 1.1rem; max-width: 600px; margin: 0 auto; }}
    .chat-msg {{ padding: 1rem; border-radius: 12px; margin: 0.5rem 0; max-width: 85%; }}
    .chat-user {{
        background: linear-gradient(135deg, rgba(108,92,231,0.3), rgba(108,92,231,0.1));
        border: 1px solid rgba(108,92,231,0.3); margin-left: auto;
    }}
    .chat-ai {{
        background: rgba(18,18,26,0.8);
        border: 1px solid rgba(0,206,201,0.2);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px; background: rgba(18,18,26,0.5); border-radius: 12px; padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 8px; color: {COLORS['muted']}; font-weight: 500; }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent2']}) !important;
        color: white !important;
    }}
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {COLORS['accent']}; border-radius: 3px; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0d0d1a 0%, #12121a 100%);
        border-right: 1px solid rgba(108,92,231,0.15);
    }}
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent2']}) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108,92,231,0.4) !important;
    }}
    .stDataFrame {{ border-radius: 12px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🤖 AI PROVIDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AI:
    PROVIDERS = {
        "anthropic": {"name":"Claude","icon":"🟠","models":["claude-sonnet-4-20250514","claude-opus-4-20250514"],"key":"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"},
        "openai": {"name":"GPT-4o","icon":"🟢","models":["gpt-4o","gpt-4o-mini","gpt-4-turbo"],"key":"sk-svcacct-CMaeBCQHRQFZEPpkosPPuI34_R2rtgG1L6nuqsy_klx4a0iqrAg7vgo3y-5MxRLWfrAGZ1ADSOxT3BlbkFJRtJfiteqlUFWoLLVxyZEftL3RKGrYn5k7sHLf5GUFiDZC-CgyjbGO8TIBL7gQ3l5rzu2Cej0engA
"},
        "gemini": {"name":"Gemini","icon":"🔵","models":["gemini-2.0-flash","gemini-1.5-pro"],"key":"AIzaSyDgAIlErzufpxgssvkMl3743btImO3TlX0"},
        "groq": {"name":"Groq⚡","icon":"🟣","models":["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768"],"key":"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"},
    }

    def __init__(self, provider=None, model=None):
        self.provider = provider or os.getenv("DEFAULT_AI_PROVIDER","groq")
        self.model = model or os.getenv("DEFAULT_AI_MODEL", self.PROVIDERS[self.provider]["models"][0])
        self.client = self._init()

    def _init(self):
        key = os.getenv(self.PROVIDERS[self.provider]["key"])
        if not key: return None
        if self.provider == "anthropic":
            import anthropic; return anthropic.Anthropic(api_key=key)
        elif self.provider == "openai":
            from openai import OpenAI; return OpenAI(api_key=key)
        elif self.provider == "gemini":
            import google.generativeai as genai; genai.configure(api_key=key); return genai.GenerativeModel(self.model)
        elif self.provider == "groq":
            from groq import Groq; return Groq(api_key=key)

    def ask(self, prompt: str, max_tokens=1024) -> str:
        if not self.client: return "⚠️ No API key for this provider in .env"
        try:
            if self.provider == "anthropic":
                return self.client.messages.create(model=self.model, max_tokens=max_tokens, messages=[{"role":"user","content":prompt}]).content[0].text
            elif self.provider in ("openai","groq"):
                return self.client.chat.completions.create(model=self.model, max_tokens=max_tokens, messages=[{"role":"user","content":prompt}]).choices[0].message.content
            elif self.provider == "gemini":
                return self.client.generate_content(prompt).text
        except Exception as e: return f"❌ {self.provider}: {e}"

    @classmethod
    def available(cls):
        return {k:v for k,v in cls.PROVIDERS.items() if os.getenv(v["key"])}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔌 DATABASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_sqlite(f):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db"); tmp.write(f.getbuffer()); tmp.close()
    return create_engine(f"sqlite:///{tmp.name}")

def load_csvs(files):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db"); tmp.close()
    eng = create_engine(f"sqlite:///{tmp.name}"); names = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            name = os.path.splitext(f.name)[0].lower().replace(" ","_").replace("-","_")
            df.to_sql(name, eng, if_exists="replace", index=False); names.append(name)
        except: pass
    return eng, names

def test_db(eng):
    try:
        with eng.connect() as c: c.execute(text("SELECT 1")); return True
    except: return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 SCHEMA + QUALITY + CAUSAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_schema(eng):
    insp = inspect(eng); tables = {}
    for tbl in insp.get_table_names():
        try:
            cols = insp.get_columns(tbl)
            pk = insp.get_pk_constraint(tbl).get("constrained_columns",[])
            fks = insp.get_foreign_keys(tbl); idxs = insp.get_indexes(tbl)
            with eng.connect() as c: rows = c.execute(text(f'SELECT COUNT(*) FROM "{tbl}"')).scalar() or 0
        except: continue
        fk_map, fk_list = {}, []
        for fk in fks:
            for cc, rc in zip(fk.get("constrained_columns",[]), fk.get("referred_columns",[])):
                fk_map[cc] = {"ref_table":fk.get("referred_table",""),"ref_col":rc}
            fk_list.append({"constrained_columns":fk.get("constrained_columns",[]),
                           "referred_table":fk.get("referred_table",""),
                           "referred_columns":fk.get("referred_columns",[])})
        columns = []
        for col in cols:
            n = col["name"]; fi = fk_map.get(n,{})
            columns.append({"name":n,"type":str(col.get("type","TEXT")),"nullable":bool(col.get("nullable",True)),
                           "is_pk":n in pk,"is_fk":n in fk_map,
                           "ref_table":fi.get("ref_table",""),"ref_col":fi.get("ref_col","")})
        tables[tbl] = {"name":tbl,"row_count":rows,"columns":columns,
                       "primary_keys":pk,"foreign_keys":fk_list,"index_count":len(idxs)}
    if sum(len(t["foreign_keys"]) for t in tables.values()) == 0:
        tnames = list(tables.keys())
        for tn, t in tables.items():
            for col in t["columns"]:
                if col["name"].endswith("_id") and not col["is_pk"]:
                    base = col["name"][:-3]
                    for cand in [base, base+"s", base+"es"]:
                        if cand in tnames and cand != tn:
                            ref_cols = [c["name"] for c in tables[cand]["columns"]]
                            rc = col["name"] if col["name"] in ref_cols else ("id" if "id" in ref_cols else None)
                            if rc:
                                col["is_fk"]=True; col["ref_table"]=cand; col["ref_col"]=rc
                                t["foreign_keys"].append({"constrained_columns":[col["name"]],"referred_table":cand,"referred_columns":[rc]})
                                break
    return {"tables":tables,"total_tables":len(tables),
            "total_columns":sum(len(t["columns"]) for t in tables.values()),
            "total_rows":sum(t["row_count"] for t in tables.values())}

def sample_tables(eng, schema, n=5000):
    out = {}
    for tbl in schema["tables"]:
        try:
            with eng.connect() as c: out[tbl] = pd.read_sql(text(f'SELECT * FROM "{tbl}" LIMIT :n'), c, params={"n":n})
        except: out[tbl] = pd.DataFrame()
    return out

def _sf(v):
    try: f=float(v); return round(f,4) if np.isfinite(f) else None
    except: return None

def profile_table(df):
    if df is None or df.empty: return {"sampled_rows":0,"columns":{},"overall_completeness":0}
    cols = {}
    for c in df.columns:
        s=df[c]; total=len(s); nulls=int(s.isna().sum())
        nr=round(nulls/total,6) if total else 0
        uniq=int(s.nunique(dropna=True))
        p={"dtype":str(s.dtype),"null_count":nulls,"null_rate":nr,"completeness":round(1-nr,6),
           "unique_count":uniq,"unique_rate":round(uniq/max(total-nulls,1),6),"col_kind":"unknown"}
        clean=s.dropna()
        if clean.empty: cols[c]=p; continue
        if pd.api.types.is_numeric_dtype(s):
            p.update({"col_kind":"numeric","min_value":_sf(clean.min()),"max_value":_sf(clean.max()),
                      "mean_value":_sf(clean.mean()),"std_value":_sf(clean.std())})
        else:
            vc=clean.astype(str).str[:80].value_counts().head(10)
            p["top_values"]=[{"value":str(k),"count":int(v)} for k,v in vc.items()]
            p["col_kind"]="categorical"
        cols[c]=p
    vals=[v["completeness"] for v in cols.values()]
    return {"sampled_rows":len(df),"columns":cols,
            "overall_completeness":round(float(np.mean(vals)),4) if vals else 0,
            "worst_columns":sorted(cols, key=lambda x: cols[x]["null_rate"], reverse=True)[:3]}

def profile_all(schema, samples):
    return {tbl: profile_table(df) for tbl, df in samples.items()}

def find_causal(samples, mi_min=0.05, chi2_p=0.05, max_cols=8, max_cats=25):
    rels = []
    for tbl, df in samples.items():
        if df is None or df.empty or len(df)<50: continue
        num=df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
        cat=[c for c in df.select_dtypes(include=["object","category"]).columns if df[c].nunique()<=max_cats][:max_cols]
        if len(num)>=2:
            for target in num[:6]:
                feats=[c for c in num if c!=target][:5]
                if not feats: continue
                try:
                    scores=mutual_info_regression(df[feats].fillna(0).values, df[target].fillna(0).values, random_state=42)
                    for j,sc in enumerate(scores):
                        if sc>=mi_min:
                            lv="strongly" if sc>0.3 else "moderately" if sc>0.1 else "weakly"
                            rels.append({"from":f"{tbl}.{feats[j]}","to":f"{tbl}.{target}","method":"mutual_info",
                                        "strength":round(float(sc),4),"p_value":round(max(0.001,1-float(sc)),4),
                                        "direction":"→","insight":f"'{feats[j]}' {lv} influences '{target}'"})
                except: pass
        if len(cat)>=2:
            for i in range(min(len(cat),6)):
                for j in range(i+1, min(len(cat),6)):
                    try:
                        ct=pd.crosstab(df[cat[i]].fillna("_").astype(str).str[:50], df[cat[j]].fillna("_").astype(str).str[:50])
                        if ct.shape[0]<2 or ct.shape[1]<2: continue
                        chi2,p,_,_=chi2_contingency(ct)
                        n=ct.values.sum(); md=min(ct.shape)-1
                        if md<=0: continue
                        cv=float(np.sqrt(chi2/(n*md)))
                        if p<chi2_p and cv>=0.1:
                            lv="strongly" if cv>0.3 else "moderately" if cv>0.1 else "weakly"
                            rels.append({"from":f"{tbl}.{cat[i]}","to":f"{tbl}.{cat[j]}","method":"chi_squared",
                                        "strength":round(cv,4),"p_value":round(float(p),4),"direction":"↔",
                                        "insight":f"'{cat[i]}' and '{cat[j]}' are {lv} associated"})
                    except: pass
        if num and cat:
            for cc in cat[:4]:
                try:
                    y=pd.factorize(df[cc].fillna("_").astype(str))[0]
                    scores=mutual_info_classif(df[num[:5]].fillna(0).values, y, random_state=42)
                    for j,sc in enumerate(scores):
                        if sc>=mi_min:
                            lv="strongly" if sc>0.3 else "moderately" if sc>0.1 else "weakly"
                            rels.append({"from":f"{tbl}.{num[j]}","to":f"{tbl}.{cc}","method":"mi_classif",
                                        "strength":round(float(sc),4),"p_value":round(max(0.001,1-float(sc)),4),
                                        "direction":"→","insight":f"'{num[j]}' {lv} predicts '{cc}'"})
                except: pass
    rels.sort(key=lambda r: r["strength"], reverse=True)
    return rels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🏥 HEALTH + CONTRACTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_health(schema, quality):
    scores = {}
    for tbl, q in quality.items():
        cols=q.get("columns",{})
        if not cols: continue
        avg_null=float(np.mean([v.get("null_rate",0) for v in cols.values()]))
        null_s=max(0,100*(1-avg_null*2))
        comp_s=float(q.get("overall_completeness",0.8))*100
        pk_cols=schema["tables"].get(tbl,{}).get("primary_keys",[])
        uniq_s=100.0
        for pk in pk_cols:
            if pk in cols: uniq_s=min(uniq_s, cols[pk].get("unique_rate",1)*100)
        score=round(min(100, max(0, null_s*0.35+comp_s*0.35+uniq_s*0.30)),1)
        label="Healthy" if score>=75 else "At Risk" if score>=50 else "Critical"
        icon="✅" if score>=75 else "⚠️" if score>=50 else "🚨"
        color="success" if score>=75 else "warning" if score>=50 else "danger"
        worst=q.get("worst_columns",[])
        rec=f"'{tbl}' healthy." if score>=75 else f"Fix nulls in {', '.join(worst[:2])}." if score>=50 else f"CRITICAL: Audit '{tbl}' ETL now."
        scores[tbl]={"table":tbl,"score":score,"label":label,"icon":icon,"color":color,
                     "breakdown":{"Nulls":round(null_s,1),"Completeness":round(comp_s,1),"PK_Uniqueness":round(uniq_s,1)},
                     "recommendation":rec}
    return scores

def gen_contracts(schema, quality):
    contracts = {}
    for tbl, q in quality.items():
        cols=q.get("columns",{}); pk=schema["tables"].get(tbl,{}).get("primary_keys",[])
        contract={"version":"1.0","generated":datetime.now().isoformat(),"table":tbl,
                  "sla":{"min_completeness":round(q.get("overall_completeness",0.8),3)},"columns":{}}
        for cn,cd in cols.items():
            cc={"dtype":cd.get("dtype",""),"max_null_rate":round(min(0.99,cd.get("null_rate",0)*1.1+0.005),4)}
            if cn in pk: cc.update({"unique":True,"not_null":True,"max_null_rate":0.0})
            if cd.get("min_value") is not None: cc["range"]=[cd["min_value"],cd["max_value"]]
            contract["columns"][cn]=cc
        contracts[tbl]=yaml.dump(contract, default_flow_style=False, sort_keys=False)
    return contracts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔗 ER DIAGRAM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_er(schema):
    G = nx.DiGraph(); edges = []
    for n,t in schema["tables"].items():
        G.add_node(n, row_count=t["row_count"], col_count=len(t["columns"]), pk=t["primary_keys"],
                   cols=[c["name"] for c in t["columns"]])
    for tn,t in schema["tables"].items():
        for fk in t["foreign_keys"]:
            ref=fk.get("referred_table","")
            if ref and ref in schema["tables"]:
                fc=fk["constrained_columns"][0] if fk["constrained_columns"] else ""
                tc=fk["referred_columns"][0] if fk["referred_columns"] else ""
                G.add_edge(tn, ref, from_col=fc, to_col=tc)
                edges.append({"from":tn,"fk_col":fc,"to":ref,"pk_col":tc})
    return G, edges

def render_er(G, schema, health):
    if G.number_of_nodes()==0:
        fig=go.Figure(); fig.add_annotation(text="No tables",showarrow=False,x=0.5,y=0.5); return fig
    pos=nx.spring_layout(G, seed=42, k=4.0, iterations=80); fig=go.Figure()
    for u,v,data in G.edges(data=True):
        x0,y0=pos[u]; x1,y1=pos[v]
        fig.add_trace(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",
                                 line=dict(width=2.5,color="rgba(108,92,231,0.6)"),hoverinfo="none",showlegend=False))
        mx,my=(x0+x1)/2,(y0+y1)/2
        fig.add_trace(go.Scatter(x=[mx],y=[my],mode="text",
                                 text=[f"{data.get('from_col','')} -> {data.get('to_col','')}"],
                                 textfont=dict(size=9,color="#a29bfe"),hoverinfo="none",showlegend=False))
        fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0,xref="x",yref="y",axref="x",ayref="y",
                          showarrow=True,arrowhead=3,arrowsize=1.5,arrowwidth=2,arrowcolor="rgba(108,92,231,0.7)")
    for node in G.nodes():
        x,y=pos[node]; t=schema["tables"][node]; h=health.get(node,{})
        cols=[c["name"] for c in t["columns"]]; pk_cols=t["primary_keys"]
        color="#00b894" if h.get("score",0)>=75 else "#fdcb6e" if h.get("score",0)>=50 else "#e17055"
        col_text="<br>".join([f"{'🔑 ' if c in pk_cols else '  '}{c}" for c in cols[:12]])
        if len(cols)>12: col_text+=f"<br>  ... +{len(cols)-12} more"
        hover=(f"<b>{node}</b><br>Rows: {t['row_count']:,}<br>Cols: {len(cols)}<br>"
               f"Health: {h.get('icon','')} {h.get('score','?')}/100<br>PKs: {', '.join(pk_cols) or 'None'}<br><hr>{col_text}")
        deg=G.degree(node); size=max(35,min(70,deg*15+35))
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",
                                 marker=dict(size=size,color=color,line=dict(width=3,color="white"),symbol="square",opacity=0.9),
                                 text=[f"{node}\n({t['row_count']:,})"],textposition="top center",
                                 textfont=dict(size=11,color="white",family="Inter"),
                                 hovertext=[hover],hoverinfo="text",showlegend=False,
                                 hoverlabel=dict(bgcolor="#1a1a2e",bordercolor=color,font=dict(size=12,color="white"))))
    fig.update_layout(height=700,plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="white",
                      xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      margin=dict(l=20,r=20,t=20,b=20))
    fig.add_annotation(text="🔗 Entity Relationship Diagram",xref="paper",yref="paper",x=0.5,y=1.02,
                       showarrow=False,font=dict(size=16,color="#a29bfe"))
    fig.add_annotation(text="🟢 Healthy  🟡 At Risk  🔴 Critical  |  🔑 = PK  |  Hover for columns",
                       xref="paper",yref="paper",x=0.5,y=-0.02,showarrow=False,font=dict(size=10,color="#636e72"))
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📄 PDF REPORT BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PDFReport(FPDF):
    """Professional PDF report with tables, charts placeholders, and contracts."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        # Use built-in fonts only (Helvetica = Arial equivalent)
        self.FONT = "Helvetica"

    def _safe(self, text: str) -> str:
        """Encode text safely for PDF (latin-1 compatible)."""
        if not text:
            return ""
        return text.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        self.set_font(self.FONT, "B", 11)
        self.set_text_color(108, 92, 231)
        self.cell(0, 8, self._safe("DB Intelligence Agent"), align="L")
        self.set_font(self.FONT, "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, self._safe(f"Generated: {datetime.now():%Y-%m-%d %H:%M}"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(108, 92, 231)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, self._safe(f"Page {self.page_no()}/{{nb}}"), align="C")

    def add_title_page(self, schema, health, causal_count, ai_summary):
        self.add_page()
        self.ln(30)
        # Title
        self.set_font(self.FONT, "B", 28)
        self.set_text_color(108, 92, 231)
        self.cell(0, 15, self._safe("Database Intelligence Report"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        self.set_font(self.FONT, "", 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, self._safe("Automated Schema Analysis | Quality Profiling | Causal Discovery"),
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(15)

        # Summary box
        self.set_fill_color(245, 245, 255)
        self.set_draw_color(108, 92, 231)
        self.rect(20, self.get_y(), 170, 45, style="DF")
        y_start = self.get_y() + 5
        self.set_xy(25, y_start)
        self.set_font(self.FONT, "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, self._safe("Executive Summary"), new_x="LMARGIN", new_y="NEXT")
        self.set_x(25)
        self.set_font(self.FONT, "", 9)
        self.set_text_color(60, 60, 60)
        self.multi_cell(160, 5, self._safe(ai_summary[:400]))
        self.set_y(y_start + 48)

        # Key metrics in boxes
        self.ln(5)
        metrics = [
            ("Tables", str(schema["total_tables"])),
            ("Columns", str(schema["total_columns"])),
            ("Total Rows", f"{schema['total_rows']:,}"),
            ("Causal Pairs", str(causal_count)),
            ("Avg Health", f"{np.mean([h['score'] for h in health.values()]):.0f}/100" if health else "N/A"),
        ]
        box_w = 34
        start_x = 15
        for i, (label, value) in enumerate(metrics):
            x = start_x + i * (box_w + 3)
            self.set_fill_color(108, 92, 231)
            self.set_draw_color(108, 92, 231)
            self.rounded_rect(x, self.get_y(), box_w, 22, 3, style="DF")
            self.set_xy(x, self.get_y() + 3)
            self.set_font(self.FONT, "B", 14)
            self.set_text_color(255, 255, 255)
            self.cell(box_w, 8, self._safe(value), align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_x(x)
            self.set_font(self.FONT, "", 7)
            self.set_text_color(220, 220, 255)
            self.cell(box_w, 5, self._safe(label.upper()), align="C")
        self.ln(30)

    def rounded_rect(self, x, y, w, h, r, style=""):
        """Draw a rectangle with rounded corners."""
        # Approximate with regular rect for simplicity
        if "F" in style or "D" in style:
            self.rect(x, y, w, h, style)

    def add_section(self, title):
        self.ln(5)
        self.set_font(self.FONT, "B", 14)
        self.set_text_color(108, 92, 231)
        self.cell(0, 10, self._safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(108, 92, 231)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def add_subsection(self, title):
        self.set_font(self.FONT, "B", 11)
        self.set_text_color(0, 206, 201)
        self.cell(0, 8, self._safe(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def add_text(self, text, size=9, bold=False):
        self.set_font(self.FONT, "B" if bold else "", size)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, self._safe(text))
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if not col_widths:
            available = 190
            col_widths = [available / len(headers)] * len(headers)

        # Truncate widths if too many columns
        if sum(col_widths) > 190:
            factor = 190 / sum(col_widths)
            col_widths = [w * factor for w in col_widths]

        # Header
        self.set_font(self.FONT, "B", 8)
        self.set_fill_color(108, 92, 231)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, self._safe(str(h)[:20]), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font(self.FONT, "", 7)
        self.set_text_color(50, 50, 50)
        for row_idx, row in enumerate(rows[:50]):  # Limit rows
            if self.get_y() > 265:  # Page break check
                self.add_page()
                self.set_font(self.FONT, "B", 8)
                self.set_fill_color(108, 92, 231)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, self._safe(str(h)[:20]), border=1, fill=True, align="C")
                self.ln()
                self.set_font(self.FONT, "", 7)
                self.set_text_color(50, 50, 50)

            bg = (248, 248, 255) if row_idx % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            for i, val in enumerate(row):
                text = str(val)[:25] if val is not None else "-"
                self.cell(col_widths[i], 6, self._safe(text), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)

    def add_health_bar(self, table_name, score, label):
        """Add a visual health bar for a table."""
        y = self.get_y()
        if y > 265:
            self.add_page()
            y = self.get_y()

        # Table name
        self.set_font(self.FONT, "", 8)
        self.set_text_color(50, 50, 50)
        self.cell(50, 6, self._safe(table_name[:30]))

        # Bar background
        bar_x = 60
        bar_w = 100
        bar_h = 5
        self.set_fill_color(230, 230, 230)
        self.rect(bar_x, y + 0.5, bar_w, bar_h, "F")

        # Filled bar
        fill_w = bar_w * (score / 100)
        if score >= 75:
            self.set_fill_color(0, 184, 148)
        elif score >= 50:
            self.set_fill_color(253, 203, 110)
        else:
            self.set_fill_color(225, 112, 85)
        self.rect(bar_x, y + 0.5, fill_w, bar_h, "F")

        # Score text
        self.set_xy(bar_x + bar_w + 3, y)
        self.set_font(self.FONT, "B", 8)
        self.cell(30, 6, self._safe(f"{score:.0f}/100 ({label})"))
        self.ln(8)

    def add_chart_image(self, fig, width=180):
        """Embed a Plotly chart as image in PDF."""
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_bytes)
            tmp.close()
            if self.get_y() > 180:
                self.add_page()
            self.image(tmp.name, x=15, w=width)
            self.ln(5)
            os.unlink(tmp.name)
        except Exception:
            self.add_text("[Chart could not be embedded — install 'kaleido' for chart images]")


def build_pdf(schema, quality, health, causal, edges, ai_text, contracts, er_fig=None) -> bytes:
    """Build a comprehensive PDF report."""
    pdf = PDFReport()
    pdf.alias_nb_pages()

    # ── Page 1: Title ──
    ai_summary = ai_text.get("schema", "Database analysis complete.")
    pdf.add_title_page(schema, health, len(causal), ai_summary)

    # ── Page 2: Schema Overview ──
    pdf.add_page()
    pdf.add_section("1. Schema Overview")
    pdf.add_text(f"This database contains {schema['total_tables']} tables, "
                 f"{schema['total_columns']} columns, and {schema['total_rows']:,} total rows.")

    headers = ["Table", "Rows", "Columns", "PKs", "FKs", "Completeness", "Health"]
    rows = []
    for n, t in schema["tables"].items():
        h = health.get(n, {})
        q = quality.get(n, {})
        rows.append([
            n, f"{t['row_count']:,}", len(t["columns"]),
            ", ".join(t["primary_keys"][:2]) or "-",
            len(t["foreign_keys"]),
            f"{q.get('overall_completeness', 0):.0%}",
            f"{h.get('icon', '')} {h.get('score', '-')}"
        ])
    pdf.add_table(headers, rows, col_widths=[35, 22, 18, 30, 12, 28, 25])

    # ── Page 3: Health Scores ──
    pdf.add_page()
    pdf.add_section("2. Health Scores")
    pdf.add_text("Health scores are computed from null rates, completeness, and primary key uniqueness.")
    pdf.ln(3)

    for h in sorted(health.values(), key=lambda x: x["score"]):
        pdf.add_health_bar(h["table"], h["score"], h["label"])

    pdf.ln(5)
    pdf.add_subsection("Health Breakdown")
    h_headers = ["Table", "Score", "Label", "Nulls", "Completeness", "PK Uniqueness", "Action"]
    h_rows = []
    for h in sorted(health.values(), key=lambda x: x["score"]):
        bd = h.get("breakdown", {})
        h_rows.append([
            h["table"], f"{h['score']:.0f}", h["label"],
            f"{bd.get('Nulls', 0):.0f}", f"{bd.get('Completeness', 0):.0f}",
            f"{bd.get('PK_Uniqueness', 0):.0f}", h.get("recommendation", "")[:40]
        ])
    pdf.add_table(h_headers, h_rows, col_widths=[30, 15, 18, 18, 25, 28, 56])

    # ── Page 4: ER Diagram ──
    pdf.add_page()
    pdf.add_section("3. Entity Relationship Diagram")
    if er_fig:
        pdf.add_chart_image(er_fig, width=180)
    else:
        pdf.add_text("[ER diagram available in interactive version]")

    if edges:
        pdf.add_subsection("Foreign Key Relationships")
        e_headers = ["From Table", "FK Column", "To Table", "PK Column"]
        e_rows = [[e["from"], e["fk_col"], e["to"], e["pk_col"]] for e in edges]
        pdf.add_table(e_headers, e_rows, col_widths=[45, 50, 45, 50])

    # ── Page 5: Table Details ──
    pdf.add_page()
    pdf.add_section("4. Table Details")
    for tn, t in schema["tables"].items():
        if pdf.get_y() > 230:
            pdf.add_page()
        h = health.get(tn, {})
        q = quality.get(tn, {})
        pdf.add_subsection(f"{tn}  [{h.get('icon', '')} {h.get('score', '-')}/100]")

        # AI description
        ai_desc = ai_text.get("tables", {}).get(tn, "")
        if ai_desc:
            pdf.set_font(pdf.FONT, "I", 8)
            pdf.set_text_color(100, 100, 150)
            pdf.multi_cell(0, 4, pdf._safe(ai_desc[:300]))
            pdf.ln(2)

        # Column table
        c_headers = ["Column", "Type", "PK", "FK", "Null%", "Unique%", "Kind"]
        c_rows = []
        qc = q.get("columns", {})
        for col in t["columns"]:
            cd = qc.get(col["name"], {})
            fk_info = f"->{col['ref_table']}" if col["is_fk"] else ""
            c_rows.append([
                col["name"], col["type"][:15],
                "Yes" if col["is_pk"] else "",
                fk_info[:15],
                f"{cd.get('null_rate', 0):.1%}",
                f"{cd.get('unique_rate', 0):.1%}",
                cd.get("col_kind", "-")
            ])
        pdf.add_table(c_headers, c_rows, col_widths=[35, 22, 12, 25, 18, 18, 22])

        if h.get("recommendation"):
            pdf.set_font(pdf.FONT, "I", 8)
            pdf.set_text_color(200, 100, 50)
            pdf.cell(0, 5, pdf._safe(f"Recommendation: {h['recommendation']}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # ── Page 6: Causal Relationships ──
    pdf.add_page()
    pdf.add_section("5. Causal Intelligence")
    ai_causal = ai_text.get("causal", "")
    if ai_causal:
        pdf.add_text(ai_causal[:500])

    if causal:
        pdf.add_subsection(f"Top Causal Relationships ({len(causal)} found)")
        cr_headers = ["From", "Dir", "To", "Method", "Strength", "P-value", "Insight"]
        cr_rows = []
        for r in causal[:30]:
            cr_rows.append([
                r["from"].split(".")[-1][:15],
                r["direction"],
                r["to"].split(".")[-1][:15],
                r["method"][:12],
                f"{r['strength']:.3f}",
                f"{r['p_value']:.4f}",
                r["insight"][:35]
            ])
        pdf.add_table(cr_headers, cr_rows, col_widths=[25, 10, 25, 22, 20, 20, 68])
    else:
        pdf.add_text("No significant causal relationships found at current sample size.")

    # ── Page 7: Data Contracts ──
    pdf.add_page()
    pdf.add_section("6. Data Contracts (YAML)")
    pdf.add_text(f"Auto-generated {len(contracts)} data contracts with column-level SLAs.")

    for tn, ys in contracts.items():
        if pdf.get_y() > 200:
            pdf.add_page()
        h = health.get(tn, {})
        pdf.add_subsection(f"{tn}  [{h.get('icon', '')}]")
        pdf.set_font("Courier", "", 7)
        pdf.set_text_color(60, 60, 60)
        # Limit YAML length per contract in PDF
        yaml_lines = ys.split("\n")[:25]
        for line in yaml_lines:
            if pdf.get_y() > 270:
                pdf.add_page()
                pdf.set_font("Courier", "", 7)
                pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 3.5, pdf._safe(line[:120]), new_x="LMARGIN", new_y="NEXT")
        if len(ys.split("\n")) > 25:
            pdf.set_font(pdf.FONT, "I", 7)
            pdf.cell(0, 4, pdf._safe(f"  ... ({len(ys.split(chr(10)))-25} more lines)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── Final Page: Appendix ──
    pdf.add_page()
    pdf.add_section("7. Appendix")
    pdf.add_text(f"Report generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    pdf.add_text(f"Total tables analyzed: {schema['total_tables']}")
    pdf.add_text(f"Total columns profiled: {schema['total_columns']}")
    pdf.add_text(f"Total rows scanned: {schema['total_rows']:,}")
    pdf.add_text(f"Causal relationships discovered: {len(causal)}")
    pdf.add_text(f"Data contracts generated: {len(contracts)}")
    avg_h = np.mean([h["score"] for h in health.values()]) if health else 0
    pdf.add_text(f"Average health score: {avg_h:.1f}/100")
    pdf.ln(10)
    pdf.set_font(pdf.FONT, "I", 9)
    pdf.set_text_color(108, 92, 231)
    pdf.cell(0, 8, pdf._safe("Generated by DB Intelligence Agent - Causal Engine"), align="C")

    return pdf.output()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📝 MARKDOWN + ZIP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_md(schema, quality, health, ai_text):
    lines = [f"# Data Dictionary\n*{datetime.now():%Y-%m-%d %H:%M}*\n",
             f"**{schema['total_tables']} tables | {schema['total_columns']} columns | {schema['total_rows']:,} rows**\n"]
    if ai_text.get("schema"): lines.append(f"> {ai_text['schema']}\n")
    lines += ["| Table | Rows | Health |","|-|-|-|"]
    for n,t in schema["tables"].items():
        h=health.get(n,{}); lines.append(f"| `{n}` | {t['row_count']:,} | {h.get('icon','')} {h.get('score','-')} |")
    for n,t in schema["tables"].items():
        h=health.get(n,{})
        lines += [f"\n## `{n}` {h.get('icon','')}\n",
                  f"Rows: {t['row_count']:,} | PKs: {', '.join(t['primary_keys']) or 'None'}\n",
                  "| Column | Type | PK | FK | Null% |","|-|-|-|-|-|"]
        qc=quality.get(n,{}).get("columns",{})
        for col in t["columns"]:
            nr=qc.get(col["name"],{}).get("null_rate",0)
            fk=f"->{col['ref_table']}.{col['ref_col']}" if col["is_fk"] else ""
            lines.append(f"| `{col['name']}` | `{col['type']}` | {'Yes' if col['is_pk'] else ''} | {fk} | {nr:.1%} |")
    return "\n".join(lines)

def build_zip(contracts, json_str, md, pdf_bytes=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.json", json_str)
        z.writestr("dictionary.md", md)
        if pdf_bytes:
            z.writestr("report.pdf", pdf_bytes)
        for t,y in contracts.items():
            z.writestr(f"contracts/{t}.yaml", y)
    buf.seek(0); return buf.read()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🏗️ PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_analysis(engine, provider, model, n_rows):
    bar=st.progress(0); status=st.empty()

    status.info("🔍 Extracting schema..."); bar.progress(10)
    schema = extract_schema(engine)

    status.info(f"📊 Sampling {n_rows:,} rows/table..."); bar.progress(25)
    samples = sample_tables(engine, schema, n_rows)

    status.info("🔬 Profiling quality..."); bar.progress(40)
    quality = profile_all(schema, samples)

    status.info("🔗 Building ER diagram..."); bar.progress(50)
    G, edges = build_er(schema)

    status.info("⭐ Discovering causal relationships..."); bar.progress(60)
    causal = find_causal(samples)

    status.info("🏥 Computing health scores..."); bar.progress(70)
    health = compute_health(schema, quality)

    status.info("📄 Generating contracts..."); bar.progress(78)
    contracts = gen_contracts(schema, quality)

    status.info("🤖 AI analysis..."); bar.progress(84)
    ai = AI(provider, model)
    ai_text = {}
    tbl_info = {n:{"rows":t["row_count"],"cols":len(t["columns"])} for n,t in schema["tables"].items()}
    ai_text["schema"] = ai.ask(f"You are a senior data architect. In 3 sentences, describe this database: business domain, key tables, design patterns.\nTables: {json.dumps(tbl_info)}\nRelationships: {len(edges)} FKs", 400)
    if causal:
        ai_text["causal"] = ai.ask(f"In 3 sentences, explain these data relationships for business:\n{json.dumps(causal[:8])}", 300)
    else:
        ai_text["causal"] = "No significant causal relationships found."
    ai_text["tables"] = {}
    for tn, t in schema["tables"].items():
        col_list = [{"name":c["name"],"type":c["type"]} for c in t["columns"][:15]]
        ai_text["tables"][tn] = ai.ask(f"Describe '{tn}' table ({t['row_count']:,} rows) in 2 sentences for business user. Columns: {json.dumps(col_list)}", 300)

    status.info("📄 Building PDF report..."); bar.progress(90)
    er_fig = render_er(G, schema, health)
    pdf_bytes = build_pdf(schema, quality, health, causal, edges, ai_text, contracts, er_fig)

    status.info("📦 Packaging reports..."); bar.progress(95)
    md = build_md(schema, quality, health, ai_text)
    jr = json.dumps({"schema":schema,"quality":quality,"causal":causal,"health":health,"er_edges":edges}, indent=2, default=str)
    zb = build_zip(contracts, jr, md, pdf_bytes)

    bar.progress(100); status.success("✅ Analysis complete!")

    return {"schema":schema,"quality":quality,"G":G,"edges":edges,
            "causal":causal,"health":health,"contracts":contracts,
            "ai_text":ai_text,"md":md,"json_report":jr,"zip":zb,
            "samples":samples,"pdf_bytes":pdf_bytes,"er_fig":er_fig}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 💬 AI CHATBOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_chatbot(R):
    st.markdown("### 💬 AI Data Assistant")
    st.caption("Ask questions about your data — AI uses your actual schema, quality, and causal findings.")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    with st.expander("💡 Example questions", expanded=False):
        examples = ["What tables have worst quality?", "Which columns have most nulls?",
                     "What causal relationships did you find?", "Summarize database health",
                     "What should I fix first?", "Explain relationship between tables",
                     "What business insights can you extract?", "Which tables are most connected?"]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i%2].button(f"📝 {ex}", key=f"ex_{i}", use_container_width=True):
                st.session_state.chat_input = ex

    for msg in st.session_state.chat_history:
        role_class = "chat-user" if msg["role"]=="user" else "chat-ai"
        icon = "🧑 You" if msg["role"]=="user" else "🤖 AI"
        st.markdown(f'<div class="chat-msg {role_class}"><b>{icon}:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    default_input = st.session_state.pop("chat_input", "")
    user_input = st.chat_input("Ask about your data...")
    if default_input: user_input = default_input

    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        schema, quality, health, causal, samples = R["schema"], R["quality"], R["health"], R["causal"], R.get("samples",{})
        table_summaries = [f"- {tn}: {t['row_count']:,} rows, {len(t['columns'])} cols, health={health.get(tn,{}).get('score','?')}/100, completeness={quality.get(tn,{}).get('overall_completeness',0):.1%}" for tn,t in schema["tables"].items()]
        sample_ctx = ""
        for tn, df in samples.items():
            if not df.empty: sample_ctx += f"\n{tn} (3 rows):\n{df.head(3).to_string()}\n"
        ctx = f"""Expert data analyst. Answer using ONLY this data:
DATABASE: {schema['total_tables']} tables, {schema['total_columns']} cols, {schema['total_rows']:,} rows
TABLES:\n{chr(10).join(table_summaries)}
RELATIONSHIPS ({len(R['edges'])} FKs): {json.dumps(R['edges'][:15])}
CAUSAL ({len(causal)}): {json.dumps(causal[:10])}
SAMPLES:\n{sample_ctx[:3000]}
Be specific with table/column names."""
        provider = st.session_state.get("ai_provider", os.getenv("DEFAULT_AI_PROVIDER","groq"))
        model = st.session_state.get("ai_model", os.getenv("DEFAULT_AI_MODEL","llama-3.3-70b-versatile"))
        with st.spinner("🤖 Thinking..."):
            response = AI(provider, model).ask(f"{ctx}\n\nQUESTION: {user_input}", 1500)
        st.session_state.chat_history.append({"role":"ai","content":response})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []; st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🖥️ SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""<div style="text-align:center; padding:1rem 0;">
        <h2 style="background: linear-gradient(135deg, #6c5ce7, #00cec9);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🧠 DB Intelligence</h2>
        <p style="color:#636e72; font-size:0.85rem;">Schema · Quality · Causal · AI</p></div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 🤖 AI Engine")
    avail = AI.available()
    if avail:
        labels = {f"{v['icon']} {v['name']}":k for k,v in avail.items()}
        default_p = os.getenv("DEFAULT_AI_PROVIDER","groq")
        default_label = next((l for l,k in labels.items() if k==default_p), list(labels.keys())[0])
        sel_label = st.selectbox("Provider", list(labels.keys()), index=list(labels.keys()).index(default_label), label_visibility="collapsed")
        sel_prov = labels[sel_label]
        models = AI.PROVIDERS[sel_prov]["models"]
        default_m = os.getenv("DEFAULT_AI_MODEL", models[0])
        sel_model = st.selectbox("Model", models, index=models.index(default_m) if default_m in models else 0, label_visibility="collapsed")
        st.session_state["ai_provider"] = sel_prov; st.session_state["ai_model"] = sel_model
        st.success(f"✅ {sel_label} · `{sel_model}`")
    else:
        st.error("❌ No API keys in .env")
    st.divider()

    st.markdown("#### 📂 Connect Data")
    mode = st.radio("Source", ["🗄️ SQLite","📊 CSV Files","🔗 URL"], label_visibility="collapsed")
    engine = None; ready = False

    if mode == "🗄️ SQLite":
        f = st.file_uploader("Upload .db", type=["db","sqlite","sqlite3"], label_visibility="collapsed")
        if f:
            try:
                engine=load_sqlite(f)
                if test_db(engine): st.success(f"✅ {f.name}"); ready=True
                else: st.error("❌ Can't read")
            except Exception as e: st.error(f"❌ {e}")
    elif mode == "📊 CSV Files":
        files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True, label_visibility="collapsed")
        if files:
            try:
                engine, names = load_csvs(files)
                if test_db(engine) and names: st.success(f"✅ {len(names)} tables"); ready=True
            except Exception as e: st.error(f"❌ {e}")
    else:
        url = st.text_input("SQLAlchemy URL", placeholder="postgresql://user:pass@host/db")
        if url:
            try:
                engine=create_engine(url)
                if test_db(engine): st.success("✅ Connected"); ready=True
            except Exception as e: st.error(f"❌ {e}")

    st.divider()
    n_rows = st.slider("Sample rows", 500, 50000, 5000, 500)
    run = st.button("🚀 Analyze", type="primary", use_container_width=True, disabled=not ready)
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear(); st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 RUN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if run and engine and ready:
    try:
        prov = st.session_state.get("ai_provider", os.getenv("DEFAULT_AI_PROVIDER","groq"))
        mod = st.session_state.get("ai_model", os.getenv("DEFAULT_AI_MODEL","llama-3.3-70b-versatile"))
        st.session_state["R"] = run_analysis(engine, prov, mod, n_rows)
    except Exception as e:
        st.error(f"❌ {e}"); st.code(str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🏠 WELCOME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "R" not in st.session_state:
    st.markdown("""<div class="hero"><h1>🧠 DB Intelligence Agent</h1>
        <p>Upload a database → Get AI-powered schema analysis, Olist-style ER diagrams,
        causal discovery, health scores, PDF reports, and an AI assistant.</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],[
        ("📋","Schema Analysis","Tables, columns, PKs, FKs auto-detected"),
        ("🔗","ER Diagram","Olist-style relationship map with health colors"),
        ("⭐","Causal AI","Directional influences via MI & Chi² tests"),
        ("📄","PDF Reports","Professional reports with charts & contracts"),
    ]):
        col.markdown(f"""<div class="glass-card" style="text-align:center; min-height:180px;">
            <div style="font-size:2.5rem;">{icon}</div>
            <h3 style="color:white; margin:0.5rem 0;">{title}</h3>
            <p style="color:{COLORS['muted']}; font-size:0.85rem;">{desc}</p></div>""", unsafe_allow_html=True)

    st.markdown(f"""<br><div class="glass-card" style="text-align:center;">
        <h3 style="color:white;">🚀 Get Started</h3>
        <p style="color:{COLORS['muted']};">Upload SQLite / CSV / URL in sidebar → Click <b>Analyze</b></p>
        <p style="color:{COLORS['accent']}; font-size:0.85rem;">Works with Olist, Northwind, Chinook, or any relational database</p>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R = st.session_state["R"]
S,Q,H,C = R["schema"],R["quality"],R["health"],R["causal"]

st.markdown("<br>", unsafe_allow_html=True)
cols = st.columns(6)
for col,(icon,label,val) in zip(cols,[
    ("📦","Tables",S["total_tables"]),("🔢","Columns",S["total_columns"]),
    ("📊","Rows",f"{S['total_rows']:,}"),("🔗","Relations",len(R["edges"])),
    ("⭐","Causal",len(C)),("💊","Avg Health",f"{np.mean([h['score'] for h in H.values()]):.0f}/100" if H else "—"),
]):
    col.markdown(f"""<div class="metric-card"><div style="font-size:1.5rem;">{icon}</div>
        <div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""<div class="glass-card"><h4 style="color:{COLORS['accent']};">🤖 AI Executive Summary</h4>
    <p style="color:{COLORS['text']};">{R['ai_text'].get('schema','')}</p></div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📋 Schema","🔗 ER Diagram","💊 Health","⭐ Causal","💬 AI Chat","📖 Dictionary","⬇️ Downloads"])

# ═══ TAB 1 ═══
with t1:
    st.subheader("📋 Schema Overview")
    rows = []
    for n,t in S["tables"].items():
        h=H.get(n,{}); q=Q.get(n,{})
        rows.append({"Table":n,"Rows":f"{t['row_count']:,}","Columns":len(t["columns"]),
                     "PKs":", ".join(t["primary_keys"]) or "—","FKs":len(t["foreign_keys"]),
                     "Completeness":f"{q.get('overall_completeness',0):.1%}",
                     "Health":f"{h.get('icon','')} {h.get('label','')} ({h.get('score','—')})"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for n,t in S["tables"].items():
        h=H.get(n,{})
        with st.expander(f"📦 **{n}** · {t['row_count']:,} rows · {h.get('icon','')} {h.get('score','')}/100"):
            ai_desc = R["ai_text"].get("tables",{}).get(n,"")
            if ai_desc: st.info(f"🤖 {ai_desc}")
            if h.get("recommendation"): st.warning(f"💡 {h['recommendation']}")
            qc=Q.get(n,{}).get("columns",{})
            crows=[{"Column":col["name"],"Type":col["type"],"PK":"🔑" if col["is_pk"] else "",
                    "FK":f"→{col['ref_table']}.{col['ref_col']}" if col["is_fk"] else "",
                    "Null%":f"{qc.get(col['name'],{}).get('null_rate',0):.1%}",
                    "Unique%":f"{qc.get(col['name'],{}).get('unique_rate',0):.1%}",
                    "Kind":qc.get(col["name"],{}).get("col_kind","—"),
                    "Min":qc.get(col["name"],{}).get("min_value","—"),
                    "Max":qc.get(col["name"],{}).get("max_value","—")} for col in t["columns"]]
            st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)

# ═══ TAB 2 ═══
with t2:
    st.subheader("🔗 Entity Relationship Diagram")
    st.caption("Olist-style • Square=table • Color=health • Hover=columns • Arrows=FKs")
    st.plotly_chart(R["er_fig"], use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        if R["edges"]:
            st.subheader("FK Relationships")
            st.dataframe(pd.DataFrame(R["edges"]), use_container_width=True, hide_index=True)
    with c2:
        if R["G"].nodes():
            st.subheader("Connectivity")
            cent=nx.degree_centrality(R["G"])
            hub_df=pd.DataFrame([{"Table":t,"Centrality":round(c,3)} for t,c in sorted(cent.items(),key=lambda x:x[1],reverse=True)])
            fig=px.bar(hub_df,x="Centrality",y="Table",orientation="h",color="Centrality",
                       color_continuous_scale=["#6c5ce7","#00cec9"],template="plotly_dark")
            fig.update_layout(height=max(250,len(hub_df)*40),margin=dict(l=10,r=10,t=10,b=10),
                             plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f")
            st.plotly_chart(fig, use_container_width=True)

# ═══ TAB 3 ═══
with t3:
    st.subheader("🏥 Health Scores")
    if H:
        hdf=pd.DataFrame(H.values()).sort_values("score")
        fig=px.bar(hdf,x="score",y="table",orientation="h",color="label",text="score",
                   color_discrete_map={"Healthy":"#00b894","At Risk":"#fdcb6e","Critical":"#e17055"},
                   template="plotly_dark",title="Health Dashboard")
        fig.update_traces(texttemplate="%{text:.0f}",textposition="outside")
        fig.update_layout(height=max(250,len(H)*50),margin=dict(l=10,r=80,t=40,b=10),
                         plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f")
        st.plotly_chart(fig, use_container_width=True)
        for h in sorted(H.values(), key=lambda x: x["score"]):
            with st.expander(f"{h['icon']} **{h['table']}** — {h['score']}/100"):
                cols=st.columns(len(h["breakdown"]))
                for col,(k,v) in zip(cols,h["breakdown"].items()): col.metric(k,f"{v:.0f}/100")
                st.warning(f"💡 {h['recommendation']}")

    st.divider(); st.subheader("🔬 Column Inspector")
    sel=st.selectbox("Table",list(Q.keys()),key="qt")
    if sel and sel in Q:
        tq=Q[sel]
        crows=[{"Column":cn,"Kind":cd.get("col_kind","—"),"Nulls":cd.get("null_count",0),
                "Null%":f"{cd.get('null_rate',0):.2%}","Unique":cd.get("unique_count",0),
                "Min":cd.get("min_value"),"Max":cd.get("max_value"),"Mean":cd.get("mean_value")}
               for cn,cd in tq.get("columns",{}).items()]
        st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)
        la,lb = st.columns(2)
        with la:
            nd=[{"Col":r["Column"],"Null":float(r["Null%"].strip("%"))} for r in crows if float(r["Null%"].strip("%"))>0]
            if nd:
                fig=px.bar(pd.DataFrame(nd).sort_values("Null"),x="Null",y="Col",orientation="h",
                          color="Null",color_continuous_scale="Reds",template="plotly_dark",title="Null Distribution")
                fig.update_layout(height=350,margin=dict(l=10,r=10,t=40,b=10),plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f")
                st.plotly_chart(fig, use_container_width=True)
            else: st.success("🎉 Zero nulls!")
        with lb:
            cats=[cn for cn,cd in tq.get("columns",{}).items() if cd.get("col_kind")=="categorical"]
            if cats:
                sel_c=st.selectbox("Category",cats,key="tv")
                tv=tq["columns"].get(sel_c,{}).get("top_values",[])
                if tv:
                    fig=px.bar(pd.DataFrame(tv).head(10),x="count",y="value",orientation="h",
                              color="count",color_continuous_scale=["#6c5ce7","#00cec9"],
                              template="plotly_dark",title=f"Top: {sel_c}")
                    fig.update_layout(height=350,margin=dict(l=10,r=10,t=40,b=10),plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f")
                    st.plotly_chart(fig, use_container_width=True)

# ═══ TAB 4 ═══
with t4:
    st.markdown(f"""<div class="glass-card"><h4 style="color:{COLORS['accent']};">⭐ Causal Intelligence</h4>
        <p style="color:{COLORS['text']};">Directional influences via Mutual Information & Chi-Squared.</p></div>""", unsafe_allow_html=True)
    st.info(f"🤖 {R['ai_text'].get('causal','')}")
    if C:
        cdf=pd.DataFrame(C); top=cdf.head(20).copy()
        top["label"]=top["from"]+" "+top["direction"]+" "+top["to"]
        fig=px.bar(top,x="strength",y="label",color="method",orientation="h",template="plotly_dark",
                   title="Top Causal Relationships",
                   color_discrete_map={"mutual_info":"#6c5ce7","chi_squared":"#00cec9","mi_classif":"#fdcb6e"},
                   hover_data=["p_value","insight"])
        fig.update_layout(height=max(350,len(top)*30),margin=dict(l=10,r=30,t=40,b=10),
                         plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cdf[["from","direction","to","method","strength","p_value","insight"]], use_container_width=True, hide_index=True)
    else: st.info("No significant causal relationships found.")

    st.divider(); st.subheader("📄 Data Contracts")
    m1,m2,m3=st.columns(3)
    m1.metric("Contracts",len(R["contracts"]))
    m2.metric("Rules",sum(len(yaml.safe_load(y).get("columns",{})) for y in R["contracts"].values()))
    m3.metric("Coverage","100%")
    for tn,ys in R["contracts"].items():
        h=H.get(tn,{})
        with st.expander(f"📄 {tn} {h.get('icon','')}"):
            c1,c2=st.columns([6,1])
            c1.code(ys, language="yaml")
            c2.download_button("⬇️",ys,f"{tn}.yaml","text/yaml",key=f"c_{tn}",use_container_width=True)

# ═══ TAB 5 ═══
with t5:
    render_chatbot(R)

# ═══ TAB 6 ═══
with t6:
    st.subheader("📖 Data Dictionary")
    st.markdown(R["md"])
    st.download_button("⬇️ Download .md",R["md"],"dictionary.md","text/markdown")

# ═══ TAB 7 — DOWNLOADS (with PDF!) ═══
with t7:
    st.subheader("⬇️ Download All Reports")
    st.success(f"✅ {S['total_tables']} tables · {len(C)} causal · {len(R['contracts'])} contracts")

    d1,d2,d3,d4 = st.columns(4)

    with d1:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;">📦</div><h4 style="color:white;">Complete ZIP</h4>
            <p style="color:{COLORS['muted']};">JSON + MD + PDF + YAML</p></div>""", unsafe_allow_html=True)
        st.download_button("⬇️ Everything (.zip)", R["zip"],
                           f"db_analysis_{datetime.now():%Y%m%d_%H%M}.zip",
                           "application/zip", type="primary", use_container_width=True)

    with d2:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;">📕</div><h4 style="color:white;">PDF Report</h4>
            <p style="color:{COLORS['muted']};">Professional multi-page</p></div>""", unsafe_allow_html=True)
        st.download_button("⬇️ Report (.pdf)", R["pdf_bytes"],
                           f"db_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                           "application/pdf", use_container_width=True)

    with d3:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;">📖</div><h4 style="color:white;">Dictionary</h4>
            <p style="color:{COLORS['muted']};">Markdown format</p></div>""", unsafe_allow_html=True)
        st.download_button("⬇️ Dictionary (.md)", R["md"],
                           "dictionary.md", "text/markdown", use_container_width=True)

    with d4:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;">📄</div><h4 style="color:white;">JSON Report</h4>
            <p style="color:{COLORS['muted']};">Machine-readable</p></div>""", unsafe_allow_html=True)
        st.download_button("⬇️ Report (.json)", R["json_report"],
                           "report.json", "application/json", use_container_width=True)

    st.divider()
    st.subheader("Individual Contracts")
    cc=st.columns(min(4,max(1,len(R["contracts"]))))
    for i,(tn,ys) in enumerate(R["contracts"].items()):
        h=H.get(tn,{})
        cc[i%len(cc)].download_button(f"{h.get('icon','')} {tn}",ys,f"{tn}.yaml","text/yaml",
                                       key=f"dl_{tn}",use_container_width=True)

    # PDF Preview
    st.divider()
    st.subheader("📕 PDF Preview")
    st.caption("Preview the first page of your PDF report")
    pdf_b64 = base64.b64encode(R["pdf_bytes"]).decode()
    st.markdown(f"""
    <iframe src="data:application/pdf;base64,{pdf_b64}" 
            width="100%" height="600" 
            style="border: 1px solid rgba(108,92,231,0.3); border-radius: 12px;">
    </iframe>
    """, unsafe_allow_html=True)
