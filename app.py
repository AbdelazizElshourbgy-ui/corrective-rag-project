import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="NEXUS | Corrective RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- FUTURISTIC UI -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #061316;
    --panel: #0b2023;
    --panel2: #0d292c;
    --teal: #20f2c2;
    --teal2: #00bfa5;
    --muted: #8eaaa8;
    --line: rgba(32,242,194,.20);
}
html, body, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse at 15% 0%, rgba(0,191,165,.15), transparent 34%),
      radial-gradient(ellipse at 90% 12%, rgba(32,242,194,.10), transparent 30%),
      linear-gradient(135deg, #040d10 0%, #07191c 48%, #061316 100%);
    color: #eafffa;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3, p, label, span, div { font-family: 'Inter', sans-serif; }
.hero {
    position: relative; overflow: hidden;
    padding: 34px 38px 30px; margin: 4px 0 28px;
    border: 1px solid rgba(32,242,194,.32);
    border-radius: 24px;
    background: linear-gradient(115deg, rgba(10,37,40,.96), rgba(5,20,24,.92));
    box-shadow: 0 0 35px rgba(0,220,180,.10), inset 0 0 35px rgba(32,242,194,.035);
}
.hero:after {
    content:""; position:absolute; width:360px; height:360px; right:-90px; top:-210px;
    border:1px solid rgba(32,242,194,.20); border-radius:50%;
    box-shadow: 0 0 0 28px rgba(32,242,194,.025), 0 0 0 58px rgba(32,242,194,.02), 0 0 80px rgba(32,242,194,.10);
}
.eyebrow {
    color: var(--teal); font-family:'Space Grotesk',sans-serif;
    letter-spacing: 4px; font-size: 11px; font-weight: 700;
}
.hero-title {
    font-family:'Space Grotesk',sans-serif; font-size: clamp(34px,5vw,58px);
    font-weight:800; letter-spacing:-2px; line-height:1.05; margin:12px 0 10px;
    color:#f0fffc;
}
.hero-title span { color:var(--teal); text-shadow:0 0 22px rgba(32,242,194,.55); }
.hero-sub { color:#a5c3bf; font-size:14px; max-width:650px; line-height:1.7; }
.status-pill {
    display:inline-flex; align-items:center; gap:9px; margin-top:20px;
    padding:8px 13px; border:1px solid rgba(32,242,194,.30);
    border-radius:30px; color:#b9fff0; background:rgba(32,242,194,.07);
    font-size:11px; letter-spacing:1.5px; font-weight:700;
}
.status-dot { width:7px; height:7px; border-radius:50%; background:var(--teal); box-shadow:0 0 12px var(--teal); }
.section-kicker {
    color:var(--teal); font-family:'Space Grotesk',sans-serif;
    font-size:11px; font-weight:700; letter-spacing:2.5px; margin:8px 0 10px;
}
.section-title { color:#edfffb; font-size:22px; font-weight:750; margin:0 0 5px; }
.section-desc { color:#8eaaa8; font-size:13px; margin:0 0 18px; }
.panel {
    background:linear-gradient(145deg, rgba(13,41,44,.92), rgba(7,25,29,.96));
    border:1px solid var(--line); border-radius:18px;
    padding:22px 24px; box-shadow:0 10px 30px rgba(0,0,0,.18), inset 0 1px rgba(255,255,255,.025);
}
.metric {
    background:linear-gradient(145deg, rgba(12,42,44,.95), rgba(7,25,29,.95));
    border:1px solid rgba(32,242,194,.19); border-radius:16px;
    padding:17px 18px; min-height:92px;
}
.metric-label { color:#8eaaa8; font-size:10px; letter-spacing:1.7px; font-weight:700; }
.metric-value { color:var(--teal); font-family:'Space Grotesk',sans-serif; font-size:25px; font-weight:700; margin-top:7px; text-shadow:0 0 15px rgba(32,242,194,.25); }
div[data-testid="stTextInput"] label, div[data-testid="stFileUploader"] label,
div[data-testid="stSlider"] label { color:#d5eeea !important; font-weight:600 !important; font-size:13px !important; }
div[data-testid="stTextInput"] input {
    background:#07191c !important; color:#eafffa !important;
    border:1px solid rgba(32,242,194,.28) !important; border-radius:12px !important;
    min-height:48px; box-shadow:inset 0 0 15px rgba(0,0,0,.15);
}
div[data-testid="stTextInput"] input:focus {
    border-color:var(--teal) !important; box-shadow:0 0 0 1px var(--teal), 0 0 18px rgba(32,242,194,.15) !important;
}
div[data-testid="stFileUploader"] section {
    background:rgba(7,25,29,.78) !important; border:1px dashed rgba(32,242,194,.45) !important;
    border-radius:15px !important;
}
div[data-testid="stFileUploader"] section:hover { border-color:var(--teal) !important; background:rgba(32,242,194,.045) !important; }
div[data-testid="stFileUploader"] button {
    color:var(--teal) !important; border:1px solid rgba(32,242,194,.45) !important;
    background:rgba(32,242,194,.08) !important; border-radius:9px !important;
}
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width:100%; min-height:52px; border:1px solid var(--teal) !important;
    border-radius:12px !important; color:#031614 !important;
    background:linear-gradient(100deg,#20f2c2,#00cba9) !important;
    font-family:'Space Grotesk',sans-serif !important; font-weight:900 !important;
    letter-spacing:2px !important; box-shadow:0 0 18px rgba(32,242,194,.28), 0 0 38px rgba(32,242,194,.10);
    transition:all .22s ease !important;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    transform:translateY(-2px); box-shadow:0 0 25px rgba(32,242,194,.48), 0 0 55px rgba(32,242,194,.18);
    color:#00110f !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--teal) !important; }
div[data-testid="stSlider"] [data-testid="stTickBar"] { background:rgba(32,242,194,.25) !important; }
.answer-box {
    padding:22px 24px; border-radius:16px; border:1px solid rgba(32,242,194,.35);
    background:linear-gradient(130deg,rgba(10,48,47,.75),rgba(7,25,29,.95));
    box-shadow:0 0 25px rgba(32,242,194,.08); color:#eafffa; line-height:1.8;
}
.source-chip {
    display:inline-block; margin:5px 6px 0 0; padding:7px 11px; border-radius:8px;
    border:1px solid rgba(32,242,194,.28); background:rgba(32,242,194,.07);
    color:#aaffeb; font-family:monospace; font-size:12px;
}
hr { border-color:rgba(32,242,194,.13) !important; }
div[data-testid="stExpander"] {
    border:1px solid rgba(32,242,194,.18) !important; border-radius:12px !important;
    background:rgba(8,29,32,.7) !important;
}
div[data-testid="stExpander"] summary { color:#dffff8 !important; }
.stAlert { border-radius:12px !important; }
.footer {
    text-align:center; color:#587875; font-size:10px; letter-spacing:2px;
    padding-top:30px; margin-top:38px; border-top:1px solid rgba(32,242,194,.12);
}
[data-testid="stCaptionContainer"] { color:#8eaaa8 !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- HEADER -----------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">INTELLIGENCE • RETRIEVAL • EVALUATION</div>
  <div class="hero-title">NEXUS <span>RAG</span></div>
  <div class="hero-sub">Corrective Retrieval-Augmented Generation system. Ask questions, retrieve relevant evidence, and generate grounded answers from your knowledge base.</div>
  <div class="status-pill"><span class="status-dot"></span> SYSTEM READY &nbsp; / &nbsp; EVIDENCE-BASED MODE</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------- CORE LOGIC -----------------------------
@st.cache_resource
def build_index(texts):
    word = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=180000, sublinear_tf=True)
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), max_features=120000, sublinear_tf=True)
    return word, char, word.fit_transform(texts), char.fit_transform(texts)

STOPWORDS = set("the a an and or but is are was were to of in on for with from by as at be can how what when where why which who does do did".split())

def important_terms(text, limit=8):
    tokens = re.findall(r"[A-Za-z0-9_./-]+", str(text).lower())
    terms = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    return list(dict.fromkeys(terms))[:limit]

def retrieve(q, df, idx, k=5):
    w, c, wm, cm = idx

    def once(query):
        sw = cosine_similarity(w.transform([query]), wm).ravel()
        sc = cosine_similarity(c.transform([query]), cm).ravel()
        scores = .75 * sw + .25 * sc
        inds = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in inds]

    first = once(q)
    best = first[0][1] if first else 0.0
    retry = best < .055
    merged = {i: s for i, s in first}
    if retry:
        for i, s in once(q + " " + " ".join(important_terms(q, limit=10))):
            merged[i] = max(merged.get(i, 0), s)
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(df.iloc[i], s) for i, s in ranked], retry, best

def grounded_answer(q, results):
    if not results:
        return "I could not find sufficient evidence in the knowledge base.", []
    qterms = set(important_terms(q, limit=30))
    candidates = []
    for row, score in results:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\\s+|\\n+", str(row["chunk_text"])) if len(s.strip()) > 25]
        for sent in sents:
            overlap = len(qterms & set(important_terms(sent, limit=100))) / max(1, len(qterms))
            candidates.append((.65 * score + .35 * overlap, sent, str(row["chunk_id"])))
    if not candidates:
        row, _ = results[0]
        return str(row["chunk_text"])[:700], [str(row["chunk_id"])]
    candidates.sort(key=lambda x: x[0], reverse=True)
    answer, citations = [], []
    for _, sent, cid in candidates:
        if sent not in answer:
            answer.append(sent)
            if cid not in citations:
                citations.append(cid)
        if len(answer) >= 3 or len(citations) >= 2:
            break
    if not answer:
        return "I could not find sufficient evidence in the knowledge base.", []
    return " ".join(answer), citations

# ----------------------------- DATA UPLOAD -----------------------------
st.markdown('<div class="section-kicker">01 / KNOWLEDGE BASE</div><div class="section-title">Connect your data</div><div class="section-desc">Upload the corpus file to initialize the retrieval engine.</div>', unsafe_allow_html=True)

upload_col, path_col = st.columns([1.15, 1], gap="large")
with upload_col:
    uploaded = st.file_uploader("UPLOAD CORPUS.CSV", type=["csv"], help="Choose your Kaggle corpus.csv file.")
with path_col:
    path = st.text_input("OR USE A LOCAL FILE PATH", placeholder="e.g. ./data/corpus.csv")

df = None
load_error = None
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded).fillna("")
    elif path.strip():
        df = pd.read_csv(path.strip()).fillna("")
except Exception as exc:
    load_error = str(exc)

if load_error:
    st.error(f"Could not load the CSV file: {load_error}")

if df is None:
    st.markdown("""
    <div class="panel" style="margin-top:12px;">
      <div class="section-kicker">WAITING FOR INPUT</div>
      <div style="color:#d9f5ef;font-size:15px;font-weight:600;">Your knowledge base will appear here</div>
      <div style="color:#86a6a1;font-size:12px;margin-top:7px;">Supported format: CSV · Required columns: chunk_id, chunk_text</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="footer">NEXUS RAG SYSTEM &nbsp; • &nbsp; POWERED BY TF-IDF RETRIEVAL</div>', unsafe_allow_html=True)
    st.stop()

for col in ["doc_id", "title", "doc_type"]:
    if col not in df.columns:
        df[col] = ""
if not {"chunk_id", "chunk_text"}.issubset(df.columns):
    st.error("CSV must contain chunk_id and chunk_text columns.")
    st.stop()

df["chunk_id"] = df["chunk_id"].astype(str)
df["chunk_text"] = df["chunk_text"].astype(str)
df["search_text"] = (df["title"].astype(str) + " " + df["doc_type"].astype(str) + " " + df["chunk_text"]).str.strip()
df = df.drop_duplicates("chunk_id").reset_index(drop=True)

with st.spinner("Initializing neural retrieval index..."):
    index = build_index(df["search_text"].tolist())

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric"><div class="metric-label">KNOWLEDGE CHUNKS</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric"><div class="metric-label">UNIQUE DOCUMENTS</div><div class="metric-value">{df["doc_id"].nunique():,}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric"><div class="metric-label">ENGINE STATUS</div><div class="metric-value" style="font-size:20px;">ONLINE ●</div></div>', unsafe_allow_html=True)

st.markdown('<hr style="margin:34px 0 26px;">', unsafe_allow_html=True)

# ----------------------------- QUERY CONSOLE -----------------------------
st.markdown('<div class="section-kicker">02 / QUERY CONSOLE</div><div class="section-title">Ask your knowledge base</div><div class="section-desc">Enter a question and configure how many evidence chunks should be retrieved.</div>', unsafe_allow_html=True)

with st.form("query_form", clear_on_submit=False):
    question = st.text_input("YOUR QUESTION", placeholder="Type your question here...")
    k = st.slider("NUMBER OF RETRIEVED CHUNKS", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("✦  SUBMIT QUERY  ✦", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Searching knowledge base and evaluating evidence..."):
            results, retry, score = retrieve(question, df, index, k)
            answer, citations = grounded_answer(question, results)

        st.markdown('<hr style="margin:32px 0 26px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">03 / INTELLIGENCE OUTPUT</div><div class="section-title">Analysis results</div><div class="section-desc">Generated response with traceable supporting evidence.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-kicker" style="margin-top:22px;">GENERATED ANSWER</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="section-kicker" style="margin-top:22px;">SOURCE REFERENCES</div>', unsafe_allow_html=True)
            chips = "".join(f'<span class="source-chip">{cid}</span>' for cid in citations) if citations else '<span style="color:#8eaaa8;">No source found</span>'
            st.markdown(f'<div class="panel">{chips}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-kicker" style="margin-top:22px;">RETRIEVAL DIAGNOSTICS</div>', unsafe_allow_html=True)
            retry_label = "YES — RETRY ACTIVATED" if retry else "NO — FIRST PASS"
            st.markdown(f"""
            <div class="panel">
              <div class="metric-label">CORRECTIVE RETRIEVAL</div>
              <div style="color:#20f2c2;font-weight:700;margin:6px 0 14px;">{retry_label}</div>
              <div class="metric-label">INITIAL MATCH SCORE</div>
              <div style="color:#eafffa;font-family:monospace;font-size:21px;margin-top:6px;">{score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-kicker" style="margin-top:28px;">EVIDENCE EXPLORER</div><div class="section-title" style="font-size:18px;">Retrieved chunks</div>', unsafe_allow_html=True)
        for row, similarity in results:
            title = str(row["title"]) if str(row["title"]).strip() else "Untitled document"
            with st.expander(f"{row['chunk_id']}  |  {title}  |  MATCH {similarity:.4f}"):
                st.caption(f"{row['doc_id']}  ·  {row['doc_type']}")
                st.write(row["chunk_text"])

st.markdown('<div class="footer">NEXUS RAG SYSTEM &nbsp; • &nbsp; RETRIEVE WITH PRECISION. ANSWER WITH EVIDENCE.</div>', unsafe_allow_html=True)
