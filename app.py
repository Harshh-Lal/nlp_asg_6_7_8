"""
app.py - IntelliDesk: Intelligent Query Classification & NLP Lab
Tab 1: Assignment 6 (TF-IDF Text Classification)
Tab 2: Assignment 7 (N-Gram Language Models, Smoothing & Perplexity)
Tab 3: Assignment 8 (Latent Semantic Indexing using TF-IDF & Truncated SVD)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Module 1 Imports (TF-IDF Classification)
from tfidf_classifier import load_dataset, TFIDFClassifierPipeline

# Module 2 Imports (Scratch N-Gram Engine - Pure Standard Library)
from ngram_engine import (
    NGramLanguageModel,
    DEFAULT_LAB_CORPUS,
    SUPPORT_TICKET_CORPUS
)

# Module 3 Imports (Latent Semantic Indexing & Truncated SVD)
from lsi_engine import (
    LatentSemanticIndexingEngine,
    LAB_MANUAL_ML_CORPUS,
    CUSTOMER_SUPPORT_OPINION_CORPUS,
    DEERWESTER_IR_CORPUS
)

# ==============================================================================
# PAGE CONFIGURATION & CUSTOM STYLES
# ==============================================================================
st.set_page_config(
    page_title="IntelliDesk — NLP Intelligence Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech CSS styling
st.markdown("""
<style>
    /* Global accents */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .verif-box {
        background: #0f172a;
        border: 1px solid #0284c7;
        border-left: 5px solid #0284c7;
        border-radius: 8px;
        padding: 16px;
        margin: 14px 0;
    }
    .match-badge {
        background-color: #065f46;
        color: #6ee7b7;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 700;
        border: 1px solid #10b981;
        display: inline-block;
    }
    .step-card {
        background: #1e293b;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .gen-box {
        background: #0f172a;
        border: 1px solid #8b5cf6;
        border-left: 6px solid #8b5cf6;
        border-radius: 10px;
        padding: 20px;
        margin: 16px 0;
        font-size: 18px;
        color: #e2e8f0;
        font-style: italic;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 22px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS (SEPARATED FOR ASG 6, ASG 7 & ASG 8)
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=55)
    st.title("⚡ IntelliDesk")
    st.caption("AI-Powered Support Query Routing & NLP Lab Suite")
    st.markdown("---")

    sidebar_module_focus = st.radio(
        "🎛️ Configure Settings For:",
        options=[
            "📊 Assignment 6 (TF-IDF)",
            "📖 Assignment 7 (N-Gram Models)",
            "📑 Assignment 8 (Latent Semantic Indexing)"
        ],
        index=0
    )

    if "Assignment 6" in sidebar_module_focus:
        st.markdown("### 📊 Asg 6: Dataset & Split")
        dataset_choice = st.radio(
            "Domain Support Dataset:",
            options=["🏦 Banking Support", "🎓 Student Query Helpdesk"],
            index=0,
            help="Switch between Banking Support (ATM, Account, Card, Loan, UPI) and Student Query (Exam, Placement, Fees, Hostel)"
        )
        domain_key = "banking" if "Banking" in dataset_choice else "student"

        st.markdown("---")
        st.markdown("### ⚙️ TF-IDF Parameters")
        st.markdown("""
        - **Case**: `lowercase=True`
        - **Stop Words**: `english`
        - **Formula**: `smooth_idf=True`
        - **Norm**: `L2 Euclidean`
        """)
        asg6_test_size = st.slider("Test Split Ratio:", min_value=0.15, max_value=0.35, value=0.25, step=0.05)
        asg6_seed = st.number_input("Random Seed (Asg 6):", min_value=1, max_value=999, value=42, step=1)

    elif "Assignment 7" in sidebar_module_focus:
        st.markdown("### 📖 Asg 7: Training Corpus")
        corpus_choice = st.radio(
            "Select Training Corpus:",
            options=[
                "📚 Default Lab Manual NLP Corpus (15 sentences)",
                "🎫 Support Ticket Domain Corpus (113 sentences)",
                "✍️ Custom User Corpus"
            ],
            index=0
        )
        st.markdown("---")
        st.markdown("### ⚙️ Scratch N-Gram Engine")
        st.markdown("""
        - **Library**: `collections.Counter`, `math`, `random`, `re`
        - **Zero External NLP Libraries**
        - **Boundary Tokens**: `<s>`, `</s>`
        - **Smoothing**: Laplace (Add-1)
        """)

    else:
        st.markdown("### 📑 Asg 8: LSI Corpus & Topics")
        lsi_corpus_choice = st.radio(
            "Select Corpus for LSI:",
            options=[
                "(a) 📚 Exact 5-Document ML/NLP Corpus (Lab Manual)",
                "(b) 💬 Customer Support / Opinion Mining Corpus (12 Reviews)",
                "(c) 🏛️ Classic Deerwester IR Corpus (9 Docs)"
            ],
            index=0
        )
        
        # Max components depending on corpus
        max_k_default = 4 if "(a)" in lsi_corpus_choice else 6
        lsi_k_components = st.slider("Latent Concepts (k components):", min_value=2, max_value=max_k_default, value=2, step=1)

        st.markdown("---")
        st.markdown("### ⚙️ SVD Hyperparameters")
        st.markdown("""
        - **Vectorizer**: `TfidfVectorizer(stop_words='english')`
        - **Decomposition**: `TruncatedSVD(random_state=42)`
        - **Similarity**: Cosine in Latent Space
        """)

    st.markdown("---")
    st.caption("IntelliDesk NLP Lab • Academic Session 2026")

# Set sensible defaults for whichever module is not focused
if "domain_key" not in locals():
    domain_key = "banking"
    dataset_choice = "🏦 Banking Support"
    asg6_test_size = 0.25
    asg6_seed = 42

if "corpus_choice" not in locals():
    corpus_choice = "📚 Default Lab Manual NLP Corpus (15 sentences)"

if "lsi_corpus_choice" not in locals():
    lsi_corpus_choice = "(a) 📚 Exact 5-Document ML/NLP Corpus (Lab Manual)"
    lsi_k_components = 2

# ==============================================================================
# PIPELINE CACHING (ASG 6, ASG 7 & ASG 8)
# ==============================================================================

# Module 1 Pipeline Caching
@st.cache_data(show_spinner=False)
def get_cached_tfidf_pipeline(domain: str, split_ratio: float, seed: int):
    df_data = load_dataset(domain)
    pipeline_obj = TFIDFClassifierPipeline(lowercase=True, stop_words="english")
    bench = pipeline_obj.fit_and_benchmark(df_data, test_size=split_ratio, random_state=seed)
    return df_data, pipeline_obj, bench

df_asg6, pipeline_asg6, benchmark_results_asg6 = get_cached_tfidf_pipeline(domain_key, asg6_test_size, asg6_seed)

# Module 2 N-Gram Model Caching
@st.cache_resource(show_spinner=False)
def get_cached_ngram_model(choice: str, custom_text: str = ""):
    if "Default Lab" in choice:
        corpus = DEFAULT_LAB_CORPUS
    elif "Support Ticket" in choice:
        corpus = SUPPORT_TICKET_CORPUS
    else:
        lines = [line.strip() for line in custom_text.split("\n") if line.strip()]
        corpus = lines if lines else DEFAULT_LAB_CORPUS
    model = NGramLanguageModel(corpus)
    return model, corpus

custom_corpus_text = st.session_state.get("custom_corpus_text", "")
ngram_model, active_corpus_list = get_cached_ngram_model(corpus_choice, custom_corpus_text)

# Module 3 LSI Engine Caching
@st.cache_resource(show_spinner=False)
def get_cached_lsi_engine(corpus_label: str, k_comp: int):
    if "(a)" in corpus_label:
        corpus = LAB_MANUAL_ML_CORPUS
    elif "(b)" in corpus_label:
        corpus = CUSTOMER_SUPPORT_OPINION_CORPUS
    else:
        corpus = DEERWESTER_IR_CORPUS
    engine = LatentSemanticIndexingEngine(stop_words="english", random_state=42)
    engine.fit(corpus, n_components=k_comp)
    return engine, corpus

lsi_engine, lsi_corpus_active = get_cached_lsi_engine(lsi_corpus_choice, lsi_k_components)

# ==============================================================================
# MAIN TABS (CLEAN SEPARATION OF ASSIGNMENT 6, 7 & 8)
# ==============================================================================
tab_asg6, tab_asg7, tab_asg8 = st.tabs([
    "📊 Assignment 6: TF-IDF Text Classification",
    "📖 Assignment 7: N-Gram Language Models, Smoothing & Perplexity",
    "📑 Assignment 8: Latent Semantic Indexing (LSI) & Truncated SVD"
])


# ==============================================================================
# ==============================================================================
# TAB 1: ASSIGNMENT 6 — TF-IDF TEXT CLASSIFICATION
# ==============================================================================
# ==============================================================================
with tab_asg6:
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0 0 6px 0; color: #f8fafc; font-size: 26px;">
            ⚡ Assignment 6: TF-IDF Text Classification & Step-by-Step Verification
        </h1>
        <p style="margin: 0; color: #94a3b8; font-size: 14px;">
            Module 1: TfidfVectorizer Representation, Exact Mathematical Proof (Difference: 0.000000), 
            Multi-Model Benchmark (Logistic Regression, Naive Bayes, Linear SVM), and Real-Time Router.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric Row
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    best_acc = max([res["accuracy"] for res in benchmark_results_asg6.values()]) * 100
    with col_m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(df_asg6)}</div><div class="metric-lbl">Total Samples</div></div>""", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(pipeline_asg6.classes)}</div><div class="metric-lbl">Categories</div></div>""", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(pipeline_asg6.feature_names)}</div><div class="metric-lbl">Vocabulary Size</div></div>""", unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{int(len(df_asg6)*(1-asg6_test_size))}/{int(len(df_asg6)*asg6_test_size)}</div><div class="metric-lbl">Train / Test Split</div></div>""", unsafe_allow_html=True)
    with col_m5:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #4ade80;">{best_acc:.1f}%</div><div class="metric-lbl">Top Accuracy</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander(f"📋 Explore Current Dataset: {dataset_choice} ({len(df_asg6)} Queries)", expanded=False):
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            sel_cat = st.selectbox("Filter by Category:", options=["All"] + pipeline_asg6.classes, key="asg6_cat_sel")
        with f_col2:
            search_query = st.text_input("Search in queries:", placeholder="e.g. pin, fee, statement, refund...", key="asg6_srch")
        
        filtered_df = df_asg6.copy()
        if sel_cat != "All":
            filtered_df = filtered_df[filtered_df["category"] == sel_cat]
        if search_query:
            filtered_df = filtered_df[filtered_df["text"].str.contains(search_query, case=False)]
        
        st.dataframe(filtered_df.reset_index(drop=True), width=1000, height=220)

    st.markdown("---")

    # ASG 6 - SECTION 1: VOCABULARY & SPARSE DTM
    st.subheader("1️⃣ Vocabulary & Sparse Document-Term TF-IDF Matrix")
    st.caption("Inspection of learned tokens, document frequencies, and the sparse TF-IDF feature space.")

    sub_t1, sub_t2 = st.tabs(["🔤 Learned Vocabulary Dictionary", "📑 Document-Term Matrix (DTM)"])

    with sub_t1:
        v_col1, v_col2 = st.columns([2, 1])
        vocab_df = pipeline_asg6.get_vocabulary_df()
        with v_col1:
            v_search = st.text_input("Filter Vocabulary Terms:", placeholder="Type a term to search (e.g., atm, loan, balance, card)...", key="asg6_vocab_srch")
            if v_search:
                display_vocab_df = vocab_df[vocab_df["Term"].str.contains(v_search.lower())]
            else:
                display_vocab_df = vocab_df
            st.dataframe(display_vocab_df, width=700, height=320)
        
        with v_col2:
            st.markdown(r"""
            **💡 Vocabulary Insights:**
            - **Tokenization**: Handled by scikit-learn analyzer extracting alphanumeric n-grams of word length $\ge 2$.
            - **Stop Words**: Standard English stop words (e.g. *the, is, at, which, on*) filtered out.
            - **IDF Weight**: Words appearing in fewer documents get higher IDF weights, serving as discriminative signals.
            """)
            st.info(f"**Vocabulary Size:** {len(vocab_df)} unique unigrams discovered across {len(df_asg6)} documents.")

    with sub_t2:
        dtm_df, dtm_stats = pipeline_asg6.get_doc_term_matrix_df(sample_docs=df_asg6["text"].tolist()[:14], max_cols=18)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Sample Rows (Docs)", dtm_stats.get("Total Documents", 0))
        col_s2.metric("Matrix Features (Words)", dtm_stats.get("Total Vocabulary Features", 0))
        col_s3.metric("Non-Zero Weights", dtm_stats.get("Non-zero Entries", 0))
        col_s4.metric("Matrix Sparsity", f"{dtm_stats.get('Sparsity Percentage', 0)}%")

        st.caption("Displaying top 18 active terms across first 14 corpus documents (styled by TF-IDF weight intensity):")
        st.dataframe(
            dtm_df.style.background_gradient(cmap="Blues", low=0.0, high=0.8).format("{:.4f}"),
            width=1000,
            height=340
        )

    st.markdown("---")

    # ASG 6 - SECTION 2: MANUAL STEP-BY-STEP MATHEMATICAL VERIFICATION
    st.subheader("2️⃣ Manual Step-by-Step Mathematical Verification")
    st.caption("Exact derivation matching Scikit-Learn's smoothed IDF and L2 Euclidean normalization down to 6 decimal places.")

    st.markdown(r"""
    <div class="verif-box">
        <h4 style="margin-top: 0; color: #38bdf8;">📐 Scikit-Learn TfidfVectorizer Mathematical Formulation:</h4>
        <div style="font-size: 15px; line-height: 1.8;">
            <b>1. Term Frequency:</b> $\text{TF}(t, d) = \text{Raw frequency count of term } t \text{ in document } d$<br>
            <b>2. Smoothed IDF:</b> $\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1.0 \quad \text{(where } N \text{ is total documents)}$<br>
            <b>3. Raw TF-IDF Weight:</b> $v(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$<br>
            <b>4. L2 Euclidean Normalization:</b> $\text{TF-IDF}_{\text{norm}}(t, d) = \frac{v(t, d)}{\sqrt{\sum_{w \in d} v(w, d)^2}}$
        </div>
    </div>
    """, unsafe_allow_html=True)

    math_col1, math_col2 = st.columns([3, 2])
    with math_col1:
        doc_options = df_asg6["text"].tolist()
        selected_doc = st.selectbox(
            "Select Document for Verification:",
            options=doc_options,
            index=0,
            format_func=lambda x: f"[{df_asg6.loc[df_asg6['text']==x, 'category'].values[0]}] {x}",
            key="asg6_math_doc"
        )
    
    with math_col2:
        analyzer_asg6 = pipeline_asg6.vectorizer.build_analyzer()
        doc_valid_tokens = [tok for tok in analyzer_asg6(selected_doc) if tok in pipeline_asg6.vocab]
        unique_doc_tokens = sorted(list(set(doc_valid_tokens)))
        
        if unique_doc_tokens:
            selected_word = st.selectbox("Select Target Word from Document:", options=unique_doc_tokens, index=0, key="asg6_math_w")
        else:
            selected_word = st.text_input("Enter Target Word:", value="atm", key="asg6_math_w_in")

    verif = pipeline_asg6.verify_math_step_by_step(selected_doc, selected_word)

    st.markdown("<br>", unsafe_allow_html=True)
    step_c1, step_c2 = st.columns(2)
    
    with step_c1:
        st.markdown(f"""
        <div class="step-card">
            <h4 style="margin: 0 0 6px 0; color: #818cf8;">Step 1: Document Tokenization & Term Frequency (TF)</h4>
            <p><b>Selected Document:</b> "<em>{verif['document']}</em>"</p>
            <p><b>Extracted Valid Tokens:</b> <code>{verif['valid_tokens']}</code></p>
            <p><b>Target Term ($t$):</b> <strong style="color: #38bdf8;">"{verif['target_term']}"</strong></p>
            <p><b>Raw Count:</b> $\\text{{TF}}("{verif['target_term']}", d) = {verif['tf_raw']}$</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="step-card">
            <h4 style="margin: 0 0 6px 0; color: #818cf8;">Step 2: Document Frequency (DF) & Smoothed IDF</h4>
            <p><b>Total Corpus Documents ($N$):</b> {verif['N']}</p>
            <p><b>Documents containing "{verif['target_term']}" ($\text{{DF}}(t)$):</b> {verif['df_count']}</p>
            <p><b>Smoothed IDF Substitution:</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(rf"\text{{IDF}}('{verif['target_term']}') = \ln\left(\frac{{1 + {verif['N']}}}{{1 + {verif['df_count']}}}\right) + 1.0 = \ln\left({round((1 + verif['N']) / (1 + verif['df_count']), 4)}\right) + 1.0 = {verif['idf_smooth']:.6f}")

    with step_c2:
        st.markdown(f"""
        <div class="step-card">
            <h4 style="margin: 0 0 6px 0; color: #818cf8;">Step 3: Document L2 Normalization Vector</h4>
            <p>Sum of squared weights across all unique valid tokens in document $d$:</p>
        </div>
        """, unsafe_allow_html=True)

        if verif["token_breakdowns"]:
            df_tokens = pd.DataFrame(verif["token_breakdowns"]).rename(columns={
                "token": "Token (w)",
                "tf": "TF(w, d)",
                "df": "DF(w)",
                "idf": "IDF(w)",
                "raw_tfidf": "TF x IDF",
                "squared": "(TF x IDF)²"
            })
            st.dataframe(df_tokens, width=500, height=160)
            st.latex(rf"\sum_{{w \in d}} v(w, d)^2 = {verif['sum_squares']:.6f}")
            st.latex(rf"\|v\|_2 = \sqrt{{\sum v(w, d)^2}} = \sqrt{{{verif['sum_squares']:.6f}}} = {verif['l2_norm']:.6f}")

    st.markdown("### Step 4: Final L2 Normalized Score & Scikit-Learn Parity Check")
    st.latex(rf"\text{{TF-IDF}}_{{\text{{norm}}}}('{verif['target_term']}', d) = \frac{{\text{{TF}} \times \text{{IDF}}}}{{\|v\|_2}} = \frac{{{verif['tf_raw']} \times {verif['idf_smooth']:.6f}}}{{{verif['l2_norm']:.6f}}} = \mathbf{{{verif['manual_tfidf']:.6f}}}")

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Hand-Calculated TF-IDF", f"{verif['manual_tfidf']:.6f}")
    res_col2.metric("Scikit-Learn Output", f"{verif['sklearn_tfidf']:.6f}")
    res_col3.metric("Absolute Difference", f"{verif['difference']:.8f}")

    if verif["is_verified"]:
        st.markdown(f"""
        <div style="text-align: center; margin-top: 10px;">
            <div class="match-badge">
                ✅ EXACT MATHEMATICAL MATCH VERIFIED (Difference: {verif['difference']:.8f})
            </div>
            <p style="color: #94a3b8; font-size: 13px; margin-top: 6px;">
                Manual math rigorously matches Scikit-Learn's TfidfVectorizer.transform() output.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"Note: Term '{verif['target_term']}' is not in document or filtered as stop word.")

    st.markdown("---")

    # ASG 6 - SECTION 3: CLASSIFIER BENCHMARK
    st.subheader("3️⃣ Multi-Model Benchmark & Confusion Matrix")
    st.caption("Comparison across Logistic Regression, Multinomial Naive Bayes, and Linear SVM on the stratified holdout test set.")

    b_col1, b_col2, b_col3 = st.columns(3)
    models_list = list(benchmark_results_asg6.keys())
    
    for i, col in enumerate([b_col1, b_col2, b_col3]):
        m_name = models_list[i]
        m_acc = benchmark_results_asg6[m_name]["accuracy"] * 100
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {'#38bdf8' if i==0 else '#a855f7' if i==1 else '#10b981'};">
                <div style="font-size: 15px; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">{m_name}</div>
                <div class="metric-val">{m_acc:.1f}%</div>
                <div class="metric-lbl">Test Set Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown("##### 📊 Accuracy Comparison Chart")
        acc_data = pd.DataFrame({
            "Model": list(benchmark_results_asg6.keys()),
            "Accuracy": [benchmark_results_asg6[m]["accuracy"] * 100 for m in benchmark_results_asg6.keys()]
        })
        fig_acc = px.bar(
            acc_data,
            x="Model",
            y="Accuracy",
            color="Model",
            text="Accuracy",
            color_discrete_sequence=["#38bdf8", "#a855f7", "#10b981"],
            range_y=[0, 110]
        )
        fig_acc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_acc.update_layout(
            template="plotly_dark",
            showlegend=False,
            height=330,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="",
            yaxis_title="Accuracy (%)"
        )
        st.plotly_chart(fig_acc, width=500)

    with chart_col2:
        selected_model_inspect = st.selectbox("Inspect Confusion Matrix for:", options=models_list, index=2, key="asg6_cm_model")
        cm = benchmark_results_asg6[selected_model_inspect]["confusion_matrix"]
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=pipeline_asg6.classes,
            y=pipeline_asg6.classes,
            text_auto=True,
            color_continuous_scale="Viridis",
            title=f"Confusion Matrix: {selected_model_inspect}"
        )
        fig_cm.update_layout(
            template="plotly_dark",
            height=330,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cm, width=500)

    st.markdown(f"##### 📑 Detailed Classification Report ({selected_model_inspect})")
    rep = benchmark_results_asg6[selected_model_inspect]["report"]
    rep_rows = []
    for cls in pipeline_asg6.classes:
        if cls in rep:
            rep_rows.append({
                "Category": cls,
                "Precision": round(rep[cls]["precision"], 3),
                "Recall": round(rep[cls]["recall"], 3),
                "F1-Score": round(rep[cls]["f1-score"], 3),
                "Support (Test Count)": int(rep[cls]["support"])
            })
    st.dataframe(pd.DataFrame(rep_rows), width=1000, height=190)

    st.markdown("---")

    # ASG 6 - SECTION 4: REAL-TIME QUERY CLASSIFIER
    st.subheader("4️⃣ Real-Time Query Classifier & Confidence Diagnostic")
    st.caption("Input any live customer/student query to route it to the appropriate department with confidence score bars.")

    st.markdown("**⚡ Quick Test Prompts (Click to Load):**")
    if domain_key == "banking":
        q_samples = [
            "ATM machine swallowed my card and did not dispense cash",
            "How do I check my account balance and request mini statement?",
            "Please block my lost credit card to prevent unauthorized fraud",
            "What is the home loan interest rate and monthly EMI?",
            "UPI payment failed on Google Pay but money debited from account"
        ]
    else:
        q_samples = [
            "When will end semester examination timetable and hall ticket be released?",
            "How to register for campus placement drives and verify resume?",
            "Tuition fee payment debited but receipt not generated on portal",
            "Hostel room fan and geyser water heater not working repair request"
        ]

    chip_cols = st.columns(len(q_samples))
    for idx, sample in enumerate(q_samples):
        if chip_cols[idx].button(f"Prompt {idx+1}", key=f"asg6_chip_{idx}", help=sample):
            st.session_state["query_input"] = sample

    default_text = st.session_state.get("query_input", q_samples[0])
    user_query = st.text_area(
        "Enter Query Text:",
        value=default_text,
        height=85,
        placeholder="Type customer inquiry here...",
        key="asg6_text_box"
    )

    pred_col1, pred_col2 = st.columns([1, 2])
    with pred_col1:
        model_choice_pred = st.selectbox(
            "Classifier for Inference:",
            options=["Linear SVM", "Logistic Regression", "Multinomial Naive Bayes"],
            index=0,
            key="asg6_pred_m"
        )
        st.button("🚀 Classify Query", type="primary", key="asg6_pred_btn")

    if user_query:
        pred_res = pipeline_asg6.predict_query(user_query, model_name=model_choice_pred)
        
        with pred_col2:
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px 20px;">
                <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Predicted Department</div>
                <div style="font-size: 28px; font-weight: 700; color: #38bdf8; margin: 2px 0;">
                    🏷️ {pred_res['predicted_class']}
                </div>
                <div style="font-size: 14px; color: #cbd5e1;">
                    Confidence: <strong style="color: #4ade80;">{pred_res['confidence_pct']}%</strong> | Model: <em>{pred_res['model_used']}</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        diag_col1, diag_col2 = st.columns([3, 2])

        with diag_col1:
            st.markdown("##### 📊 Class Confidence Distribution")
            conf_df = pd.DataFrame([
                {"Category": k, "Confidence (%)": round(v * 100, 2)}
                for k, v in pred_res["confidences"].items()
            ]).sort_values(by="Confidence (%)", ascending=True)

            fig_bar = px.bar(
                conf_df,
                x="Confidence (%)",
                y="Category",
                orientation="h",
                text="Confidence (%)",
                color="Confidence (%)",
                color_continuous_scale="Tealgrn"
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(
                template="plotly_dark",
                height=250,
                margin=dict(l=20, r=40, t=10, b=10),
                xaxis=dict(range=[0, 115])
            )
            st.plotly_chart(fig_bar, width=550)

        with diag_col2:
            st.markdown("##### 🔍 Contributing TF-IDF Keywords")
            if pred_res["contributing_terms"]:
                st.caption("Active tokens in vocabulary that influenced the classification decision:")
                df_contrib = pd.DataFrame(pred_res["contributing_terms"]).rename(columns={
                    "term": "Matched Token",
                    "tfidf": "TF-IDF Score",
                    "idf": "Learned IDF"
                })
                st.dataframe(df_contrib, width=400, height=200)
            else:
                st.info("No learned vocabulary words found in query. Prior baseline applied.")


# ==============================================================================
# ==============================================================================
# TAB 2: ASSIGNMENT 7 — N-GRAM LANGUAGE MODELS, SMOOTHING & PERPLEXITY
# ==============================================================================
# ==============================================================================
with tab_asg7:
    st.markdown(f"""
    <div class="main-header" style="border-left: 5px solid #a855f7;">
        <h1 style="margin: 0 0 6px 0; color: #f8fafc; font-size: 26px;">
            📖 Assignment 7: N-Gram Language Models, Smoothing & Perplexity
        </h1>
        <p style="margin: 0; color: #cbd5e1; font-size: 14px;">
            Module 2: Complete Scratch Implementation (<code>collections.Counter</code>, <code>math</code>, <code>random</code>, <code>re</code> — Zero NLP Libraries).
            Follows the exact 8-Option Menu Structure from the NLP Lab Manual.
        </p>
    </div>
    """, unsafe_allow_html=True)

    summary_asg7 = ngram_model.get_corpus_summary()
    c_col1, c_col2, c_col3, c_col4, c_col5 = st.columns(5)
    with c_col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{summary_asg7['sentence_count']}</div><div class="metric-lbl">Total Sentences</div></div>""", unsafe_allow_html=True)
    with c_col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{summary_asg7['total_word_tokens']}</div><div class="metric-lbl">Word Tokens (N)</div></div>""", unsafe_allow_html=True)
    with c_col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #a855f7;">{summary_asg7['vocabulary_size']}</div><div class="metric-lbl">Vocabulary (V)</div></div>""", unsafe_allow_html=True)
    with c_col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{summary_asg7['unique_bigrams']}</div><div class="metric-lbl">Unique Bigrams</div></div>""", unsafe_allow_html=True)
    with c_col5:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{summary_asg7['unique_trigrams']}</div><div class="metric-lbl">Unique Trigrams</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander(f"📚 Inspect Active Corpus: {corpus_choice} ({len(active_corpus_list)} Sentences)", expanded=False):
        st.write("**First 10 Corpus Sentences:**")
        for i, s in enumerate(active_corpus_list[:10], 1):
            st.markdown(f"`[{i}]` {s}")
        if "Custom" in corpus_choice:
            st.markdown("---")
            new_custom = st.text_area("Edit Custom Corpus (one sentence per line):", value=custom_corpus_text, height=120)
            if st.button("Update Custom Corpus", type="secondary"):
                st.session_state["custom_corpus_text"] = new_custom
                st.rerun()

    st.markdown("---")

    st.subheader("📑 Lab Manual 8-Option Navigation Menu")
    st.caption("Select any menu option below to explore N-gram generations, probability mathematics, and evaluation.")

    menu_tabs = st.tabs([
        "1. 📜 Corpus Overview",
        "2. 📊 N-Gram Tables",
        "3. 🎲 MLE Probabilities",
        "4. 🧮 Laplace Smoothing",
        "5. 📜 Sentence Probability",
        "6. 📉 Perplexity PP(W)",
        "7. 🔮 Autocomplete Predictor",
        "8. ✍️ Text Generator"
    ])

    # OPTION 1: CORPUS OVERVIEW
    with menu_tabs[0]:
        st.markdown("#### Option 1: Corpus Preprocessing & Vocabulary Architecture")
        st.markdown("""
        In pure-scratch N-gram language modeling:
        - **Preprocessing**: Lowercasing and extracting alphanumeric word tokens using pure regex: `re.findall(r'\\b[a-z0-9]+\\b', text.lower())`.
        - **Sentence Boundary Padding**:
          - For Bigrams: `[<s>, w1, w2, ..., wn, </s>]`
          - For Trigrams: `[<s>, <s>, w1, w2, ..., wn, </s>]`
        - **Vocabulary ($V$)**: Includes all unique words discovered in the corpus plus the sentence stop token `</s>` ($V = |\\text{word types}| + 1$).
        """)
        
        o1_c1, o1_c2 = st.columns(2)
        with o1_c1:
            st.info(f"""
            - **Sentences in Training Set**: {summary_asg7['sentence_count']}
            - **Total Word Tokens ($N$)**: {summary_asg7['total_word_tokens']}
            - **Vocabulary Size ($V$)**: {summary_asg7['vocabulary_size']}
            - **Start Token (`<s>`)**: Sentence boundary conditioning marker
            - **End Token (`</s>`)**: Valid target token for sentence termination
            """)
        with o1_c2:
            st.markdown("**Sample Preprocessed Sentence Boundary Formats:**")
            sample_preview = active_corpus_list[0]
            tokens_preview = NGramLanguageModel.tokenize(sample_preview)
            st.code(f"Raw: \"{sample_preview}\"\nUnigrams: {tokens_preview}\nBigram Padded: {['<s>'] + tokens_preview + ['</s>']}\nTrigram Padded: {['<s>', '<s>'] + tokens_preview + ['</s>']}")

    # OPTION 2: N-GRAM FREQUENCY TABLES
    with menu_tabs[1]:
        st.markdown("#### Option 2: Unigram, Bigram, and Trigram Frequency Tables")
        st.caption("Extracted and counted purely using Python `collections.Counter` without any external NLP libraries.")

        o2_col1, o2_col2 = st.columns([1, 2])
        with o2_col1:
            n_select = st.radio("Select N-Gram Order:", options=[1, 2, 3], format_func=lambda x: f"n={x} ({'Unigrams' if x==1 else 'Bigrams' if x==2 else 'Trigrams'})", horizontal=True)
            top_k_select = st.slider("Display Top K N-Grams:", min_value=5, max_value=40, value=15, step=5)

        table_data = ngram_model.get_ngram_frequency_table(n=n_select, top_k=top_k_select)
        df_table = pd.DataFrame(table_data)

        with o2_col2:
            if not df_table.empty:
                col_name = "Unigram" if n_select == 1 else "Bigram" if n_select == 2 else "Trigram"
                fig_ngram = px.bar(
                    df_table.head(10),
                    x=col_name,
                    y="Count",
                    color="Count",
                    color_continuous_scale="Purples",
                    title=f"Top 10 Most Frequent {col_name}s"
                )
                fig_ngram.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_ngram, width=550)

        st.dataframe(df_table, width=1000, height=280)

    # OPTION 3: MLE PROBABILITIES
    with menu_tabs[2]:
        st.markdown("#### Option 3: Maximum Likelihood Estimation (MLE) Probabilities (Unsmoothed)")
        st.markdown(r"""
        Maximum Likelihood Estimation assigns probability based purely on observed relative counts:
        - **Unigram**: $P_{\text{MLE}}(w_i) = \frac{C(w_i)}{N}$
        - **Bigram**: $P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$
        - **Trigram**: $P_{\text{MLE}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}$
        """)

        o3_col1, o3_col2 = st.columns(2)
        with o3_col1:
            mle_tgt = st.text_input("Target Word ($w_i$):", value="language", key="mle_tgt")
            mle_ctx = st.text_input("Context Word(s) separated by space (leave empty for Unigram):", value="natural", key="mle_ctx")
            ctx_tuple = tuple(mle_ctx.strip().split()) if mle_ctx.strip() else ()

        mle_res = ngram_model.calculate_mle_probability(mle_tgt, ctx_tuple)
        
        with o3_col2:
            st.markdown(f"""
            <div class="step-card">
                <h4 style="margin: 0 0 6px 0; color: #38bdf8;">MLE Probability Result ({mle_res.get('model', 'Model')})</h4>
                <p><b>Target Word:</b> <code>{mle_res.get('target_word', '')}</code></p>
                <p><b>Context:</b> <code>{mle_res.get('context', '')}</code></p>
                <p><b>Observed Joint Count $C$:</b> {mle_res.get('count', 0)}</p>
                <p><b>Context Count $C_{{\\text{{context}}}}$:</b> {mle_res.get('denominator', 0)}</p>
                <div style="font-size: 22px; font-weight: 700; color: #4ade80;">P = {mle_res.get('probability', 0.0):.6f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if "formula" in mle_res:
            st.latex(mle_res["formula"])

    # OPTION 4: LAPLACE SMOOTHING
    with menu_tabs[3]:
        st.markdown("#### Option 4: Laplace (Add-1) Smoothing Calculator")
        st.markdown(r"""
        Laplace Add-1 smoothing prevents zero probabilities for unseen N-grams by adding pseudo-counts:
        $$\text{Numerator} = \text{Count} + 1, \qquad \text{Denominator} = \text{Context\_Count} + V$$
        $$P_{\text{Laplace}}(w_i \mid \text{context}) = \frac{C(\text{context}, w_i) + 1}{C(\text{context}) + V}$$
        """)

        o4_col1, o4_col2 = st.columns([1, 1])
        with o4_col1:
            lap_tgt = st.text_input("Target Word ($w_i$):", value="processing", key="lap_tgt")
            lap_ctx = st.text_input("Context Word(s) (e.g. 'natural language' for trigram, 'language' for bigram):", value="natural language", key="lap_ctx")
            lap_ctx_tuple = tuple(lap_ctx.strip().split()) if lap_ctx.strip() else ()

        lap_res = ngram_model.calculate_laplace_probability(lap_tgt, lap_ctx_tuple)
        mle_comp = ngram_model.calculate_mle_probability(lap_tgt, lap_ctx_tuple)

        with o4_col2:
            st.markdown(f"""
            <div class="verif-box" style="margin: 0;">
                <h4 style="margin: 0 0 6px 0; color: #a855f7;">Exact Numerator & Denominator Breakdown:</h4>
                <p><b>Model:</b> {lap_res.get('model', '')}</p>
                <p><b>Observed Count $C$:</b> {lap_res.get('count', 0)}</p>
                <p><b>Context Count $C_{{\\text{{context}}}}$:</b> {lap_res.get('context_count', 0)}</p>
                <p><b>Vocabulary Size ($V$):</b> {lap_res.get('vocab_size', 0)}</p>
                <p><b>Exact Numerator ($C + 1$):</b> <strong style="color: #38bdf8;">{lap_res.get('numerator', 0)}</strong></p>
                <p><b>Exact Denominator ($C_{{\\text{{context}}}} + V$):</b> <strong style="color: #f43f5e;">{lap_res.get('denominator', 0)}</strong></p>
                <div style="font-size: 20px; font-weight: 700; color: #4ade80;">P_Laplace = {lap_res.get('probability', 0.0):.6f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.latex(lap_res.get("formula_latex", ""))

        c_comp1, c_comp2 = st.columns(2)
        with c_comp1:
            st.metric("MLE Probability (Unsmoothed)", f"{mle_comp.get('probability', 0.0):.6f}", help="Zero if n-gram is unseen!")
        with c_comp2:
            st.metric("Laplace Probability (Smoothed)", f"{lap_res.get('probability', 0.0):.6f}", help="Always strictly positive!")

        if mle_comp.get("probability", 0.0) == 0.0:
            st.warning("⚠️ Notice: The MLE probability is 0.0 (unseen n-gram). Laplace smoothing assigned a non-zero probability, avoiding the zero-frequency failure!")

    # OPTION 5: SENTENCE PROBABILITY
    with menu_tabs[4]:
        st.markdown("#### Option 5: Sentence Probability Calculator (Chain Rule)")
        st.caption("Computes total sequence probability across Unigram, Bigram, and Trigram models with side-by-side Smoothing comparison.")

        st.markdown("**⚡ Quick Test Sentences (Click to load):**")
        sample_sents = [
            "natural language processing is an exciting field",
            "language models assign probabilities to sequences of words",
            "atm machine did not dispense cash",
            "i do not like green eggs and ham"
        ]
        s_btn_cols = st.columns(len(sample_sents))
        for idx, s_txt in enumerate(sample_sents):
            if s_btn_cols[idx].button(f"Sentence {idx+1}", key=f"asg7_s_btn_{idx}", help=s_txt):
                st.session_state["sent_eval_input"] = s_txt

        eval_sent = st.text_input(
            "Enter Sentence to Evaluate:",
            value=st.session_state.get("sent_eval_input", sample_sents[0]),
            key="asg7_sent_in"
        )

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            eval_model_type = st.selectbox("Select N-Gram Model:", options=["Bigram", "Trigram", "Unigram"], index=0, key="asg7_eval_m")
        with p_col2:
            eval_smooth_toggle = st.checkbox("Enable Laplace (Add-1) Smoothing", value=True, key="asg7_smooth_chk")

        sent_res = ngram_model.calculate_sentence_probability(eval_sent, model_type=eval_model_type, smoothed=eval_smooth_toggle)

        if "error" in sent_res:
            st.error(sent_res["error"])
        else:
            k_col1, k_col2, k_col3 = st.columns(3)
            with k_col1:
                st.metric("Total Transition Steps", sent_res["total_transitions"])
            with k_col2:
                prob_fmt = f"{sent_res['sentence_prob']:.6e}" if sent_res['sentence_prob'] > 0 else "0.0"
                st.metric(f"Total Probability P(W)", prob_fmt)
            with k_col3:
                log_fmt = f"{sent_res['log_prob_sum']:.4f}" if sent_res['log_prob_sum'] != -float("inf") else "-Infinity"
                st.metric("Log Likelihood ln P(W)", log_fmt)

            if sent_res["has_zero_prob"]:
                st.error("🚨 Sentence probability evaluated to 0.0 because of unseen transition(s) in unsmoothed model! Enable Laplace Smoothing to fix this.")

            st.markdown("##### 🔍 Step-by-Step Chain Rule Multiplication Table:")
            df_steps = pd.DataFrame(sent_res["steps"])
            st.dataframe(df_steps, width=1000, height=240)

    # OPTION 6: PERPLEXITY
    with menu_tabs[5]:
        st.markdown("#### Option 6: Perplexity Evaluation PP(W)")
        st.markdown(r"""
        Perplexity measures the branching factor and predictive uncertainty of the language model:
        $$PP(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \ln P(w_i \mid \text{context})\right)$$
        *Lower perplexity indicates higher confidence and superior predictive accuracy.*
        """)

        pp_sent = st.text_input("Test Sentence for Perplexity Evaluation:", value=st.session_state.get("sent_eval_input", "natural language processing is exciting"), key="asg7_pp_sent")
        
        pp_c1, pp_c2, pp_c3 = st.columns(3)
        pp_uni = ngram_model.calculate_perplexity(pp_sent, model_type="unigram", smoothed=True)
        pp_bi = ngram_model.calculate_perplexity(pp_sent, model_type="bigram", smoothed=True)
        pp_tri = ngram_model.calculate_perplexity(pp_sent, model_type="trigram", smoothed=True)

        with pp_c1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #38bdf8;">
                <div style="font-size: 15px; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">Unigram PP(W)</div>
                <div class="metric-val">{pp_uni.get('perplexity', 0.0):.2f}</div>
                <div class="metric-lbl">Cross-Entropy: {pp_uni.get('cross_entropy', 0.0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with pp_c2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #a855f7;">
                <div style="font-size: 15px; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">Bigram PP(W)</div>
                <div class="metric-val">{pp_bi.get('perplexity', 0.0):.2f}</div>
                <div class="metric-lbl">Cross-Entropy: {pp_bi.get('cross_entropy', 0.0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with pp_c3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #10b981;">
                <div style="font-size: 15px; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">Trigram PP(W)</div>
                <div class="metric-val">{pp_tri.get('perplexity', 0.0):.2f}</div>
                <div class="metric-lbl">Cross-Entropy: {pp_tri.get('cross_entropy', 0.0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_pp = px.bar(
            x=["Unigram", "Bigram", "Trigram"],
            y=[pp_uni.get('perplexity', 0.0), pp_bi.get('perplexity', 0.0), pp_tri.get('perplexity', 0.0)],
            labels={"x": "Model Order", "y": "Perplexity (Lower is Better)"},
            color=["Unigram", "Bigram", "Trigram"],
            color_discrete_sequence=["#38bdf8", "#a855f7", "#10b981"],
            title="Model Perplexity Comparison on Test Sequence (Laplace Smoothed)"
        )
        fig_pp.update_layout(template="plotly_dark", height=280, showlegend=False, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_pp, width=650)

    # OPTION 7: AUTOCOMPLETE PREDICTOR
    with menu_tabs[6]:
        st.markdown("#### Option 7: Trigram Next-Word Autocomplete Predictor")
        st.caption("Conditioned on the preceding 2 words (or 1 word fallback), predicts and ranks the most probable next word.")

        ac_col1, ac_col2 = st.columns([1, 1])
        with ac_col1:
            ac_prompt = st.text_input("Enter 1 or 2 word context prompt:", value="natural language", key="asg7_ac_p")
            ac_topk = st.slider("Top Candidates to Predict:", min_value=3, max_value=10, value=5, key="asg7_ac_k")
            ac_use_laplace = st.checkbox("Use Laplace Smoothed Probabilities", value=True, key="asg7_ac_lap")

        ac_res = ngram_model.autocomplete_next_words(ac_prompt, top_k=ac_topk, use_laplace=ac_use_laplace)

        with ac_col2:
            st.info(f"**Context Mode:** {ac_res['context_label']}\n\n**Context Frequency in Corpus:** {ac_res['context_count']}")

        if ac_res["predictions"]:
            df_pred = pd.DataFrame(ac_res["predictions"])
            
            p_chart_col1, p_chart_col2 = st.columns([3, 2])
            with p_chart_col1:
                fig_ac = px.bar(
                    df_pred,
                    x="Probability",
                    y="Word",
                    orientation="h",
                    color="Probability",
                    text="Percentage (%)",
                    color_continuous_scale="Purples",
                    title=f"Next Word Candidates for: '{ac_prompt}'"
                )
                fig_ac.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_ac.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=40, t=35, b=10))
                st.plotly_chart(fig_ac, width=500)

            with p_chart_col2:
                st.markdown("**Top Recommended Completions:**")
                for item in ac_res["predictions"]:
                    st.markdown(f"- 💡 `'{ac_prompt} {item['Word']}'` (Prob: **{item['Probability']:.4f}**)")

    # OPTION 8: TEXT GENERATOR
    with menu_tabs[7]:
        st.markdown("#### Option 8: Weighted Random Sentence Generator")
        st.caption("Generates novel sentences by sampling words from conditioned N-gram distributions starting from `<s>` until `</s>`.")

        gen_c1, gen_c2, gen_c3 = st.columns(3)
        with gen_c1:
            gen_model_choice = st.selectbox("Generator Model:", options=["Trigram", "Bigram"], index=0, key="asg7_gen_m")
        with gen_c2:
            gen_max_len = st.slider("Maximum Token Length:", min_value=10, max_value=35, value=20, key="asg7_gen_len")
        with gen_c3:
            gen_seed = st.number_input("Random Seed (0 for random):", min_value=0, max_value=9999, value=42, key="asg7_gen_seed")

        if st.button("🎲 Generate New Sentence", type="primary", key="asg7_gen_btn"):
            st.session_state["gen_counter"] = st.session_state.get("gen_counter", 0) + 1

        actual_seed = None if gen_seed == 0 else gen_seed + st.session_state.get("gen_counter", 0)
        gen_result = ngram_model.generate_sentence(model_type=gen_model_choice, max_tokens=gen_max_len, seed=actual_seed)

        st.markdown(f"""
        <div class="gen-box">
            "{gen_result['generated_text']}"
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Generated via **{gen_result['model_type']} Model** • {gen_result['token_count']} word tokens produced.")

        with st.expander("🔍 View Step-by-Step Transition Trace", expanded=False):
            st.dataframe(pd.DataFrame(gen_result["trace_steps"]), width=800, height=220)


# ==============================================================================
# ==============================================================================
# TAB 3: ASSIGNMENT 8 — LATENT SEMANTIC INDEXING (LSI) & TRUNCATED SVD
# ==============================================================================
# ==============================================================================
with tab_asg8:
    st.markdown(f"""
    <div class="main-header" style="border-left: 5px solid #06b6d4;">
        <h1 style="margin: 0 0 6px 0; color: #f8fafc; font-size: 26px;">
            📑 Assignment 8: Latent Semantic Indexing (LSI) & Truncated SVD
        </h1>
        <p style="margin: 0; color: #cbd5e1; font-size: 14px;">
            Module 3: Dimensionality Reduction via Truncated Singular Value Decomposition ($A \\approx U \\Sigma V^T$), 
            Topic Term Loadings, Latent Cosine Similarity Heatmap, and Semantic Information Retrieval / Opinion Mining.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics for LSI
    df_A_mat, stats_A = lsi_engine.get_matrix_a_df()
    var_info = lsi_engine.get_explained_variance_data()
    m_docs = len(lsi_corpus_active)
    n_terms = len(lsi_engine.vocabulary)
    reduction_pct = round((1.0 - (lsi_engine.n_components / n_terms)) * 100, 1) if n_terms > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{m_docs}</div><div class="metric-lbl">Documents (m)</div></div>""", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{n_terms}</div><div class="metric-lbl">Vocabulary (n)</div></div>""", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #06b6d4;">k={lsi_engine.n_components}</div><div class="metric-lbl">Latent Concepts</div></div>""", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #4ade80;">{var_info.get('total_variance_explained', 0.0)}%</div><div class="metric-lbl">Explained Variance</div></div>""", unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #f59e0b;">{reduction_pct}%</div><div class="metric-lbl">Dimension Reduction</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Expandable Corpus Inspector
    with st.expander(f"📚 Inspect Active LSI Corpus: {lsi_corpus_choice} ({m_docs} Documents)", expanded=False):
        df_corp_preview = pd.DataFrame([
            {
                "ID": d["id"],
                "Title": d["title"],
                "Category": d.get("category", "N/A"),
                "Sentiment": d.get("sentiment", "N/A"),
                "Full Text": d["text"]
            }
            for d in lsi_corpus_active
        ])
        st.dataframe(df_corp_preview, width=1000, height=230)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # LSI SECTION 1: DOCUMENT-TERM TF-IDF MATRIX A
    # --------------------------------------------------------------------------
    st.subheader("1️⃣ Document-Term TF-IDF Matrix $A$ & Vocabulary")
    st.caption("Initial high-dimensional sparse vector space representation before SVD dimensionality reduction.")

    lsi_s1_col1, lsi_s1_col2, lsi_s1_col3, lsi_s1_col4 = st.columns(4)
    lsi_s1_col1.metric("Matrix Shape (m x n)", stats_A.get("Matrix Shape (m x n)", "N/A"))
    lsi_s1_col2.metric("Total Document Rows", stats_A.get("Total Documents (m)", 0))
    lsi_s1_col3.metric("Vocabulary Features", stats_A.get("Total Vocabulary Terms (n)", 0))
    lsi_s1_col4.metric("Matrix Sparsity", f"{stats_A.get('Sparsity (%)', 0)}%")

    st.caption("Displaying TF-IDF matrix values (documents as rows, vocabulary terms as columns):")
    st.dataframe(
        df_A_mat.style.background_gradient(cmap="Blues", low=0.0, high=0.8).format("{:.4f}"),
        width=1000,
        height=260
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # LSI SECTION 2: SVD DECOMPOSITION (U, Σ, V^T)
    # --------------------------------------------------------------------------
    st.subheader("2️⃣ Truncated SVD Matrix Factorization: $A \\approx U_k \\Sigma_k V_k^T$")
    st.caption("Low-rank matrix factorization decomposing the document-term space into latent semantic concepts.")

    st.markdown(r"""
    <div class="verif-box">
        <h4 style="margin-top: 0; color: #06b6d4;">📐 Mathematical Formulation of Latent Semantic Indexing:</h4>
        <div style="font-size: 15px; line-height: 1.8;">
            <b>1. Low-Rank Approximation:</b> $A_{m \times n} \approx U_{m \times k} \cdot \Sigma_{k \times k} \cdot V_{k \times n}^T$<br>
            <b>2. Document-Topic Matrix $U$:</b> Orthonormal columns representing document coordinates along $k$ latent semantic concepts.<br>
            <b>3. Singular Values $\Sigma$:</b> Diagonal matrix with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k > 0$, measuring the singular energy of each concept.<br>
            <b>4. Topic-Term Loading Matrix $V^T$:</b> Rows represent latent topics; columns represent term loadings (positive/negative associations).
        </div>
    </div>
    """, unsafe_allow_html=True)

    decomp_matrices = lsi_engine.get_decomposed_matrices()
    topic_profiles = lsi_engine.get_latent_topics_profile(top_n_keywords=6)

    svd_tab_U, svd_tab_Sigma, svd_tab_VT, svd_tab_profiles = st.tabs([
        "📄 Document-Topic Matrix U (m × k)",
        "💎 Singular Values Σ (k × k)",
        "🔤 Topic-Term Matrix V^T (k × n)",
        "🏷️ Latent Topic Concept Profiles"
    ])

    with svd_tab_U:
        u_col1, u_col2 = st.columns([1, 1])
        with u_col1:
            st.markdown(f"**Orthonormal Matrix $U$ ($m \\times k = {m_docs} \\times {lsi_engine.n_components}$):**")
            st.dataframe(
                decomp_matrices["U"].style.background_gradient(cmap="Purples", low=0.0, high=0.8).format("{:.4f}"),
                width=500,
                height=260
            )
        with u_col2:
            st.markdown("**2D Document Projection in Latent Space (Topics 1 & 2):**")
            df_scatter = pd.DataFrame({
                "Doc": [f"{d['id']}: {d['title']}" for d in lsi_corpus_active],
                "Category": [d.get("category", "N/A") for d in lsi_corpus_active],
                "Topic 1 Loading": lsi_engine.doc_lsi_embeddings[:, 0],
                "Topic 2 Loading": lsi_engine.doc_lsi_embeddings[:, 1]
            })
            fig_scatter = px.scatter(
                df_scatter,
                x="Topic 1 Loading",
                y="Topic 2 Loading",
                color="Category",
                text="Doc",
                title="Semantic Document Coordinates in Latent Space"
            )
            fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
            fig_scatter.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_scatter, width=500)

    with svd_tab_Sigma:
        st.markdown(f"**Diagonal Singular Values Matrix $\\Sigma$ ($k \\times k = {lsi_engine.n_components} \\times {lsi_engine.n_components}$):**")
        st.dataframe(
            decomp_matrices["Sigma"].style.background_gradient(cmap="Greens", low=0.0, high=0.8).format("{:.4f}"),
            width=500,
            height=180
        )
        sig_list = decomp_matrices["singular_values"]
        st.info(" • ".join([f"**Concept {i+1} Singular Value (σ_{i+1})**: `{round(val, 4)}`" for i, val in enumerate(sig_list)]))

    with svd_tab_VT:
        st.markdown(f"**Topic-Term Matrix $V^T$ ($k \\times n = {lsi_engine.n_components} \\times {n_terms}$):** (Showing top 25 salient terms)")
        st.dataframe(
            decomp_matrices["VT"].style.background_gradient(cmap="coolwarm", low=0.0, high=0.8).format("{:.4f}"),
            width=1000,
            height=200
        )

    with svd_tab_profiles:
        st.markdown("#### Latent Concept Profiles & Top Keyword Loadings:")
        prof_cols = st.columns(len(topic_profiles))
        for idx, prof in enumerate(topic_profiles):
            with prof_cols[idx]:
                st.markdown(f"""
                <div class="step-card" style="border-left-color: {'#38bdf8' if idx==0 else '#a855f7' if idx==1 else '#10b981' if idx==2 else '#f59e0b'};">
                    <h4 style="margin: 0 0 6px 0; color: #f8fafc;">{prof['topic_name']}</h4>
                    <p style="font-size: 13px; color: #94a3b8; margin-bottom: 8px;">
                        <b>σ = {prof['singular_value']}</b> | Variance: <b>{prof['variance_explained_pct']}%</b>
                    </p>
                    <p><b>Strongest Keywords:</b></p>
                    <p><code>{', '.join(prof['dominant_keywords'])}</code></p>
                    <p style="font-size: 12px; color: #cbd5e1;">
                        <b>Top Positive:</b> {', '.join([t[0] for t in prof['top_positive'][:3]])}<br>
                        <b>Top Negative:</b> {', '.join([t[0] for t in prof['top_negative'][:3]])}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # LSI SECTION 3: EXPLAINED VARIANCE RATIO
    # --------------------------------------------------------------------------
    st.subheader("3️⃣ Explained Variance Ratio")
    st.caption("Evaluates the amount of semantic information and variance retained after Truncated SVD dimensionality reduction.")

    ev_col1, ev_col2 = st.columns([2, 1])
    with ev_col1:
        df_var = pd.DataFrame({
            "Component": var_info["components"],
            "Individual Variance (%)": var_info["individual_variance"],
            "Cumulative Variance (%)": var_info["cumulative_variance"]
        })
        
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=df_var["Component"],
            y=df_var["Individual Variance (%)"],
            name="Individual Variance (%)",
            marker_color="#38bdf8",
            text=df_var["Individual Variance (%)"].apply(lambda x: f"{x:.1f}%"),
            textposition="auto"
        ))
        fig_var.add_trace(go.Scatter(
            x=df_var["Component"],
            y=df_var["Cumulative Variance (%)"],
            name="Cumulative Variance (%)",
            mode="lines+markers+text",
            marker=dict(color="#f43f5e", size=8),
            line=dict(width=3),
            text=df_var["Cumulative Variance (%)"].apply(lambda x: f"{x:.1f}%"),
            textposition="top center"
        ))
        fig_var.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(range=[0, 115]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_var, width=700)

    with ev_col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: left; padding: 20px;">
            <h4 style="margin: 0 0 8px 0; color: #4ade80;">💡 SVD Compression Insight:</h4>
            <p style="font-size: 14px; color: #cbd5e1; line-height: 1.6;">
                Reducing dimensionality from <b>{n_terms} vocabulary words</b> to <b>{lsi_engine.n_components} latent concepts</b> retains 
                <strong style="color: #38bdf8;">{var_info.get('total_variance_explained', 0.0)}%</strong> of the total corpus variance.
            </p>
            <p style="font-size: 13px; color: #94a3b8; margin: 0;">
                By discarding noisy minor singular values, LSI uncovers latent semantic relationships and eliminates synonymy / polysemy artifacts.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # LSI SECTION 4: COSINE SIMILARITY HEATMAP
    # --------------------------------------------------------------------------
    st.subheader("4️⃣ Interactive Document Cosine Similarity Heatmap")
    st.caption("Comparison of document-to-document similarity in reduced Latent Semantic Space vs. Raw TF-IDF Keyword Space.")

    sim_choice = st.radio(
        "Select Similarity Space:",
        options=["✨ LSI Latent Space Cosine Similarity (Semantic Concepts)", "📊 Raw TF-IDF Cosine Similarity (Exact Keywords)"],
        horizontal=True,
        key="asg8_sim_space"
    )

    doc_short_labels = [f"{d['id']}: {d['title'][:18]}" for d in lsi_corpus_active]
    sim_matrix = lsi_engine.lsi_cosine_sim_matrix if "LSI" in sim_choice else lsi_engine.tfidf_cosine_sim_matrix

    fig_heat = px.imshow(
        sim_matrix,
        x=doc_short_labels,
        y=doc_short_labels,
        labels=dict(x="Document", y="Document", color="Cosine Similarity"),
        text_auto=".2f",
        color_continuous_scale="Viridis" if "LSI" in sim_choice else "Blues",
        range_color=[0, 1] if "LSI" in sim_choice else [0, 1],
        title=f"Document-to-Document Cosine Similarity Heatmap ({'LSI Latent Space' if 'LSI' in sim_choice else 'Raw TF-IDF Space'})"
    )
    fig_heat.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_heat, width=1000)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # LSI SECTION 5: SEMANTIC QUERY SEARCH & OPINION MINING
    # --------------------------------------------------------------------------
    st.subheader("5️⃣ Semantic Query Search & Document Categorization")
    st.caption("Input any query or opinion. LSI projects it into latent space to find the semantically closest document even when exact keywords don't match.")

    # Contextual quick prompts depending on active corpus
    if "(a)" in lsi_corpus_choice:
        lsi_q_samples = [
            "neural network GPU computation",
            "statistical syntax speech analysis",
            "visual digital frame detection",
            "AI combining algorithms and text"
        ]
    elif "(b)" in lsi_corpus_choice:
        lsi_q_samples = [
            "ATM didn't give money and card trapped",
            "very smooth fast cash withdrawal kiosk",
            "mobile application password expired",
            "high loan interest penalty and hidden fees"
        ]
    else:
        lsi_q_samples = [
            "human computer interaction response time",
            "graph tree algorithms ordering",
            "system testing of EPS"
        ]

    st.markdown("**⚡ Quick Test Prompts (Click to Load):**")
    lsi_btn_cols = st.columns(len(lsi_q_samples))
    for idx, sample in enumerate(lsi_q_samples):
        if lsi_btn_cols[idx].button(f"Query {idx+1}", key=f"asg8_q_btn_{idx}", help=sample):
            st.session_state["lsi_query_in"] = sample

    default_lsi_q = st.session_state.get("lsi_query_in", lsi_q_samples[0])
    lsi_user_query = st.text_input(
        "Enter Search Query / Customer Opinion:",
        value=default_lsi_q,
        key="asg8_search_input"
    )

    if lsi_user_query:
        search_res = lsi_engine.query_semantic_search(lsi_user_query, top_k=min(5, m_docs))
        top_match = search_res["ranked_documents"][0]

        q_res_col1, q_res_col2 = st.columns([1, 2])
        with q_res_col1:
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px;">
                <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Top Semantic Match</div>
                <div style="font-size: 24px; font-weight: 700; color: #06b6d4; margin: 4px 0;">
                    🏆 {top_match['title']}
                </div>
                <div style="font-size: 14px; color: #cbd5e1; margin-top: 6px;">
                    <b>LSI Similarity:</b> <strong style="color: #4ade80;">{top_match['lsi_similarity']}</strong><br>
                    <b>Predicted Category:</b> <span style="color: #38bdf8;">{search_res['predicted_category']}</span><br>
                    <b>Dominant Latent Concept:</b> <em>{search_res['dominant_topic']}</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with q_res_col2:
            st.markdown(f"""
            <div class="verif-box" style="margin: 0; padding: 16px;">
                <h4 style="margin: 0 0 4px 0; color: #38bdf8;">🧠 Semantic Information Retrieval Diagnostic:</h4>
                <p><b>User Query:</b> "<em>{search_res['query']}</em>"</p>
                <p><b>Exact Keywords Matched in Vocabulary:</b> <code>{search_res['raw_matched_tokens'] if search_res['raw_matched_tokens'] else 'None (Zero Lexical Overlap!)'}</code></p>
                <p><b>LSI Latent Projection Vector ($q_{{\\text{{LSI}}}}$):</b> <code>{search_res['query_lsi_vector']}</code></p>
                <p style="font-size: 13px; color: #a5f3fc; margin: 0;">
                    {"✨ <b>Zero Lexical Overlap Solved:</b> Query shared few or no exact words, yet LSI correctly routed to the conceptually relevant document via shared latent topic space!" if not search_res['raw_matched_tokens'] or top_match['semantic_boost'] > 0 else "Query successfully categorized through latent semantic projection."}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📑 Ranked Document Retrieval & Semantic Boost Table:")
        df_ranked = pd.DataFrame(search_res["ranked_documents"])[[
            "rank", "doc_id", "title", "category", "lsi_similarity", "tfidf_similarity", "semantic_boost", "text"
        ]].rename(columns={
            "rank": "Rank",
            "doc_id": "Doc ID",
            "title": "Document Title",
            "category": "Category",
            "lsi_similarity": "LSI Similarity (Semantic)",
            "tfidf_similarity": "Raw TF-IDF Similarity (Keyword)",
            "semantic_boost": "Semantic Boost (LSI - TFIDF)",
            "text": "Document Excerpt"
        })
        st.dataframe(df_ranked, width=1000, height=240)
