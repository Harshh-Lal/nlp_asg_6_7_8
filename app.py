"""
app.py - IntelliDesk: Intelligent Query Classification & NLP Lab
Tab 1: Assignment 6 (TF-IDF Text Classification)
Tab 2: Assignment 7 (N-Gram Language Models, Smoothing & Perplexity)
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
# SIDEBAR CONTROLS (SEPARATED FOR ASG 6 & ASG 7)
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=55)
    st.title("⚡ IntelliDesk")
    st.caption("AI-Powered Support Query Routing & NLP Lab Suite")
    st.markdown("---")

    sidebar_module_focus = st.radio(
        "🎛️ Configure Settings For:",
        options=["📊 Assignment 6 (TF-IDF)", "📖 Assignment 7 (N-Gram Models)"],
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

    else:
        # Defaults for Asg 6 when not in focus
        domain_key = "banking"
        dataset_choice = "🏦 Banking Support"
        asg6_test_size = 0.25
        asg6_seed = 42

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

    st.markdown("---")
    st.caption("IntelliDesk NLP Lab • Academic Session 2026")

# Set default corpus choice if Asg 6 was focused
if "corpus_choice" not in locals():
    corpus_choice = "📚 Default Lab Manual NLP Corpus (15 sentences)"

# ==============================================================================
# PIPELINE CACHING (ASG 6 & ASG 7)
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
        # Custom user corpus
        lines = [line.strip() for line in custom_text.split("\n") if line.strip()]
        corpus = lines if lines else DEFAULT_LAB_CORPUS
    model = NGramLanguageModel(corpus)
    return model, corpus

custom_corpus_text = st.session_state.get("custom_corpus_text", "")
ngram_model, active_corpus_list = get_cached_ngram_model(corpus_choice, custom_corpus_text)

# ==============================================================================
# MAIN TABS (CLEAN SEPARATION OF ASSIGNMENT 6 & ASSIGNMENT 7)
# ==============================================================================
tab_asg6, tab_asg7 = st.tabs([
    "📊 Assignment 6: TF-IDF Text Classification",
    "📖 Assignment 7: N-Gram Language Models, Smoothing & Perplexity"
])


# ==============================================================================
# ==============================================================================
# TAB 1: ASSIGNMENT 6 — TF-IDF TEXT CLASSIFICATION
# ==============================================================================
# ==============================================================================
with tab_asg6:
    # Header Banner
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

    # Accordion to explore raw dataset
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

    # --------------------------------------------------------------------------
    # ASG 6 - SECTION 1: VOCABULARY & SPARSE DTM
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # ASG 6 - SECTION 2: MANUAL STEP-BY-STEP MATHEMATICAL VERIFICATION
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # ASG 6 - SECTION 3: CLASSIFIER BENCHMARK
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # ASG 6 - SECTION 4: REAL-TIME QUERY CLASSIFIER
    # --------------------------------------------------------------------------
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
    # Header Banner
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

    # Corpus Summary KPI Cards
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

    # Expandable Corpus Explorer & Custom Sentences
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

    # ==========================================================================
    # 8-OPTION LAB MANUAL MENU STRUCTURE
    # ==========================================================================
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

    # --------------------------------------------------------------------------
    # OPTION 1: CORPUS OVERVIEW & PREPROCESSING
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # OPTION 2: N-GRAM FREQUENCY TABLES
    # --------------------------------------------------------------------------
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
            # Bar chart of top n-grams
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

    # --------------------------------------------------------------------------
    # OPTION 3: MLE PROBABILITIES
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # OPTION 4: LAPLACE (ADD-1) SMOOTHING CALCULATOR
    # --------------------------------------------------------------------------
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

        # Side by side comparison: MLE vs Laplace
        c_comp1, c_comp2 = st.columns(2)
        with c_comp1:
            st.metric("MLE Probability (Unsmoothed)", f"{mle_comp.get('probability', 0.0):.6f}", help="Zero if n-gram is unseen!")
        with c_comp2:
            st.metric("Laplace Probability (Smoothed)", f"{lap_res.get('probability', 0.0):.6f}", help="Always strictly positive!")

        if mle_comp.get("probability", 0.0) == 0.0:
            st.warning("⚠️ Notice: The MLE probability is 0.0 (unseen n-gram). Laplace smoothing assigned a non-zero probability, avoiding the zero-frequency probability failure!")

    # --------------------------------------------------------------------------
    # OPTION 5: SENTENCE PROBABILITY CALCULATOR
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # OPTION 6: PERPLEXITY EVALUATION
    # --------------------------------------------------------------------------
    with menu_tabs[5]:
        st.markdown("#### Option 6: Perplexity Evaluation PP(W)")
        st.markdown(r"""
        Perplexity measures the branching factor and predictive uncertainty of the language model:
        $$PP(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \ln P(w_i \mid \text{context})\right)$$
        *Lower perplexity indicates higher confidence and superior predictive accuracy.*
        """)

        pp_sent = st.text_input("Test Sentence for Perplexity Evaluation:", value=st.session_state.get("sent_eval_input", "natural language processing is exciting"), key="asg7_pp_sent")
        
        pp_c1, pp_c2, pp_c3 = st.columns(3)
        # Evaluate across Unigram, Bigram, and Trigram
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
        # Plotly comparison
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

    # --------------------------------------------------------------------------
    # OPTION 7: TRIGRAM NEXT-WORD AUTOCOMPLETE PREDICTOR
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # OPTION 8: WEIGHTED RANDOM SENTENCE GENERATOR
    # --------------------------------------------------------------------------
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
