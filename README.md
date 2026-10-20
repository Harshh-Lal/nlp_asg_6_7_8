# IntelliDesk: AI-Powered Support Query Routing & NLP Lab
## NLP Laboratory Suite: Assignment 6, Assignment 7 & Assignment 8

---

### 📌 Overview
**IntelliDesk** is a modular enterprise NLP intelligence and evaluation platform containing 3 complete laboratory modules:
1. **Module 1 (Assignment 6)**: TF-IDF Text Classification & Department Routing.
2. **Module 2 (Assignment 7)**: N-Gram Language Models, Laplace (Add-1) Smoothing & Perplexity Evaluation.
3. **Module 3 (Assignment 8)**: Latent Semantic Indexing (LSI) & Truncated Singular Value Decomposition (SVD) for Document Categorization & Opinion Mining.

The system features three cleanly separated tabs and dedicated sidebar settings controls, allowing students and instructors to independently inspect, verify, benchmark, and interact with all three assignments.

---

### 📁 Architecture & File Structure
```
Asg-6_7/
├── app.py                  # Main Streamlit UI (Tab 1: Asg 6, Tab 2: Asg 7, Tab 3: Asg 8)
├── tfidf_classifier.py     # Module 1: TF-IDF Vectorizer, Math Proof, Multi-Model Benchmark
├── ngram_engine.py         # Module 2: Pure Scratch N-Gram Language Model & 8-Option Lab Menu
├── lsi_engine.py           # Module 3: Latent Semantic Indexing (LSI) & Truncated SVD Engine
└── README.md               # Complete Project & Mathematical Documentation
```

---

## 📊 Module 1: Assignment 6 (TF-IDF Text Classification)

### Features:
- **Datasets**:
  - `Banking Support`: ATM, Account, Card, Loan, UPI (65 expanded realistic inquiries).
  - `Student Query Helpdesk`: Exam, Placement, Fees, Hostel (48 realistic queries).
- **TF-IDF Feature Space**:
  - `TfidfVectorizer(lowercase=True, stop_words='english')`.
  - Searchable learned vocabulary with Feature Index, Document Frequency ($DF$), and $IDF$ scores.
  - Sparse Document-Term Matrix (DTM) with real-time sparsity percentage ($>95\%$ sparse).
- **Exact Step-by-Step Mathematical Verification**:
  $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1.0$$
  $$\text{TF-IDF}_{\text{norm}}(t, d) = \frac{\text{TF}(t, d) \times \text{IDF}(t)}{\sqrt{\sum_{w \in d} (\text{TF}(w, d) \times \text{IDF}(w))^2}}$$
  - **Difference with Scikit-Learn: `0.00000000`** (`✅ EXACT MATHEMATICAL MATCH VERIFIED`).
- **Multi-Model Benchmark**:
  - Logistic Regression, Multinomial Naive Bayes, Linear SVM.
  - Interactive Plotly Confusion Matrix and Classification Report.
- **Real-Time Department Router**:
  - Predicts category, provides confidence bars, and breaks down active TF-IDF keywords.

---

## 📖 Module 2: Assignment 7 (N-Gram Language Models & Smoothing)

*Implemented completely from scratch using Python standard libraries (`collections.Counter`, `math`, `random`, `re`) — zero external NLP dependencies.*

### Exact 8-Option Lab Manual Menu Structure:

| Option | Feature | Implementation Details |
| :---: | :--- | :--- |
| **1** | **Corpus Overview & Architecture** | Alphanumeric regex tokenization, boundary padding (`<s>`, `</s>`), token counts $N$, vocabulary size $V = \|\text{unique types}\| + 1$. |
| **2** | **N-Gram Frequency Tables** | Frequency tables and Plotly bar charts for Unigram ($n=1$), Bigram ($n=2$), and Trigram ($n=3$) models using `collections.Counter`. |
| **3** | **MLE Probability Calculator** | Unsmoothed conditional probabilities: $P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$. |
| **4** | **Laplace (Add-1) Smoothing Calculator** | Exact step-by-step display of **Numerator ($C + 1$)** and **Denominator ($C_{\text{context}} + V$)**; explains prevention of zero-frequency probability failures. |
| **5** | **Sentence Probability Calculator** | Chain Rule evaluation: $P(W) = \prod_{i=1}^M P(w_i \mid \text{context})$ with full tabular breakdown of each token, context, count, fraction, and log-likelihood $\sum \ln P$. |
| **6** | **Perplexity Evaluation** | Evaluates $PP(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \ln P(w_i \mid \text{context})\right)$; provides cross-entropy and comparative Unigram vs. Bigram vs. Trigram chart. |
| **7** | **Trigram Autocomplete Predictor** | Given a 1 or 2 word prompt, extracts candidate continuation tokens and ranks them by conditional probability with Plotly horizontal bars. |
| **8** | **Weighted Random Sentence Generator** | Samples tokens conditioned on preceding N-gram context starting from `<s>` until `</s>` using `random.choices`, returning generated text and transition trace. |

---

## 📑 Module 3: Assignment 8 (Latent Semantic Indexing & Truncated SVD)

*Implements Latent Semantic Indexing (LSI / LSA) using `TfidfVectorizer(stop_words='english')` and `TruncatedSVD(n_components=k, random_state=42)`.*

### Corpora Supported:
1. **(a) Exact 5-Document ML/NLP Corpus (Lab Manual)**:
   - ML Foundations, Deep Neural Networks, Natural Language Processing, Computer Vision, and AI Integration.
2. **(b) Customer Support / Opinion Mining Review Corpus**:
   - 12 realistic reviews across banking & academic services with explicit Positive/Negative sentiment tags.
3. **(c) Classic Deerwester Information Retrieval Corpus**:
   - 9 classic IR titles (HCI vs. Graph Theory).

### Mathematical Formulation:
$$A_{m \times n} \approx U_{m \times k} \cdot \Sigma_{k \times k} \cdot V_{k \times n}^T$$
- **Document-Term Matrix $A$ ($m \times n$)**: Initial TF-IDF feature space.
- **Document-Topic Matrix $U$ ($m \times k$)**: Orthonormal document coordinates in latent semantic concept space.
- **Singular Values Matrix $\Sigma$ ($k \times k$)**: Diagonal matrix holding singular energies $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k > 0$.
- **Topic-Term Loading Matrix $V^T$ ($k \times n$)**: Salient keyword loadings defining each latent semantic topic.

### Key Capabilities:
1. **Interactive Slider for $k$ Latent Concepts**: Dynamically re-factors SVD and updates matrices and similarity spaces.
2. **Explained Variance Ratio**: Plotly bar chart with cumulative curve measuring retained corpus information.
3. **Interactive Document Cosine Similarity Heatmap**:
   - Computes $S_{ij} = \frac{\mathbf{d}_i \cdot \mathbf{d}_j}{\|\mathbf{d}_i\|_2 \|\mathbf{d}_j\|_2}$ in $k$-dimensional latent space.
   - Toggle to compare side-by-side with raw TF-IDF space to visualize semantic concept clustering.
4. **Semantic Query Search & Opinion Mining**:
   - Projects user query: $q_{\text{LSI}} = \text{svd.transform}(q_{\text{tfidf}})$.
   - Ranks documents by LSI Cosine Similarity.
   - Computes **Semantic Boost** ($S_{\text{LSI}} - S_{\text{TF-IDF}}$), solving **synonymy and polysemy** even when exact keywords do not match!

---

### 🚀 Running the Application

1. **Launch the Unified Streamlit App (All 3 Modules)**:
```bash
python -m streamlit run app.py --server.port 8502
```
Access the application at: **`http://localhost:8502`**.

2. **Run Standalone Terminal 8-Option Menu for Assignment 7**:
```bash
python ngram_engine.py
```
This launches the interactive console menu directly in your command line.
