# IntelliDesk: AI-Powered Support Query Routing & NLP Lab
## NLP Laboratory Suite: Assignment 6 & Assignment 7

---

### 📌 Overview
**IntelliDesk** is a dual-module NLP intelligence and evaluation platform designed for:
1. **Module 1 (Assignment 6)**: TF-IDF Text Classification & Department Routing.
2. **Module 2 (Assignment 7)**: N-Gram Language Models, Laplace (Add-1) Smoothing & Perplexity Evaluation.

The system features two completely separated tabs and settings controls, allowing students and instructors to independently inspect, verify, benchmark, and interact with both assignments.

---

### 📁 Architecture & File Structure
```
Asg-6_7/
├── app.py                  # Main Streamlit UI (Tab 1: Asg 6, Tab 2: Asg 7)
├── tfidf_classifier.py     # Module 1: TF-IDF Vectorizer, Math Proof, Classifiers
├── ngram_engine.py         # Module 2: Pure Scratch N-Gram Language Model & 8-Option Lab Menu
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

### Corpora Supported:
1. **Default Lab Manual NLP Corpus**: 15 foundational NLP sentences covering language modeling, smoothing, and perplexity.
2. **Support Ticket Domain Corpus**: 113 realistic banking and student queries from Module 1.
3. **Custom User Corpus**: Interactive text area allowing users to enter custom domain text.

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

### 🚀 Running the Application

1. **Launch the Unified Streamlit App**:
```bash
python -m streamlit run app.py --server.port 8502
```
Access the application at: **`http://localhost:8502`**.

2. **Run Standalone Terminal 8-Option Menu for Assignment 7**:
```bash
python ngram_engine.py
```
This runs the interactive CLI menu directly in your terminal following the exact numbered lab manual format.
