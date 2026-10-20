"""
tfidf_classifier.py - Core Engine for IntelliDesk (Assignment 6: TF-IDF Text Classification)
Provides dataset generation, TfidfVectorizer pipeline, mathematical verification,
model benchmarking (Logistic Regression, Multinomial Naive Bayes, Linear SVM),
and real-time query inference.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================================================================
# DATASETS
# ==============================================================================

# Banking Support Dataset (Exact lab manual categories: ATM, Account, Card, Loan, UPI)
# Expanded to 65+ realistic customer queries
BANKING_DATA: List[Dict[str, str]] = [
    # ATM (13 samples)
    {"text": "ATM machine did not dispense cash but amount deducted from my account", "category": "ATM"},
    {"text": "My ATM card got stuck inside the automated teller machine", "category": "ATM"},
    {"text": "How do I generate a new ATM green PIN at the kiosk?", "category": "ATM"},
    {"text": "ATM transaction failed due to cash dispenser shutter timeout", "category": "ATM"},
    {"text": "Nearest ATM is out of cash and screen is blank", "category": "ATM"},
    {"text": "Cash deposit machine accepted money but receipt was not printed", "category": "ATM"},
    {"text": "Exceeded daily ATM cash withdrawal limit of 25000 rupees", "category": "ATM"},
    {"text": "ATM screen displayed switch host error during withdrawal", "category": "ATM"},
    {"text": "Card swallowed by branch ATM after entering wrong PIN thrice", "category": "ATM"},
    {"text": "Charged extra fee for using third party bank ATM machine", "category": "ATM"},
    {"text": "ATM cash dispenser made counting noise but dispensed no currency notes", "category": "ATM"},
    {"text": "Where can I find an active 24x7 cash deposit and withdrawal ATM?", "category": "ATM"},
    {"text": "Unable to change debit card ATM pin at the ATM terminal", "category": "ATM"},

    # Account (13 samples)
    {"text": "I want to open a new zero balance savings bank account", "category": "Account"},
    {"text": "How can I download last six months bank account statement pdf?", "category": "Account"},
    {"text": "Please update my registered mobile number and email address in account", "category": "Account"},
    {"text": "My savings account status is showing dormant due to inactivity", "category": "Account"},
    {"text": "Need to submit re-KYC documents and address proof for my account", "category": "Account"},
    {"text": "How to request a new 50 leaf cheque book for current account?", "category": "Account"},
    {"text": "What is the minimum average balance requirement for salary account?", "category": "Account"},
    {"text": "Procedure to add a nominee to my joint savings account", "category": "Account"},
    {"text": "Close my inactive secondary bank account and refund remaining balance", "category": "Account"},
    {"text": "Check current balance and recent transactions on my savings account", "category": "Account"},
    {"text": "Transfer funds between my primary savings account and fixed deposit", "category": "Account"},
    {"text": "Quarterly interest rate credit details for senior citizen account", "category": "Account"},
    {"text": "My net banking login password expired for online account access", "category": "Account"},

    # Card (13 samples)
    {"text": "Block my lost credit card immediately to prevent unauthorized fraud", "category": "Card"},
    {"text": "Enable international contactless transactions on my Visa debit card", "category": "Card"},
    {"text": "Request a replacement for damaged chip on Mastercard credit card", "category": "Card"},
    {"text": "Apply for credit limit enhancement on my platinum rewards credit card", "category": "Card"},
    {"text": "Credit card annual renewal fee charged despite meeting annual spend threshold", "category": "Card"},
    {"text": "How to redeem accumulated reward points on my lifestyle credit card?", "category": "Card"},
    {"text": "Debit card CVV number got erased and online payment rejected", "category": "Card"},
    {"text": "Convert my recent credit card purchase of laptop into 6 months EMI", "category": "Card"},
    {"text": "Generate virtual credit card for safe e-commerce transaction", "category": "Card"},
    {"text": "Credit card bill payment reflected late and finance charges added", "category": "Card"},
    {"text": "Track delivery status of newly dispatched contactless debit card", "category": "Card"},
    {"text": "Debit card expired this month how to order new contactless card", "category": "Card"},
    {"text": "Temporary freeze on credit card through mobile banking application", "category": "Card"},

    # Loan (13 samples)
    {"text": "What are the latest floating interest rates for home loan?", "category": "Loan"},
    {"text": "Check eligibility criteria and required documents for personal loan", "category": "Loan"},
    {"text": "Personal loan amount sanctioned but disbursement pending to account", "category": "Loan"},
    {"text": "How much down payment required for new electric car vehicle loan?", "category": "Loan"},
    {"text": "Foreclosure charges and procedure for closing education loan early", "category": "Loan"},
    {"text": "Download annual loan interest certificate for income tax deduction 80C", "category": "Loan"},
    {"text": "Calculate monthly EMI for 30 lakh housing loan over 20 years", "category": "Loan"},
    {"text": "Instant gold loan against 22 carat jewelry with minimal documentation", "category": "Loan"},
    {"text": "Apply for two wheeler bike loan with low down payment", "category": "Loan"},
    {"text": "Request loan moratorium or restructure repayment tenure due to emergency", "category": "Loan"},
    {"text": "Loan EMI auto debit bounced due to insufficient funds charge reversal", "category": "Loan"},
    {"text": "Business loan overdraft limit extension for small enterprise MSME", "category": "Loan"},
    {"text": "Obtain No Objection Certificate NOC after full home loan repayment", "category": "Loan"},

    # UPI (13 samples)
    {"text": "UPI payment failed on Google Pay but money debited from bank account", "category": "UPI"},
    {"text": "Forgot UPI security PIN how to reset UPI PIN using debit card details", "category": "UPI"},
    {"text": "QR code scanner in PhonePe showing merchant server unreachable", "category": "UPI"},
    {"text": "Daily UPI transaction limit of 1 lakh reached unable to send money", "category": "UPI"},
    {"text": "Money sent to wrong UPI ID and mobile number how to reverse transaction", "category": "UPI"},
    {"text": "UPI recurring mandate auto debited without prior notification reminder", "category": "UPI"},
    {"text": "Unable to link multiple bank accounts to my Paytm UPI virtual payment address", "category": "UPI"},
    {"text": "UPI transaction status showing pending at beneficiary bank server", "category": "UPI"},
    {"text": "Change primary bank account linked to BHIM UPI application", "category": "UPI"},
    {"text": "Suspect fraudulent UPI payment request link received on SMS", "category": "UPI"},
    {"text": "UPI peer to peer transfer timed out network connection error", "category": "UPI"},
    {"text": "How to set up UPI Lite for PIN-less small daily merchant payments", "category": "UPI"},
    {"text": "Payment gateway refund for failed UPI merchant order not credited yet", "category": "UPI"}
]

# Student Query Dataset (Categories: Exam, Placement, Fees, Hostel)
# 48 realistic queries
STUDENT_DATA: List[Dict[str, str]] = [
    # Exam (12 samples)
    {"text": "When will end semester examination timetable and schedule be announced?", "category": "Exam"},
    {"text": "Unable to download exam hall ticket admit card from student portal", "category": "Exam"},
    {"text": "Procedure to apply for revaluation and photocopy of midterm answer script", "category": "Exam"},
    {"text": "Registration deadline for supplementary backlog compartment exam", "category": "Exam"},
    {"text": "Discrepancy in internal marks and end semester grade sheet marksheet", "category": "Exam"},
    {"text": "Exam hall seating arrangement and room allocation list for tomorrow", "category": "Exam"},
    {"text": "Medical certificate submission for missing mid semester exam test", "category": "Exam"},
    {"text": "Syllabus and blueprint pattern for final semester theory examination", "category": "Exam"},
    {"text": "How to apply for duplicate degree certificate and official transcript?", "category": "Exam"},
    {"text": "Clash of two elective exam dates in university provisional timetable", "category": "Exam"},
    {"text": "Minimum passing marks criteria and credit requirements for next semester", "category": "Exam"},
    {"text": "Carry forward backlog policy rules for engineering students", "category": "Exam"},

    # Placement (12 samples)
    {"text": "Campus placement drive eligibility criteria for computer science students", "category": "Placement"},
    {"text": "How to get resume verified and approved by training placement cell?", "category": "Placement"},
    {"text": "Schedule of upcoming mock technical interviews and aptitude preparation", "category": "Placement"},
    {"text": "List of tier 1 IT companies visiting campus for final year hiring", "category": "Placement"},
    {"text": "NOC certificate application for 6 months corporate internship semester", "category": "Placement"},
    {"text": "Pre-placement talk PPT attendance mandatory for registration in drive", "category": "Placement"},
    {"text": "Placement policy regarding second job offer letter dream company", "category": "Placement"},
    {"text": "Upload updated resume profile and GitHub projects on placement portal", "category": "Placement"},
    {"text": "Average package CTC and hiring statistics for previous batch students", "category": "Placement"},
    {"text": "Registration link for national hackathon and campus hiring challenge", "category": "Placement"},
    {"text": "Interview dress code and identity card guidelines for placement drives", "category": "Placement"},
    {"text": "Offer letter verification and joining date extension request procedure", "category": "Placement"},

    # Fees (12 samples)
    {"text": "How to pay college semester tuition fees through online net banking?", "category": "Fees"},
    {"text": "Tuition fee payment debited but fee receipt not generated on portal", "category": "Fees"},
    {"text": "Request for fee payment in two installments due to financial constraints", "category": "Fees"},
    {"text": "Late fee fine waiver application for delayed semester fee submission", "category": "Fees"},
    {"text": "Government post matric scholarship fee reimbursement refund status", "category": "Fees"},
    {"text": "Download official fee structure breakdown certificate for education bank loan", "category": "Fees"},
    {"text": "Hostel mess fee refund policy after vacating college hostel room", "category": "Fees"},
    {"text": "Examination form registration fee payment portal link and last date", "category": "Fees"},
    {"text": "Bus transport facility fee payment and annual identity pass issuance", "category": "Fees"},
    {"text": "Caution deposit refund after completion of four year degree program", "category": "Fees"},
    {"text": "Receipt needed for income tax deduction on university educational fees", "category": "Fees"},
    {"text": "Challan submission verification at campus accounts department counter", "category": "Fees"},

    # Hostel (12 samples)
    {"text": "Hostel room allotment list for second year students out on notice board", "category": "Hostel"},
    {"text": "Complaint regarding poor quality and hygiene of hostel mess food", "category": "Hostel"},
    {"text": "Room ceiling fan and bathroom water heater geyser not working repair request", "category": "Hostel"},
    {"text": "Hostel outpass permission and night curfew gate closing time policy", "category": "Hostel"},
    {"text": "How to request hostel room change with another roommate in block B?", "category": "Hostel"},
    {"text": "Hostel high speed wifi internet connection disconnected in dormitory", "category": "Hostel"},
    {"text": "Application for staying in hostel during summer vacation project work", "category": "Hostel"},
    {"text": "Laundry service schedule and washing machine availability in hostel", "category": "Hostel"},
    {"text": "Warden contact number and hostel disciplinary rules regulations booklet", "category": "Hostel"},
    {"text": "Drinking water filter RO cooler repair complaint on third floor", "category": "Hostel"},
    {"text": "Hostel room inventory checklist handover during end of semester", "category": "Hostel"},
    {"text": "Guest room booking for parents visiting university campus accommodation", "category": "Hostel"}
]


def load_dataset(dataset_name: str = "banking") -> pd.DataFrame:
    """Load selected dataset as a pandas DataFrame."""
    if dataset_name.lower().startswith("student"):
        data = STUDENT_DATA
    else:
        data = BANKING_DATA
    return pd.DataFrame(data)


# ==============================================================================
# TF-IDF CLASSIFICATION PIPELINE
# ==============================================================================

class TFIDFClassifierPipeline:
    """
    Modular TF-IDF Classification System supporting TfidfVectorizer,
    step-by-step mathematical verification, multi-model benchmark,
    and real-time inference.
    """
    def __init__(self, lowercase: bool = True, stop_words: str = "english"):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(lowercase=lowercase, stop_words=stop_words)
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
            "Linear SVM": LinearSVC(random_state=42)
        }
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.vocab: Dict[str, int] = {}
        self.classes: List[str] = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.benchmark_results: Dict[str, Any] = {}
        self.corpus_docs: List[str] = []

    def fit_and_benchmark(self, df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        """
        Fit TfidfVectorizer on corpus, split dataset, train 3 models,
        and evaluate performance metrics.
        """
        self.corpus_docs = df["text"].tolist()
        y = df["category"].values

        # Fit vectorizer on full corpus so vocabulary covers all dataset terms
        X_all = self.vectorizer.fit_transform(self.corpus_docs)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        self.vocab = self.vectorizer.vocabulary_
        self.classes = sorted(list(set(y)))

        # Stratified split to ensure balanced class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_all, y, test_size=test_size, random_state=random_state, stratify=y
        )

        results = {}
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, target_names=self.classes, output_dict=True, zero_division=0)
            cm = confusion_matrix(self.y_test, y_pred, labels=self.classes)

            results[name] = {
                "accuracy": acc,
                "report": report,
                "confusion_matrix": cm,
                "y_pred": y_pred
            }

        self.benchmark_results = results
        self.is_fitted = True
        return results

    def get_vocabulary_df(self) -> pd.DataFrame:
        """
        Return DataFrame containing vocabulary terms, their feature index,
        and their IDF score learned by scikit-learn.
        """
        if not self.is_fitted:
            return pd.DataFrame()

        idf_scores = self.vectorizer.idf_
        analyzer = self.vectorizer.build_analyzer()
        
        # Calculate DF for each word
        df_counts = {}
        for doc in self.corpus_docs:
            seen_tokens = set(analyzer(doc))
            for tok in seen_tokens:
                if tok in self.vocab:
                    df_counts[tok] = df_counts.get(tok, 0) + 1

        data = []
        for word, idx in self.vocab.items():
            data.append({
                "Term": word,
                "Feature Index": idx,
                "Document Frequency (DF)": df_counts.get(word, 0),
                "IDF Weight": round(idf_scores[idx], 5)
            })

        df_vocab = pd.DataFrame(data).sort_values(by="Term").reset_index(drop=True)
        return df_vocab

    def get_doc_term_matrix_df(self, sample_docs: List[str] = None, max_cols: int = 25) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Return Document-Term TF-IDF Matrix as a styled pandas DataFrame,
        along with sparsity statistics.
        """
        if not self.is_fitted:
            return pd.DataFrame(), {}

        if sample_docs is None:
            sample_docs = self.corpus_docs[:12]

        matrix = self.vectorizer.transform(sample_docs)
        matrix_array = matrix.toarray()

        total_elements = matrix_array.size
        non_zeros = np.count_nonzero(matrix_array)
        sparsity_pct = (1.0 - (non_zeros / total_elements)) * 100 if total_elements > 0 else 0

        # Choose top columns by max value across samples for clean visualization
        col_means = matrix_array.mean(axis=0)
        top_col_indices = np.argsort(col_means)[::-1][:max_cols]
        top_col_indices = sorted(top_col_indices)

        selected_features = [self.feature_names[i] for i in top_col_indices]
        display_array = matrix_array[:, top_col_indices]

        doc_labels = [f"Doc {i+1}: {doc[:42]}..." if len(doc) > 42 else f"Doc {i+1}: {doc}" for i, doc in enumerate(sample_docs)]
        df_dtm = pd.DataFrame(np.round(display_array, 4), index=doc_labels, columns=selected_features)

        stats = {
            "Total Documents": len(sample_docs),
            "Total Vocabulary Features": len(self.feature_names),
            "Non-zero Entries": int(non_zeros),
            "Total Matrix Elements": int(total_elements),
            "Sparsity Percentage": round(sparsity_pct, 2)
        }

        return df_dtm, stats

    def verify_math_step_by_step(self, doc_text: str, target_term: str) -> Dict[str, Any]:
        """
        Performs manual step-by-step mathematical verification of TF-IDF for a given
        document and target word, following exact scikit-learn smooth_idf=True & L2 norm formula:

        IDF(t) = ln((1 + N) / (1 + DF(t))) + 1
        TF-IDF_raw(t, d) = TF(t, d) * IDF(t)
        TF-IDF_norm(t, d) = TF-IDF_raw(t, d) / sqrt(sum(TF-IDF_raw(w, d)^2))
        """
        analyzer = self.vectorizer.build_analyzer()
        raw_tokens = analyzer(doc_text)
        target_term = target_term.strip().lower()

        # In scikit-learn, only tokens present in self.vocab are retained in the document vector!
        valid_tokens = [tok for tok in raw_tokens if tok in self.vocab]
        
        train_docs = self.corpus_docs
        N = len(train_docs)

        # 1. Term Frequency in document d (count in doc_text)
        tf_raw = raw_tokens.count(target_term)

        # 2. Document Frequency DF(t) across training corpus
        df_count = sum(1 for d in train_docs if target_term in analyzer(d))

        # 3. Smoothed IDF
        idf_smooth = math.log((1 + N) / (1 + df_count)) + 1.0

        # 4. Compute unnormalized weights for all UNIQUE VALID tokens in doc d
        unique_valid_tokens = sorted(list(set(valid_tokens)))
        token_breakdowns = []
        raw_weights_squared_sum = 0.0

        for tok in unique_valid_tokens:
            tok_tf = valid_tokens.count(tok)
            tok_df = sum(1 for d in train_docs if tok in analyzer(d))
            tok_idf = math.log((1 + N) / (1 + tok_df)) + 1.0
            tok_raw_tfidf = tok_tf * tok_idf
            raw_weights_squared_sum += (tok_raw_tfidf ** 2)
            token_breakdowns.append({
                "token": tok,
                "tf": tok_tf,
                "df": tok_df,
                "idf": round(tok_idf, 6),
                "raw_tfidf": round(tok_raw_tfidf, 6),
                "squared": round(tok_raw_tfidf ** 2, 6)
            })

        l2_norm = math.sqrt(raw_weights_squared_sum) if raw_weights_squared_sum > 0 else 1.0
        
        # If target term is valid in doc, compute manual TF-IDF
        if target_term in valid_tokens and l2_norm > 0:
            manual_tfidf = (tf_raw * idf_smooth) / l2_norm
        else:
            manual_tfidf = 0.0

        # 5. Fetch Scikit-Learn's actual matrix value
        sk_vector = self.vectorizer.transform([doc_text])
        if target_term in self.vocab:
            feature_idx = self.vocab[target_term]
            sklearn_tfidf = float(sk_vector[0, feature_idx])
        else:
            feature_idx = -1
            sklearn_tfidf = 0.0

        diff = abs(manual_tfidf - sklearn_tfidf)
        is_verified = (diff < 1e-6)

        return {
            "document": doc_text,
            "target_term": target_term,
            "raw_tokens": raw_tokens,
            "valid_tokens": valid_tokens,
            "N": N,
            "tf_raw": tf_raw,
            "df_count": df_count,
            "idf_formula": r"\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1",
            "idf_smooth": idf_smooth,
            "token_breakdowns": token_breakdowns,
            "sum_squares": raw_weights_squared_sum,
            "l2_norm": l2_norm,
            "manual_tfidf": manual_tfidf,
            "sklearn_tfidf": sklearn_tfidf,
            "difference": diff,
            "is_verified": is_verified,
            "feature_index": feature_idx
        }

    def predict_query(self, query_text: str, model_name: str = "Linear SVM") -> Dict[str, Any]:
        """
        Classify real-time user query, returning predicted class,
        confidence scores for all classes, and contributing TF-IDF terms.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be trained first.")

        model = self.models.get(model_name, self.models["Linear SVM"])
        query_vec = self.vectorizer.transform([query_text])

        # Predict class
        predicted_class = model.predict(query_vec)[0]

        # Probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(query_vec)[0]
            classes = model.classes_
        else:
            # LinearSVC decision function with softmax
            decision = model.decision_function(query_vec)
            if decision.ndim == 1 or decision.shape[0] == 1:
                dec = decision[0] if decision.ndim > 1 else decision
                exp_d = np.exp(dec - np.max(dec))
                probs = exp_d / exp_d.sum()
            else:
                probs = np.ones(len(model.classes_)) / len(model.classes_)
            classes = model.classes_

        class_confidences = {cls: float(prob) for cls, prob in zip(classes, probs)}

        # Extract active TF-IDF terms in query
        analyzer = self.vectorizer.build_analyzer()
        query_tokens = analyzer(query_text)
        contributing_terms = []
        vec_array = query_vec.toarray()[0]

        for token in sorted(list(set(query_tokens))):
            if token in self.vocab:
                idx = self.vocab[token]
                score = vec_array[idx]
                if score > 0:
                    contributing_terms.append({
                        "term": token,
                        "tfidf": round(float(score), 4),
                        "idf": round(float(self.vectorizer.idf_[idx]), 4)
                    })

        # Sort terms by TF-IDF score descending
        contributing_terms.sort(key=lambda x: x["tfidf"], reverse=True)

        return {
            "query": query_text,
            "model_used": model_name,
            "predicted_class": predicted_class,
            "confidences": class_confidences,
            "confidence_pct": round(class_confidences.get(predicted_class, 0.0) * 100, 2),
            "contributing_terms": contributing_terms
        }
