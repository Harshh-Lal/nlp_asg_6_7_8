"""
lsi_engine.py - Core Engine for IntelliDesk Module 3 (Assignment 8: Latent Semantic Indexing)
Implements Latent Semantic Indexing (LSI / LSA) using TfidfVectorizer and TruncatedSVD.

Features:
- Exact 5-Document ML/NLP corpus from Lab Manual Assignment 7
- Customer Support / Opinion Mining review corpus
- SVD Decomposition: A ≈ U * Σ * V^T
- Explained Variance Ratio & Cumulative Variance
- Document-to-Document Cosine Similarity in Latent Semantic Space
- Query Projection, Semantic Search, and Document Categorization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# CORPORA DEFINITIONS
# ==============================================================================

# (a) Exact 5-Document ML/NLP Corpus from the Lab Manual
LAB_MANUAL_ML_CORPUS: List[Dict[str, str]] = [
    {
        "id": "Doc 1",
        "title": "Machine Learning Foundations",
        "text": "Machine learning algorithms learn statistical patterns and predictive models from structured training data",
        "category": "Machine Learning"
    },
    {
        "id": "Doc 2",
        "title": "Deep Neural Networks",
        "text": "Deep learning neural networks require massive training datasets and high performance GPU compute clusters",
        "category": "Deep Learning"
    },
    {
        "id": "Doc 3",
        "title": "Natural Language Processing",
        "text": "Natural language processing models understand human speech and analyze semantic text syntax",
        "category": "NLP"
    },
    {
        "id": "Doc 4",
        "title": "Computer Vision Systems",
        "text": "Computer vision algorithms detect visual objects and recognize digital video frames",
        "category": "Computer Vision"
    },
    {
        "id": "Doc 5",
        "title": "Artificial Intelligence Integration",
        "text": "Artificial intelligence systems integrate machine learning models with natural language processing technology",
        "category": "Artificial Intelligence"
    }
]

# (b) Customer Support / Opinion Mining & Sentiment Review Corpus
CUSTOMER_SUPPORT_OPINION_CORPUS: List[Dict[str, str]] = [
    {
        "id": "Rev 1",
        "title": "Fast ATM Dispenser",
        "text": "The automated teller machine was extremely fast and dispensed crisp cash notes without any delay",
        "category": "ATM",
        "sentiment": "Positive"
    },
    {
        "id": "Rev 2",
        "title": "ATM Swallowed Card",
        "text": "ATM machine swallowed my debit card and screen crashed showing switch host error during withdrawal",
        "category": "ATM",
        "sentiment": "Negative"
    },
    {
        "id": "Rev 3",
        "title": "Intuitive Mobile Banking",
        "text": "Savings bank account mobile application has an intuitive interface and easy statement download",
        "category": "Account",
        "sentiment": "Positive"
    },
    {
        "id": "Rev 4",
        "title": "Dormant Account Frustration",
        "text": "Cannot access net banking portal because savings account is dormant and KYC verification keeps failing",
        "category": "Account",
        "sentiment": "Negative"
    },
    {
        "id": "Rev 5",
        "title": "Credit Card Reward Points",
        "text": "Redeemed platinum credit card reward points for travel voucher with zero extra transaction fees",
        "category": "Card",
        "sentiment": "Positive"
    },
    {
        "id": "Rev 6",
        "title": "Fraudulent Charge Dispute",
        "text": "Unauthorized international transaction charged on my Visa credit card and helpline support was rude",
        "category": "Card",
        "sentiment": "Negative"
    },
    {
        "id": "Rev 7",
        "title": "Low Interest Housing Loan",
        "text": "Home loan approval and disbursement completed swiftly with attractive floating interest rate and low EMI",
        "category": "Loan",
        "sentiment": "Positive"
    },
    {
        "id": "Rev 8",
        "title": "Exorbitant Loan Charges",
        "text": "Personal loan foreclosure fees and hidden processing charges were exorbitant despite early loan repayment",
        "category": "Loan",
        "sentiment": "Negative"
    },
    {
        "id": "Rev 9",
        "title": "Instant UPI QR Payment",
        "text": "UPI merchant QR payment on Google Pay was lightning fast and seamless for grocery checkout",
        "category": "UPI",
        "sentiment": "Positive"
    },
    {
        "id": "Rev 10",
        "title": "Pending UPI Debit",
        "text": "Money debited from bank account but UPI transaction status shows pending and merchant received nothing",
        "category": "UPI",
        "sentiment": "Negative"
    },
    {
        "id": "Rev 11",
        "title": "Prompt Exam Schedule",
        "text": "Examination department promptly uploaded end semester exam timetable and admit card on student portal",
        "category": "Student / Exam",
        "sentiment": "Positive"
    },
    {
        "id": "Rev 12",
        "title": "Disorganized Placement Drive",
        "text": "Campus placement drive technical interviews were heavily delayed and test portal server crashed repeatedly",
        "category": "Student / Placement",
        "sentiment": "Negative"
    }
]

# Classic 9-Document Deerwester Information Retrieval Corpus (Optional classic reference)
DEERWESTER_IR_CORPUS: List[Dict[str, str]] = [
    {"id": "c1", "title": "HCI 1", "text": "Human machine interface for lab computer applications", "category": "HCI"},
    {"id": "c2", "title": "HCI 2", "text": "A survey of user opinion of computer system response time", "category": "HCI"},
    {"id": "c3", "title": "HCI 3", "text": "The EPS user interface management system", "category": "HCI"},
    {"id": "c4", "title": "HCI 4", "text": "System and human system engineering testing of EPS", "category": "HCI"},
    {"id": "c5", "title": "HCI 5", "text": "Relation of user perceived response time to error measurement", "category": "HCI"},
    {"id": "m1", "title": "Graph 1", "text": "The generation of random binary unordered trees", "category": "Graph Theory"},
    {"id": "m2", "title": "Graph 2", "text": "The intersection graph of paths in trees", "category": "Graph Theory"},
    {"id": "m3", "title": "Graph 3", "text": "Graph minors IV Widths of trees and well quasi ordering", "category": "Graph Theory"},
    {"id": "m4", "title": "Graph 4", "text": "Graph minors A survey of quasi ordering applications", "category": "Graph Theory"}
]


# ==============================================================================
# LATENT SEMANTIC INDEXING (LSI) ENGINE
# ==============================================================================

class LatentSemanticIndexingEngine:
    """
    Complete Latent Semantic Indexing Engine implementing:
    1. TfidfVectorizer Document-Term Matrix A
    2. TruncatedSVD decomposition into U_k, Σ_k, V_k^T
    3. Latent Topic profiling and keyword loadings
    4. Explained Variance Ratio and SVD diagnostics
    5. Document-to-Document Cosine Similarity Heatmap in LSI space
    6. Semantic Query Search and Document Categorization
    """

    def __init__(self, stop_words: str = "english", random_state: int = 42):
        self.stop_words = stop_words
        self.random_state = random_state
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.documents_metadata: List[Dict[str, Any]] = []
        self.documents_text: List[str] = []
        self.vocabulary: List[str] = []
        self.n_components: int = 2
        self.is_fitted: bool = False

        # Matrix Storage
        self.matrix_A = None               # TF-IDF Document-Term Matrix (m x n)
        self.U_matrix = None               # Document-Topic Matrix (m x k)
        self.sigma_vector = None           # Singular values Σ_k (k,)
        self.VT_matrix = None              # Topic-Term Matrix V^T (k x n)
        self.doc_lsi_embeddings = None     # Document coordinates in latent space (m x k)
        self.lsi_cosine_sim_matrix = None  # Document-to-document similarity in LSI space (m x m)
        self.tfidf_cosine_sim_matrix = None # Document-to-document similarity in raw TF-IDF space (m x m)

    def fit(self, documents: List[Dict[str, str]], n_components: int = 2) -> "LatentSemanticIndexingEngine":
        """
        Fits TF-IDF vectorizer and TruncatedSVD on the corpus documents.
        """
        self.documents_metadata = documents
        self.documents_text = [d["text"] for d in documents]
        m = len(self.documents_text)

        # Validate n_components: must be strictly less than min(m, n_features)
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, lowercase=True)
        self.matrix_A = self.vectorizer.fit_transform(self.documents_text)
        self.vocabulary = self.vectorizer.get_feature_names_out().tolist()
        n = len(self.vocabulary)

        # Bound k within valid mathematical limits
        max_valid_k = max(2, min(m - 1, n - 1))
        self.n_components = min(max(2, n_components), max_valid_k)

        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        # svd.fit_transform(A) yields U * Σ (document coordinates in latent topic space)
        self.doc_lsi_embeddings = self.svd.fit_transform(self.matrix_A)

        self.sigma_vector = self.svd.singular_values_
        self.VT_matrix = self.svd.components_

        # Calculate orthonormal U matrix: U = (U * Σ) / Σ
        # Safe element-wise division by singular values
        safe_sigma = np.where(self.sigma_vector > 0, self.sigma_vector, 1.0)
        self.U_matrix = self.doc_lsi_embeddings / safe_sigma

        # Compute document-to-document cosine similarity in LSI space
        self.lsi_cosine_sim_matrix = cosine_similarity(self.doc_lsi_embeddings)

        # Compute document-to-document cosine similarity in raw TF-IDF space for comparison
        self.tfidf_cosine_sim_matrix = cosine_similarity(self.matrix_A.toarray())

        self.is_fitted = True
        return self

    # ==========================================================================
    # MATRIX DATA VIEWS
    # ==========================================================================

    def get_matrix_a_df(self, max_features: int = 35) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Return the TF-IDF Document-Term Matrix A as a styled DataFrame.
        """
        if not self.is_fitted:
            return pd.DataFrame(), {}

        matrix_dense = self.matrix_A.toarray()
        doc_labels = [f"{d['id']}: {d['title']}" for d in self.documents_metadata]

        # Select most prominent features by column sum
        col_sums = matrix_dense.sum(axis=0)
        top_feature_indices = np.argsort(col_sums)[::-1][:max_features]
        top_feature_indices = sorted(top_feature_indices)

        selected_features = [self.vocabulary[i] for i in top_feature_indices]
        display_array = matrix_dense[:, top_feature_indices]

        df_A = pd.DataFrame(
            np.round(display_array, 4),
            index=doc_labels,
            columns=selected_features
        )

        stats = {
            "Total Documents (m)": matrix_dense.shape[0],
            "Total Vocabulary Terms (n)": matrix_dense.shape[1],
            "Matrix Shape (m x n)": f"{matrix_dense.shape[0]} x {matrix_dense.shape[1]}",
            "Non-zero Entries": int(np.count_nonzero(matrix_dense)),
            "Sparsity (%)": round((1.0 - (np.count_nonzero(matrix_dense) / matrix_dense.size)) * 100, 2)
        }

        return df_A, stats

    def get_decomposed_matrices(self) -> Dict[str, pd.DataFrame]:
        """
        Return DataFrames for the 3 SVD decomposed matrices:
        1. U (Document-Topic Matrix: m x k)
        2. Sigma (Singular Values Diagonal Matrix: k x k)
        3. V^T (Topic-Term Loading Matrix: k x n)
        """
        if not self.is_fitted:
            return {}

        topic_labels = [f"Latent Topic {i+1}" for i in range(self.n_components)]
        doc_labels = [f"{d['id']}: {d['title']}" for d in self.documents_metadata]

        # 1. U Matrix DataFrame (m x k)
        df_U = pd.DataFrame(
            np.round(self.U_matrix, 4),
            index=doc_labels,
            columns=topic_labels
        )

        # 2. Sigma DataFrame (Diagonal k x k)
        sigma_diag = np.diag(self.sigma_vector)
        df_sigma = pd.DataFrame(
            np.round(sigma_diag, 4),
            index=topic_labels,
            columns=topic_labels
        )

        # 3. V^T Matrix DataFrame (k x n)
        # Select top 25 terms by maximum loading magnitude across all topics
        loading_magnitudes = np.max(np.abs(self.VT_matrix), axis=0)
        top_term_indices = np.argsort(loading_magnitudes)[::-1][:25]
        top_term_indices = sorted(top_term_indices)

        selected_terms = [self.vocabulary[i] for i in top_term_indices]
        display_VT = self.VT_matrix[:, top_term_indices]

        df_VT = pd.DataFrame(
            np.round(display_VT, 4),
            index=topic_labels,
            columns=selected_terms
        )

        return {
            "U": df_U,
            "Sigma": df_sigma,
            "VT": df_VT,
            "singular_values": self.sigma_vector.tolist()
        }

    def get_latent_topics_profile(self, top_n_keywords: int = 5) -> List[Dict[str, Any]]:
        """
        Extract the top positive and negative keyword loadings defining each latent semantic topic.
        """
        if not self.is_fitted:
            return []

        profiles = []
        for i in range(self.n_components):
            row = self.VT_matrix[i]
            
            # Top positive loading keywords
            top_pos_idx = np.argsort(row)[::-1][:top_n_keywords]
            top_pos_terms = [(self.vocabulary[idx], round(float(row[idx]), 4)) for idx in top_pos_idx if row[idx] > 0]

            # Top negative loading keywords (bipolar contrast in SVD)
            top_neg_idx = np.argsort(row)[:top_n_keywords]
            top_neg_terms = [(self.vocabulary[idx], round(float(row[idx]), 4)) for idx in top_neg_idx if row[idx] < 0]

            # Strongest overall magnitude
            top_mag_idx = np.argsort(np.abs(row))[::-1][:top_n_keywords]
            top_mag_terms = [(self.vocabulary[idx], round(float(row[idx]), 4)) for idx in top_mag_idx]

            var_pct = round(float(self.svd.explained_variance_ratio_[i]) * 100, 2)
            sing_val = round(float(self.sigma_vector[i]), 4)

            profiles.append({
                "topic_index": i + 1,
                "topic_name": f"Latent Concept {i+1}",
                "singular_value": sing_val,
                "variance_explained_pct": var_pct,
                "top_positive": top_pos_terms,
                "top_negative": top_neg_terms,
                "dominant_keywords": [t[0] for t in top_mag_terms]
            })

        return profiles

    def get_explained_variance_data(self) -> Dict[str, Any]:
        """
        Return individual and cumulative explained variance ratio data.
        """
        if not self.is_fitted:
            return {}

        evr = self.svd.explained_variance_ratio_ * 100
        cum_evr = np.cumsum(evr)

        return {
            "components": [f"Topic {i+1}" for i in range(self.n_components)],
            "individual_variance": [round(float(v), 2) for v in evr],
            "cumulative_variance": [round(float(v), 2) for v in cum_evr],
            "total_variance_explained": round(float(cum_evr[-1]), 2)
        }

    # ==========================================================================
    # SEMANTIC QUERY SEARCH & CATEGORIZATION
    # ==========================================================================

    def query_semantic_search(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Project user query into the k-dimensional LSI latent space:
        q_lsi = q_tfidf * V_k * Σ_k^(-1) = svd.transform(q_tfidf)
        Computes cosine similarity against all documents in LSI space and raw TF-IDF space.
        """
        if not self.is_fitted:
            raise RuntimeError("LSI Engine must be fitted first.")

        query_cleaned = query_text.strip()
        q_tfidf = self.vectorizer.transform([query_cleaned])
        q_lsi = self.svd.transform(q_tfidf)  # Shape (1, k)

        # Check if any vocabulary words matched in raw TF-IDF
        raw_matched_words = [
            w for w in self.vectorizer.build_analyzer()(query_cleaned)
            if w in self.vectorizer.vocabulary_
        ]

        # 1. Cosine similarity in LSI space
        lsi_sims = cosine_similarity(q_lsi, self.doc_lsi_embeddings)[0]

        # 2. Cosine similarity in raw TF-IDF space
        tfidf_sims = cosine_similarity(q_tfidf, self.matrix_A)[0]

        # Rank documents by LSI similarity
        ranked_indices = np.argsort(lsi_sims)[::-1]

        ranked_results = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            doc_meta = self.documents_metadata[idx]
            ranked_results.append({
                "rank": rank,
                "doc_id": doc_meta["id"],
                "title": doc_meta["title"],
                "category": doc_meta.get("category", "General"),
                "sentiment": doc_meta.get("sentiment", "Neutral"),
                "text": doc_meta["text"],
                "lsi_similarity": round(float(lsi_sims[idx]), 4),
                "tfidf_similarity": round(float(tfidf_sims[idx]), 4),
                "semantic_boost": round(float(lsi_sims[idx] - tfidf_sims[idx]), 4)
            })

        # Identify dominant latent topic for query
        topic_loadings = q_lsi[0]
        dominant_topic_idx = int(np.argmax(np.abs(topic_loadings)))
        dominant_topic_sign = "Positive" if topic_loadings[dominant_topic_idx] >= 0 else "Negative"

        # Categorization: predicted category based on top-1 LSI match
        top_match = ranked_results[0] if ranked_results else None
        predicted_category = top_match["category"] if top_match else "Unknown"

        return {
            "query": query_cleaned,
            "raw_matched_tokens": raw_matched_words,
            "query_lsi_vector": [round(float(v), 4) for v in q_lsi[0]],
            "dominant_topic": f"Latent Topic {dominant_topic_idx + 1} ({dominant_topic_sign})",
            "dominant_topic_index": dominant_topic_idx + 1,
            "predicted_category": predicted_category,
            "ranked_documents": ranked_results
        }


# ==============================================================================
# INTERACTIVE CLI DEMO & LAB RUNNER
# ==============================================================================

def run_cli_lsi():
    """Interactive command-line execution for Assignment 8 (Latent Semantic Indexing)."""
    print("=" * 75)
    print("  ASSIGNMENT 8: LATENT SEMANTIC INDEXING (LSI) & TRUNCATED SVD")
    print("=" * 75)

    engine = LatentSemanticIndexingEngine(stop_words="english", random_state=42)
    current_corpus = LAB_MANUAL_ML_CORPUS
    current_name = "Exact 5-Document ML/NLP Corpus (Lab Manual)"
    k = 2

    # Fit default
    engine.fit(current_corpus, n_components=k)

    while True:
        print("\n" + "-" * 75)
        print(f"  ACTIVE CORPUS: {current_name}")
        print(f"  ACTIVE LATENT CONCEPTS: k = {engine.n_components}")
        print("  SELECT AN OPTION:")
        print("  1. Switch / Inspect Corpus (ML/NLP vs Customer Opinion vs Deerwester)")
        print("  2. Display Document-Term TF-IDF Matrix A & Vocabulary")
        print("  3. Display Decomposed SVD Matrices (U, Sigma, V^T)")
        print("  4. Display Latent Topic Profiles & Salient Keyword Loadings")
        print("  5. Display Explained Variance Ratio (Individual & Cumulative)")
        print("  6. Display Document-to-Document Cosine Similarity Matrix (LSI vs TF-IDF)")
        print("  7. Run Semantic Query Search & Categorization (with Semantic Boost)")
        print("  8. Change Latent Dimension k")
        print("  0. Exit")
        print("-" * 75)

        choice = input("Enter choice (0-8): ").strip()

        if choice == "0":
            print("Exiting LSI Engine. Goodbye!")
            break

        elif choice == "1":
            print("\n[Corpus Options]:")
            print("  1. Exact 5-Document ML/NLP Corpus (Lab Manual)")
            print("  2. Customer Support / Opinion Mining Review Corpus (12 reviews)")
            print("  3. Classic Deerwester IR Corpus (9 documents)")
            sub = input("Select (1-3): ").strip()
            if sub == "2":
                current_corpus = CUSTOMER_SUPPORT_OPINION_CORPUS
                current_name = "Customer Support / Opinion Mining Review Corpus"
                k = min(3, len(current_corpus) - 1)
            elif sub == "3":
                current_corpus = DEERWESTER_IR_CORPUS
                current_name = "Classic Deerwester IR Corpus"
                k = 2
            else:
                current_corpus = LAB_MANUAL_ML_CORPUS
                current_name = "Exact 5-Document ML/NLP Corpus (Lab Manual)"
                k = 2
            engine.fit(current_corpus, n_components=k)
            print(f"\nLoaded {len(current_corpus)} documents. SVD re-fit with k={engine.n_components}.")

        elif choice == "2":
            df_A, stats = engine.get_matrix_a_df(max_features=15)
            print(f"\n[Matrix A Stats]: {stats}")
            print("\nDocument-Term Matrix A (Sample 15 features):")
            print(df_A.to_string())

        elif choice == "3":
            decomp = engine.get_decomposed_matrices()
            print("\n[1. Document-Topic Matrix U (m x k)]:")
            print(decomp["U"].to_string())
            print("\n[2. Singular Values Matrix Sigma (k x k)]:")
            print(decomp["Sigma"].to_string())
            print("\n[3. Topic-Term Loading Matrix V^T (k x n) (Sample top 15 terms)]:")
            print(decomp["VT"].iloc[:, :15].to_string())

        elif choice == "4":
            profiles = engine.get_latent_topics_profile(top_n_keywords=5)
            print("\n[Latent Topic Profiles]:")
            for p in profiles:
                print(f"\n* {p['topic_name']} (Singular Value: {p['singular_value']}, Explained Variance: {p['variance_explained_pct']}%):")
                print(f"  Dominant Keywords: {', '.join(p['dominant_keywords'])}")
                print(f"  Top Positive Loadings: {p['top_positive']}")
                print(f"  Top Negative Loadings: {p['top_negative']}")

        elif choice == "5":
            var_data = engine.get_explained_variance_data()
            print("\n[Explained Variance Ratio]:")
            for comp, ind_v, cum_v in zip(var_data["components"], var_data["individual_variance"], var_data["cumulative_variance"]):
                print(f"  {comp}: Individual = {ind_v}% | Cumulative = {cum_v}%")
            print(f"  Total Variance Retained: {var_data['total_variance_explained']}%")

        elif choice == "6":
            print("\n[Document-to-Document Cosine Similarity Matrix (LSI Latent Space)]:")
            doc_labels = [d["id"] for d in current_corpus]
            df_sim_lsi = pd.DataFrame(np.round(engine.lsi_cosine_sim_matrix, 4), index=doc_labels, columns=doc_labels)
            print(df_sim_lsi.to_string())

            print("\n[Document-to-Document Cosine Similarity Matrix (Raw TF-IDF Space)]:")
            df_sim_tfidf = pd.DataFrame(np.round(engine.tfidf_cosine_sim_matrix, 4), index=doc_labels, columns=doc_labels)
            print(df_sim_tfidf.to_string())

        elif choice == "7":
            if current_corpus == LAB_MANUAL_ML_CORPUS:
                sample_q = "neural computation network"
            elif current_corpus == CUSTOMER_SUPPORT_OPINION_CORPUS:
                sample_q = "ATM swallowed card cash not given"
            else:
                sample_q = "human interface response time"

            q_in = input(f"Enter query [default: '{sample_q}']: ").strip() or sample_q
            res = engine.query_semantic_search(q_in, top_k=5)

            print(f"\n[Search Results for: '{res['query']}']:")
            print(f"  Matched Vocabulary Tokens: {res['raw_matched_tokens']}")
            print(f"  Query LSI Coordinates: {res['query_lsi_vector']}")
            print(f"  Dominant Latent Concept: {res['dominant_topic']}")
            print(f"  Predicted Category: {res['predicted_category']}")
            print("\nRanked Documents by Semantic Similarity:")
            df_res = pd.DataFrame(res["ranked_documents"])[["rank", "doc_id", "title", "category", "lsi_similarity", "tfidf_similarity", "semantic_boost"]]
            print(df_res.to_string(index=False))

        elif choice == "8":
            max_valid = max(2, min(len(current_corpus) - 1, len(engine.vocabulary) - 1))
            k_in = input(f"Enter new k (2 to {max_valid}) [current: {engine.n_components}]: ").strip()
            if k_in.isdigit():
                new_k = max(2, min(int(k_in), max_valid))
                engine.fit(current_corpus, n_components=new_k)
                print(f"Re-fitted SVD with k={engine.n_components}.")


if __name__ == "__main__":
    run_cli_lsi()
