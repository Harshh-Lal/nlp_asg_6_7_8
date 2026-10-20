"""
ngram_engine.py - Core Engine for IntelliDesk Module 2 (Assignment 7: N-Gram Language Models)
Pure scratch implementation using collections.Counter, math, random, re (no external NLP libraries).

Provides the complete 8-option lab manual functionality:
1. Corpus Selection & Text Preprocessing
2. Unigram, Bigram, Trigram Generation & Frequency Tables
3. Maximum Likelihood Estimation (MLE) N-Gram Probabilities
4. Laplace (Add-1) Smoothing with Exact Numerator (Count+1) & Denominator (Context+V)
5. Sentence Probability Calculation (Unigram / Bigram / Trigram, Smoothed vs Unsmoothed)
6. Perplexity Evaluation: PP(W) = exp(-1/N * sum(log P(w_i | context)))
7. Trigram Next-Word Autocomplete Predictor
8. Weighted Random Bigram / Trigram Sentence Generator
"""

import math
import random
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional

# ==============================================================================
# CORPORA DEFINITIONS
# ==============================================================================

# 1. Default Lab Manual NLP Corpus (Classic foundational sentences)
DEFAULT_LAB_CORPUS: List[str] = [
    "natural language processing is an exciting field of artificial intelligence",
    "language models assign probabilities to sequences of words in natural text",
    "n gram models use the previous n minus one words to predict the next word",
    "unigram models assume word independence and ignore preceding context",
    "bigram models condition the probability of a word on the single previous word",
    "trigram models use the previous two words as conditioning context",
    "laplace add one smoothing avoids zero probability problem for unseen n grams",
    "perplexity measures how well a probability model predicts a sample text",
    "lower perplexity indicates better predictive performance of the language model",
    "artificial intelligence and machine learning empower modern language technology",
    "i am sam and sam i am",
    "i do not like green eggs and ham",
    "the quick brown fox jumps over the lazy dog",
    "statistical language modeling is essential for speech recognition and machine translation",
    "text generation with n gram language models samples words conditioned on context"
]

# 2. Support Ticket Corpus (Imported from Module 1 tfidf_classifier.py)
try:
    from tfidf_classifier import BANKING_DATA, STUDENT_DATA
    SUPPORT_TICKET_CORPUS: List[str] = [item["text"] for item in BANKING_DATA] + [item["text"] for item in STUDENT_DATA]
except ImportError:
    SUPPORT_TICKET_CORPUS: List[str] = [
        "atm machine did not dispense cash but amount deducted from account",
        "how do i generate a new atm green pin at the kiosk",
        "atm transaction failed due to cash dispenser shutter timeout",
        "please update my registered mobile number and email address in account",
        "block my lost credit card immediately to prevent unauthorized fraud",
        "enable international contactless transactions on my visa debit card",
        "what are the latest floating interest rates for home loan",
        "calculate monthly emi for housing loan over twenty years",
        "upi payment failed on google pay but money debited from bank account",
        "forgot upi security pin how to reset upi pin using debit card details",
        "check current balance and recent transactions on my savings account",
        "request a replacement for damaged chip on mastercard credit card",
        "when will end semester examination timetable and schedule be announced",
        "unable to download exam hall ticket admit card from student portal",
        "procedure to apply for revaluation and photocopy of midterm answer script",
        "campus placement drive eligibility criteria for computer science students",
        "how to get resume verified and approved by training placement cell",
        "how to pay college semester tuition fees through online net banking",
        "hostel room allotment list for second year students out on notice board",
        "complaint regarding poor quality and hygiene of hostel mess food",
        "room ceiling fan and bathroom water heater geyser not working repair request"
    ]


# ==============================================================================
# N-GRAM LANGUAGE MODEL CORE CLASS
# ==============================================================================

class NGramLanguageModel:
    """
    Complete Scratch N-Gram Language Model implementing Unigrams, Bigrams, Trigrams,
    MLE Estimation, Laplace (Add-1) Smoothing, Sentence Probability, Perplexity,
    Autocomplete Prediction, and Weighted Random Text Generation.
    """

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, corpus_sentences: Optional[List[str]] = None):
        self.raw_sentences: List[str] = []
        self.tokenized_sentences: List[List[str]] = []
        self.vocabulary: List[str] = []
        self.vocab_set: set = set()
        self.vocab_size: int = 0  # V (unique words + </s>)
        self.total_word_tokens: int = 0  # Total word tokens N

        # Frequency counters
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()

        # Context lookups for fast conditional querying
        self.bigram_context_counts: Counter = Counter()      # C(w1)
        self.trigram_context_counts: Counter = Counter()     # C(w1, w2)

        # Forward transition mappings: context -> Counter(next_word: count)
        self.bigram_transitions: Dict[str, Counter] = defaultdict(Counter)
        self.trigram_transitions: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

        if corpus_sentences:
            self.train(corpus_sentences)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Pure regex tokenizer converting text to lowercase alphanumeric tokens.
        No external libraries required.
        """
        # Match alphanumeric words or hyphenated words
        cleaned = text.lower().replace("-", " ")
        tokens = re.findall(r"\b[a-z0-9]+\b", cleaned)
        return tokens

    def train(self, corpus_sentences: List[str]) -> None:
        """
        Train N-gram models on the given corpus.
        Builds unigram, bigram, and trigram counters with proper boundary padding.
        """
        self.raw_sentences = corpus_sentences
        self.tokenized_sentences = []
        
        # Reset counters
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.bigram_context_counts.clear()
        self.trigram_context_counts.clear()
        self.bigram_transitions.clear()
        self.trigram_transitions.clear()

        word_types = set()
        total_tokens = 0

        for sent in corpus_sentences:
            tokens = self.tokenize(sent)
            if not tokens:
                continue
            self.tokenized_sentences.append(tokens)

            for tok in tokens:
                word_types.add(tok)
                total_tokens += 1
                self.unigram_counts[tok] += 1

            # --- BIGRAM PADDING & EXTRACTION ---
            # Bigram padded: [<s>, w1, w2, ..., wn, </s>]
            padded_bi = [self.START_TOKEN] + tokens + [self.END_TOKEN]
            for i in range(len(padded_bi) - 1):
                w1, w2 = padded_bi[i], padded_bi[i + 1]
                self.bigram_counts[(w1, w2)] += 1
                self.bigram_context_counts[w1] += 1
                self.bigram_transitions[w1][w2] += 1

            # --- TRIGRAM PADDING & EXTRACTION ---
            # Trigram padded: [<s>, <s>, w1, w2, ..., wn, </s>]
            padded_tri = [self.START_TOKEN, self.START_TOKEN] + tokens + [self.END_TOKEN]
            for i in range(len(padded_tri) - 2):
                w1, w2, w3 = padded_tri[i], padded_tri[i + 1], padded_tri[i + 2]
                self.trigram_counts[(w1, w2, w3)] += 1
                self.trigram_context_counts[(w1, w2)] += 1
                self.trigram_transitions[(w1, w2)][w3] += 1

        # Vocabulary V includes all word types plus the end of sentence token </s>
        # <s> is a start boundary condition, not a target vocabulary word to generate
        word_types.add(self.END_TOKEN)
        self.vocabulary = sorted(list(word_types))
        self.vocab_set = word_types
        self.vocab_size = len(self.vocabulary)
        self.total_word_tokens = total_tokens

    # ==========================================================================
    # MENU OPTION 1: CORPUS SUMMARY
    # ==========================================================================
    def get_corpus_summary(self) -> Dict[str, Any]:
        """Option 1: Return key statistical summary of the loaded training corpus."""
        return {
            "sentence_count": len(self.tokenized_sentences),
            "total_word_tokens": self.total_word_tokens,
            "vocabulary_size": self.vocab_size,
            "unique_unigrams": len(self.unigram_counts),
            "unique_bigrams": len(self.bigram_counts),
            "unique_trigrams": len(self.trigram_counts),
            "sample_sentences": self.raw_sentences[:4]
        }

    # ==========================================================================
    # MENU OPTION 2: N-GRAM FREQUENCY TABLES
    # ==========================================================================
    def get_ngram_frequency_table(self, n: int = 1, top_k: int = 25) -> List[Dict[str, Any]]:
        """
        Option 2: Generate frequency table for Unigrams (n=1), Bigrams (n=2), or Trigrams (n=3).
        """
        rows = []
        if n == 1:
            total = sum(self.unigram_counts.values())
            for rank, (word, count) in enumerate(self.unigram_counts.most_common(top_k), 1):
                rows.append({
                    "Rank": rank,
                    "Unigram": word,
                    "Count": count,
                    "Relative Frequency": round(count / total, 5) if total > 0 else 0
                })
        elif n == 2:
            total = sum(self.bigram_counts.values())
            for rank, (bg, count) in enumerate(self.bigram_counts.most_common(top_k), 1):
                rows.append({
                    "Rank": rank,
                    "Bigram": f"{bg[0]} -> {bg[1]}",
                    "Context (w1)": bg[0],
                    "Target (w2)": bg[1],
                    "Count": count,
                    "Relative Frequency": round(count / total, 5) if total > 0 else 0
                })
        elif n == 3:
            total = sum(self.trigram_counts.values())
            for rank, (tg, count) in enumerate(self.trigram_counts.most_common(top_k), 1):
                rows.append({
                    "Rank": rank,
                    "Trigram": f"{tg[0]} {tg[1]} -> {tg[2]}",
                    "Context (w1, w2)": f"{tg[0]} {tg[1]}",
                    "Target (w3)": tg[2],
                    "Count": count,
                    "Relative Frequency": round(count / total, 5) if total > 0 else 0
                })
        return rows

    # ==========================================================================
    # MENU OPTION 3: MLE PROBABILITY CALCULATOR (UNSMOOTHED)
    # ==========================================================================
    def calculate_mle_probability(self, target_word: str, context: Tuple[str, ...] = ()) -> Dict[str, Any]:
        """
        Option 3: Maximum Likelihood Estimation P_MLE(w | context) without smoothing.
        - Unigram (no context): C(w) / N
        - Bigram (context of 1): C(w1, w2) / C(w1)
        - Trigram (context of 2): C(w1, w2, w3) / C(w1, w2)
        """
        target_word = target_word.lower()
        context = tuple(c.lower() for c in context)

        if len(context) == 0:
            # Unigram MLE
            count = self.unigram_counts.get(target_word, 0)
            denom = self.total_word_tokens
            prob = count / denom if denom > 0 else 0.0
            return {
                "model": "Unigram (MLE)",
                "target_word": target_word,
                "context": "None",
                "count": count,
                "denominator": denom,
                "probability": prob,
                "formula": rf"P_{{\text{{MLE}}}}('{target_word}') = \frac{{C('{target_word}')}}{{N}} = \frac{{{count}}}{{{denom}}} = {prob:.6f}"
            }
        elif len(context) == 1:
            # Bigram MLE
            w1 = context[0]
            bg = (w1, target_word)
            count = self.bigram_counts.get(bg, 0)
            denom = self.bigram_context_counts.get(w1, 0)
            prob = count / denom if denom > 0 else 0.0
            return {
                "model": "Bigram (MLE)",
                "target_word": target_word,
                "context": w1,
                "count": count,
                "denominator": denom,
                "probability": prob,
                "formula": rf"P_{{\text{{MLE}}}}('{target_word}' \mid '{w1}') = \frac{{C('{w1}', '{target_word}')}}{{C('{w1}')}} = \frac{{{count}}}{{{denom}}} = {prob:.6f}"
            }
        elif len(context) >= 2:
            # Trigram MLE
            w1, w2 = context[-2], context[-1]
            tg = (w1, w2, target_word)
            count = self.trigram_counts.get(tg, 0)
            denom = self.trigram_context_counts.get((w1, w2), 0)
            prob = count / denom if denom > 0 else 0.0
            return {
                "model": "Trigram (MLE)",
                "target_word": target_word,
                "context": f"{w1} {w2}",
                "count": count,
                "denominator": denom,
                "probability": prob,
                "formula": rf"P_{{\text{{MLE}}}}('{target_word}' \mid '{w1}', '{w2}') = \frac{{C('{w1}', '{w2}', '{target_word}')}}{{C('{w1}', '{w2}')}} = \frac{{{count}}}{{{denom}}} = {prob:.6f}"
            }
        return {}

    # ==========================================================================
    # MENU OPTION 4: LAPLACE (ADD-1) SMOOTHING CALCULATOR
    # ==========================================================================
    def calculate_laplace_probability(self, target_word: str, context: Tuple[str, ...] = ()) -> Dict[str, Any]:
        """
        Option 4: Laplace (Add-1) Smoothed Probability calculation showing exact:
        - Numerator: Count + 1
        - Denominator: Context_Count + V
        """
        target_word = target_word.lower()
        context = tuple(c.lower() for c in context)
        V = self.vocab_size

        if len(context) == 0:
            # Unigram Laplace
            count = self.unigram_counts.get(target_word, 0)
            N = self.total_word_tokens
            numerator = count + 1
            denominator = N + V
            prob = numerator / denominator if denominator > 0 else 0.0
            return {
                "model": "Unigram (Laplace Add-1)",
                "target_word": target_word,
                "context": "None",
                "count": count,
                "context_count": N,
                "vocab_size": V,
                "numerator": numerator,
                "denominator": denominator,
                "probability": prob,
                "formula_latex": rf"P_{{\text{{Laplace}}}}('{target_word}') = \frac{{C('{target_word}') + 1}}{{N + V}} = \frac{{{count} + 1}}{{{N} + {V}}} = \frac{{{numerator}}}{{{denominator}}} = {prob:.6f}",
                "arithmetic": f"({count} + 1) / ({N} + {V}) = {numerator} / {denominator} = {prob:.6f}"
            }
        elif len(context) == 1:
            # Bigram Laplace
            w1 = context[0]
            bg = (w1, target_word)
            count = self.bigram_counts.get(bg, 0)
            context_count = self.bigram_context_counts.get(w1, 0)
            numerator = count + 1
            denominator = context_count + V
            prob = numerator / denominator if denominator > 0 else 0.0
            return {
                "model": "Bigram (Laplace Add-1)",
                "target_word": target_word,
                "context": w1,
                "count": count,
                "context_count": context_count,
                "vocab_size": V,
                "numerator": numerator,
                "denominator": denominator,
                "probability": prob,
                "formula_latex": rf"P_{{\text{{Laplace}}}}('{target_word}' \mid '{w1}') = \frac{{C('{w1}', '{target_word}') + 1}}{{C('{w1}') + V}} = \frac{{{count} + 1}}{{{context_count} + {V}}} = \frac{{{numerator}}}{{{denominator}}} = {prob:.6f}",
                "arithmetic": f"({count} + 1) / ({context_count} + {V}) = {numerator} / {denominator} = {prob:.6f}"
            }
        else:
            # Trigram Laplace
            w1, w2 = context[-2], context[-1]
            tg = (w1, w2, target_word)
            count = self.trigram_counts.get(tg, 0)
            context_count = self.trigram_context_counts.get((w1, w2), 0)
            numerator = count + 1
            denominator = context_count + V
            prob = numerator / denominator if denominator > 0 else 0.0
            return {
                "model": "Trigram (Laplace Add-1)",
                "target_word": target_word,
                "context": f"{w1} {w2}",
                "count": count,
                "context_count": context_count,
                "vocab_size": V,
                "numerator": numerator,
                "denominator": denominator,
                "probability": prob,
                "formula_latex": rf"P_{{\text{{Laplace}}}}('{target_word}' \mid '{w1}', '{w2}') = \frac{{C('{w1}', '{w2}', '{target_word}') + 1}}{{C('{w1}', '{w2}') + V}} = \frac{{{count} + 1}}{{{context_count} + {V}}} = \frac{{{numerator}}}{{{denominator}}} = {prob:.6f}",
                "arithmetic": f"({count} + 1) / ({context_count} + {V}) = {numerator} / {denominator} = {prob:.6f}"
            }

    # ==========================================================================
    # MENU OPTION 5: SENTENCE PROBABILITY CALCULATOR
    # ==========================================================================
    def calculate_sentence_probability(
        self,
        sentence: str,
        model_type: str = "bigram",
        smoothed: bool = True
    ) -> Dict[str, Any]:
        """
        Option 5: Calculate total sentence probability P(W) and log probability
        using Chain Rule across Unigram, Bigram, or Trigram models (Smoothed vs Unsmoothed).
        """
        tokens = self.tokenize(sentence)
        if not tokens:
            return {"error": "Sentence has no valid tokens."}

        model_type = model_type.lower()
        V = self.vocab_size
        step_rows = []
        log_prob_sum = 0.0
        sentence_prob = 1.0
        has_zero_prob = False

        if "uni" in model_type:
            # UNIGRAM MODEL: P(w1) * P(w2) * ... * P(wn)
            N = self.total_word_tokens
            for tok in tokens:
                count = self.unigram_counts.get(tok, 0)
                if smoothed:
                    p = (count + 1) / (N + V)
                    num_str = f"{count} + 1"
                    den_str = f"{N} + {V}"
                else:
                    p = count / N if N > 0 else 0.0
                    num_str = f"{count}"
                    den_str = f"{N}"

                if p > 0:
                    lp = math.log(p)
                    log_prob_sum += lp
                    sentence_prob *= p
                else:
                    lp = -float("inf")
                    sentence_prob = 0.0
                    has_zero_prob = True

                step_rows.append({
                    "Step": len(step_rows) + 1,
                    "Target Word (wi)": tok,
                    "Context": "None",
                    "Count": count,
                    "Denominator": N if not smoothed else N + V,
                    "Fraction": f"{num_str} / {den_str}",
                    "P(wi | Context)": p,
                    "log P(wi)": round(lp, 5) if lp != -float("inf") else "-inf"
                })

        elif "bi" in model_type:
            # BIGRAM MODEL: [<s>, w1, w2, ..., wn, </s>]
            padded = [self.START_TOKEN] + tokens + [self.END_TOKEN]
            for i in range(len(padded) - 1):
                ctx, tok = padded[i], padded[i + 1]
                count = self.bigram_counts.get((ctx, tok), 0)
                ctx_count = self.bigram_context_counts.get(ctx, 0)

                if smoothed:
                    p = (count + 1) / (ctx_count + V)
                    num_str = f"{count} + 1"
                    den_str = f"{ctx_count} + {V}"
                else:
                    p = count / ctx_count if ctx_count > 0 else 0.0
                    num_str = f"{count}"
                    den_str = f"{ctx_count}"

                if p > 0:
                    lp = math.log(p)
                    log_prob_sum += lp
                    sentence_prob *= p
                else:
                    lp = -float("inf")
                    sentence_prob = 0.0
                    has_zero_prob = True

                step_rows.append({
                    "Step": len(step_rows) + 1,
                    "Target Word (wi)": tok,
                    "Context (wi-1)": ctx,
                    "Count": count,
                    "Context Count": ctx_count,
                    "Fraction": f"{num_str} / {den_str}",
                    "P(wi | Context)": p,
                    "log P(wi)": round(lp, 5) if lp != -float("inf") else "-inf"
                })

        else:
            # TRIGRAM MODEL: [<s>, <s>, w1, w2, ..., wn, </s>]
            padded = [self.START_TOKEN, self.START_TOKEN] + tokens + [self.END_TOKEN]
            for i in range(len(padded) - 2):
                w1, w2, tok = padded[i], padded[i + 1], padded[i + 2]
                count = self.trigram_counts.get((w1, w2, tok), 0)
                ctx_count = self.trigram_context_counts.get((w1, w2), 0)

                if smoothed:
                    p = (count + 1) / (ctx_count + V)
                    num_str = f"{count} + 1"
                    den_str = f"{ctx_count} + {V}"
                else:
                    p = count / ctx_count if ctx_count > 0 else 0.0
                    num_str = f"{count}"
                    den_str = f"{ctx_count}"

                if p > 0:
                    lp = math.log(p)
                    log_prob_sum += lp
                    sentence_prob *= p
                else:
                    lp = -float("inf")
                    sentence_prob = 0.0
                    has_zero_prob = True

                step_rows.append({
                    "Step": len(step_rows) + 1,
                    "Target Word (wi)": tok,
                    "Context (wi-2, wi-1)": f"{w1} {w2}",
                    "Count": count,
                    "Context Count": ctx_count,
                    "Fraction": f"{num_str} / {den_str}",
                    "P(wi | Context)": p,
                    "log P(wi)": round(lp, 5) if lp != -float("inf") else "-inf"
                })

        return {
            "sentence": sentence,
            "tokens": tokens,
            "token_count": len(tokens),
            "total_transitions": len(step_rows),
            "model_type": model_type.capitalize(),
            "smoothed": smoothed,
            "sentence_prob": sentence_prob,
            "log_prob_sum": log_prob_sum if not has_zero_prob else -float("inf"),
            "has_zero_prob": has_zero_prob,
            "steps": step_rows
        }

    # ==========================================================================
    # MENU OPTION 6: PERPLEXITY EVALUATION
    # ==========================================================================
    def calculate_perplexity(
        self,
        sentence: str,
        model_type: str = "bigram",
        smoothed: bool = True
    ) -> Dict[str, Any]:
        """
        Option 6: Perplexity evaluation using exact formula:
        PP(W) = exp( - (1 / N) * sum( log P(w_i | context) ) )
        where N is the number of predicted words in the sequence.
        """
        sent_eval = self.calculate_sentence_probability(sentence, model_type=model_type, smoothed=smoothed)
        if "error" in sent_eval:
            return sent_eval

        N = sent_eval["total_transitions"]
        log_sum = sent_eval["log_prob_sum"]
        has_zero = sent_eval["has_zero_prob"]

        if has_zero or log_sum == -float("inf"):
            perplexity = float("inf")
            cross_entropy = float("inf")
        else:
            # Cross-entropy / negative log likelihood per word
            cross_entropy = - (log_sum / N)
            try:
                perplexity = math.exp(cross_entropy)
            except OverflowError:
                perplexity = float("inf")

        return {
            "sentence": sentence,
            "model_type": model_type.capitalize(),
            "smoothed": smoothed,
            "N": N,
            "log_prob_sum": log_sum,
            "cross_entropy": cross_entropy,
            "perplexity": perplexity,
            "formula_latex": rf"PP(W) = \exp\left(-\frac{{1}}{{{N}}} \sum_{{i=1}}^{{{N}}} \ln P(w_i \mid \text{{context}})\right)",
            "steps": sent_eval["steps"]
        }

    # ==========================================================================
    # MENU OPTION 7: TRIGRAM NEXT-WORD AUTOCOMPLETE PREDICTOR
    # ==========================================================================
    def autocomplete_next_words(
        self,
        prompt: str,
        top_k: int = 5,
        use_laplace: bool = True
    ) -> Dict[str, Any]:
        """
        Option 7: Trigram (or Bigram fallback) Next-Word Autocomplete Predictor.
        Given prompt tokens, ranks candidate continuation words by conditional probability.
        """
        tokens = self.tokenize(prompt)
        V = self.vocab_size

        if len(tokens) >= 2:
            # Trigram context: last two tokens
            w1, w2 = tokens[-2], tokens[-1]
            context_tuple = (w1, w2)
            context_label = f"Trigram context: ('{w1}', '{w2}')"
            candidates_counter = self.trigram_transitions.get(context_tuple, Counter())
            ctx_count = self.trigram_context_counts.get(context_tuple, 0)
        elif len(tokens) == 1:
            # Bigram context: last single token
            w1 = tokens[-1]
            context_tuple = (w1,)
            context_label = f"Bigram fallback context: ('{w1}')"
            candidates_counter = self.bigram_transitions.get(w1, Counter())
            ctx_count = self.bigram_context_counts.get(w1, 0)
        else:
            # Unigram prompt fallback: start token
            w1 = self.START_TOKEN
            context_tuple = (w1,)
            context_label = f"Sentence start context: ('{w1}')"
            candidates_counter = self.bigram_transitions.get(w1, Counter())
            ctx_count = self.bigram_context_counts.get(w1, 0)

        # Build candidate list: words seen after this context + most frequent unigrams if zero
        candidate_words = set(candidates_counter.keys())
        if not candidate_words:
            # Unseen context: use most common unigrams as candidates
            candidate_words = set([w for w, _ in self.unigram_counts.most_common(top_k * 2)])

        predictions = []
        for word in candidate_words:
            count = candidates_counter.get(word, 0)
            if use_laplace:
                prob = (count + 1) / (ctx_count + V)
            else:
                prob = count / ctx_count if ctx_count > 0 else 0.0

            predictions.append({
                "Word": word,
                "Occurrence Count": count,
                "Probability": round(prob, 5),
                "Percentage (%)": round(prob * 100, 2)
            })

        # Sort descending by probability
        predictions.sort(key=lambda x: (x["Probability"], x["Occurrence Count"]), reverse=True)
        top_predictions = predictions[:top_k]

        return {
            "prompt": prompt,
            "context_label": context_label,
            "context_count": ctx_count,
            "vocab_size": V,
            "use_laplace": use_laplace,
            "predictions": top_predictions
        }

    # ==========================================================================
    # MENU OPTION 8: WEIGHTED RANDOM SENTENCE GENERATOR
    # ==========================================================================
    def generate_sentence(
        self,
        model_type: str = "trigram",
        max_tokens: int = 25,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Option 8: Weighted random Bigram or Trigram sentence generator
        sampling words conditioned on context until </s> or max_tokens reached.
        """
        if seed is not None:
            random.seed(seed)

        model_type = model_type.lower()
        generated_tokens = []
        trace_steps = []

        if "bi" in model_type:
            # BIGRAM GENERATOR
            current_word = self.START_TOKEN
            for step in range(max_tokens):
                transitions = self.bigram_transitions.get(current_word, None)
                if not transitions:
                    # Fallback to unigram random choice
                    words = list(self.unigram_counts.keys())
                    weights = list(self.unigram_counts.values())
                else:
                    words = list(transitions.keys())
                    weights = list(transitions.values())

                next_word = random.choices(words, weights=weights, k=1)[0]
                prob = transitions.get(next_word, 0) / sum(weights) if transitions else 0.0

                trace_steps.append({
                    "Step": step + 1,
                    "Context": current_word,
                    "Chosen Word": next_word,
                    "Transition Prob": round(prob, 4)
                })

                if next_word == self.END_TOKEN:
                    break

                generated_tokens.append(next_word)
                current_word = next_word

        else:
            # TRIGRAM GENERATOR
            w1, w2 = self.START_TOKEN, self.START_TOKEN
            for step in range(max_tokens):
                transitions = self.trigram_transitions.get((w1, w2), None)
                if not transitions:
                    # Fallback to bigram transitions on w2
                    bg_transitions = self.bigram_transitions.get(w2, None)
                    if bg_transitions:
                        words = list(bg_transitions.keys())
                        weights = list(bg_transitions.values())
                    else:
                        words = list(self.unigram_counts.keys())
                        weights = list(self.unigram_counts.values())
                else:
                    words = list(transitions.keys())
                    weights = list(transitions.values())

                next_word = random.choices(words, weights=weights, k=1)[0]
                prob = transitions.get(next_word, 0) / sum(weights) if transitions else 0.0

                trace_steps.append({
                    "Step": step + 1,
                    "Context": f"{w1} {w2}",
                    "Chosen Word": next_word,
                    "Transition Prob": round(prob, 4)
                })

                if next_word == self.END_TOKEN:
                    break

                generated_tokens.append(next_word)
                w1, w2 = w2, next_word

        sentence_str = " ".join(generated_tokens)
        formatted_sentence = sentence_str.capitalize() + "." if sentence_str else "Empty generation."

        return {
            "model_type": model_type.capitalize(),
            "generated_text": formatted_sentence,
            "token_count": len(generated_tokens),
            "trace_steps": trace_steps
        }


# ==============================================================================
# INTERACTIVE CLI RUNNER (Matching Exact 8-Option Lab Manual Menu)
# ==============================================================================
def run_cli_menu():
    """Terminal interactive 8-option menu from the lab manual."""
    print("=" * 70)
    print("  ASSIGNMENT 7: N-GRAM LANGUAGE MODELS & SMOOTHING ENGINE")
    print("=" * 70)

    # Default to lab corpus
    model = NGramLanguageModel(DEFAULT_LAB_CORPUS)
    current_corpus_name = "Default Lab Manual NLP Corpus"

    while True:
        print("\n" + "-" * 70)
        print(f"  ACTIVE CORPUS: {current_corpus_name}")
        print("  SELECT AN OPTION FROM LAB MANUAL MENU:")
        print("  1. Select / View Training Corpus")
        print("  2. Generate N-Grams (Unigram, Bigram, Trigram) & Frequency Tables")
        print("  3. Calculate Maximum Likelihood Estimation (MLE) Probabilities")
        print("  4. Calculate Laplace (Add-1) Smoothed Probability (Numerator/Denominator)")
        print("  5. Calculate Sentence Probability (Unigram/Bigram/Trigram, Smoothed vs Unsmoothed)")
        print("  6. Evaluate Sentence Perplexity PP(W)")
        print("  7. Trigram Next-Word Autocomplete Predictor")
        print("  8. Weighted Random Sentence Generator (Bigram/Trigram)")
        print("  0. Exit")
        print("-" * 70)

        choice = input("Enter choice (0-8): ").strip()
        if choice == "0":
            print("Exiting. Thank you!")
            break

        elif choice == "1":
            print("\n[OPTION 1] Corpus Selection:")
            print("  1. Default Lab Manual NLP Corpus (15 foundational NLP sentences)")
            print("  2. Support Ticket Domain Corpus (Banking & Student helpdesk)")
            sub = input("Choose (1 or 2): ").strip()
            if sub == "2":
                model = NGramLanguageModel(SUPPORT_TICKET_CORPUS)
                current_corpus_name = "Support Ticket Domain Corpus"
            else:
                model = NGramLanguageModel(DEFAULT_LAB_CORPUS)
                current_corpus_name = "Default Lab Manual NLP Corpus"
            summary = model.get_corpus_summary()
            print(f"Corpus loaded successfully: {summary['sentence_count']} sentences, {summary['total_word_tokens']} word tokens, {summary['vocabulary_size']} vocabulary size.")

        elif choice == "2":
            n_in = input("Enter N for N-gram table (1=Unigram, 2=Bigram, 3=Trigram): ").strip()
            n = int(n_in) if n_in in ["1", "2", "3"] else 2
            table = model.get_ngram_frequency_table(n=n, top_k=10)
            print(f"\nTop 10 Frequency Table (N={n}):")
            for row in table:
                print(row)

        elif choice == "3":
            tgt = input("Enter target word: ").strip()
            ctx_in = input("Enter context words separated by space (or press Enter for unigram): ").strip()
            ctx = tuple(ctx_in.split()) if ctx_in else ()
            res = model.calculate_mle_probability(tgt, ctx)
            print("\n[MLE Result]:", res)

        elif choice == "4":
            tgt = input("Enter target word: ").strip()
            ctx_in = input("Enter context words separated by space (or press Enter for unigram): ").strip()
            ctx = tuple(ctx_in.split()) if ctx_in else ()
            res = model.calculate_laplace_probability(tgt, ctx)
            print(f"\n[Laplace Add-1 Result]:")
            print(f"  Count: {res['count']}")
            print(f"  Context Count: {res['context_count']}")
            print(f"  Vocab Size (V): {res['vocab_size']}")
            print(f"  Numerator: {res['numerator']}")
            print(f"  Denominator: {res['denominator']}")
            print(f"  Probability: {res['probability']:.6f}")
            print(f"  Arithmetic: {res['arithmetic']}")

        elif choice == "5":
            sent = input("Enter sentence to evaluate: ").strip()
            m_type = input("Choose model (unigram / bigram / trigram) [default: bigram]: ").strip() or "bigram"
            smooth_in = input("Enable Laplace smoothing? (y/n) [default: y]: ").strip().lower() != "n"
            res = model.calculate_sentence_probability(sent, model_type=m_type, smoothed=smooth_in)
            print(f"\n[Sentence Probability Result]:")
            print(f"  P(W): {res['sentence_prob']}")
            print(f"  log P(W): {res['log_prob_sum']}")
            for step in res["steps"]:
                print(step)

        elif choice == "6":
            sent = input("Enter test sentence: ").strip()
            m_type = input("Choose model (bigram / trigram) [default: bigram]: ").strip() or "bigram"
            res = model.calculate_perplexity(sent, model_type=m_type, smoothed=True)
            print(f"\n[Perplexity Result]:")
            print(f"  Total Tokens (N): {res['N']}")
            print(f"  Log Probability Sum: {res['log_prob_sum']:.5f}")
            print(f"  Cross-Entropy: {res['cross_entropy']:.5f}")
            print(f"  Perplexity PP(W): {res['perplexity']:.4f}")

        elif choice == "7":
            prompt = input("Enter 1 or 2 word prompt: ").strip()
            res = model.autocomplete_next_words(prompt, top_k=5, use_laplace=True)
            print(f"\n[Autocomplete Predictions for '{prompt}'] ({res['context_label']}):")
            for p in res["predictions"]:
                print(f"  -> '{p['Word']}': Prob={p['Probability']:.4f} ({p['Percentage (%)']}%)")

        elif choice == "8":
            m_type = input("Generate using (bigram / trigram) [default: trigram]: ").strip() or "trigram"
            res = model.generate_sentence(model_type=m_type, max_tokens=20)
            print(f"\n[Generated Sentence ({res['model_type']})]:")
            print(f"  {res['generated_text']}")


if __name__ == "__main__":
    run_cli_menu()
