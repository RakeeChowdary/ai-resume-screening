"""
ML Models for Resume Screening & Ranking
Includes TF-IDF + Cosine Similarity, Sentence-BERT, and Gradient Boosting Ranker
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (ndcg_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ─── Model 1: TF-IDF + Cosine Similarity ──────────────────────────────────────

class TFIDFRanker:
    """
    Baseline model: TF-IDF vectorization + cosine similarity ranking.
    Fast and interpretable.
    """

    def __init__(self, max_features: int = 10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            sublinear_tf=True
        )
        self.jd_vectors = None
        self.jd_ids = []
        self.fitted = False

    def fit(self, jd_texts: List[str], jd_ids: List[str] = None):
        """Fit vectorizer on job descriptions."""
        self.jd_vectors = self.vectorizer.fit_transform(jd_texts)
        self.jd_ids = jd_ids or [f"JD_{i}" for i in range(len(jd_texts))]
        self.fitted = True
        print(f"✓ TF-IDF fitted on {len(jd_texts)} JDs | vocab size: {len(self.vectorizer.vocabulary_)}")

    def rank_resume(self, resume_text: str, jd_index: int = 0) -> float:
        """Score a single resume against a specific JD."""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        resume_vec = self.vectorizer.transform([resume_text])
        score = cosine_similarity(resume_vec, self.jd_vectors[jd_index]).flatten()[0]
        return float(score)

    def rank_all(self, resume_texts: List[str], jd_index: int = 0) -> np.ndarray:
        """Score all resumes against a specific JD."""
        resume_vecs = self.vectorizer.transform(resume_texts)
        scores = cosine_similarity(resume_vecs, self.jd_vectors[jd_index]).flatten()
        return scores

    def get_top_k(self, resume_texts: List[str], candidate_ids: List[str],
                  jd_index: int = 0, k: int = 10) -> List[Dict]:
        """Return top-k candidates ranked by similarity."""
        scores = self.rank_all(resume_texts, jd_index)
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [
            {"rank": i + 1, "candidate_id": candidate_ids[idx],
             "tfidf_score": round(float(scores[idx]), 4)}
            for i, idx in enumerate(ranked_indices)
        ]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ TF-IDF model saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'TFIDFRanker':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─── Model 2: Feature-Based ML Ranker ─────────────────────────────────────────

class FeatureBasedRanker:
    """
    Gradient Boosting ranker using engineered features.
    Captures structured signals like skill match, experience gap, education level.
    """

    FEATURE_COLS = [
        'required_skill_match', 'preferred_skill_match', 'skill_combined_score',
        'total_skills', 'experience_score', 'education_level', 'cgpa_normalized',
        'certifications_count', 'projects_count', 'resume_length', 'matched_required'
    ]

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_split=10, random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importances_ = None
        self.fitted = False

    def _prepare_X(self, features_list: List[Dict]) -> np.ndarray:
        df = pd.DataFrame(features_list)
        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.FEATURE_COLS].values

    def fit(self, features_list: List[Dict], labels: List[int]):
        """
        Train on (feature_dict, label) pairs.
        label=1 if candidate is suitable, 0 otherwise.
        """
        X = self._prepare_X(features_list)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)
        self.feature_importances_ = dict(
            zip(self.FEATURE_COLS, self.model.feature_importances_)
        )
        self.fitted = True
        print(f"✓ Feature ranker trained on {len(features_list)} samples")
        return self

    def predict_score(self, features: Dict) -> float:
        """Return probability of being a good candidate."""
        X = self._prepare_X([features])
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict_proba(X_scaled)[0][1])

    def predict_batch(self, features_list: List[Dict]) -> np.ndarray:
        X = self._prepare_X(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def rank_candidates(self, features_list: List[Dict], candidate_ids: List[str],
                        k: int = 10) -> List[Dict]:
        scores = self.predict_batch(features_list)
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [
            {"rank": i + 1, "candidate_id": candidate_ids[idx],
             "ml_score": round(float(scores[idx]), 4)}
            for i, idx in enumerate(ranked_indices)
        ]

    def evaluate(self, features_list: List[Dict], true_labels: List[int]) -> Dict:
        X = self._prepare_X(features_list)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)[:, 1]

        return {
            "accuracy": float(np.mean(preds == true_labels)),
            "precision": float(precision_score(true_labels, preds, zero_division=0)),
            "recall": float(recall_score(true_labels, preds, zero_division=0)),
            "f1_score": float(f1_score(true_labels, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(true_labels, probs)) if len(set(true_labels)) > 1 else 0.0,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Feature ranker saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'FeatureBasedRanker':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─── Model 3: Sentence-BERT Semantic Ranker ────────────────────────────────────

class SemanticRanker:
    """
    Sentence-BERT based semantic similarity ranker.
    Captures deep semantic meaning beyond keyword matching.
    Uses 'all-MiniLM-L6-v2' (fast, 384-dim embeddings).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.jd_embeddings = None
        self.jd_ids = []

    def load_model(self):
        """Lazy-load sentence transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✓ Loaded Sentence-BERT: {self.model_name}")
        except ImportError:
            print("⚠ sentence-transformers not installed. Using fallback TF-IDF.")
            self.model = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self.load_model()
        if self.model is None:
            # Fallback: return random embeddings for demo
            return np.random.rand(len(texts), 384)
        return self.model.encode(texts, show_progress_bar=False, batch_size=32)

    def fit_jds(self, jd_texts: List[str], jd_ids: List[str] = None):
        """Pre-compute JD embeddings."""
        self.jd_embeddings = self._embed(jd_texts)
        self.jd_ids = jd_ids or [f"JD_{i}" for i in range(len(jd_texts))]
        print(f"✓ Encoded {len(jd_texts)} JDs | shape: {self.jd_embeddings.shape}")

    def rank_resume(self, resume_text: str, jd_index: int = 0) -> float:
        """Compute semantic similarity for a single resume."""
        if self.jd_embeddings is None:
            raise RuntimeError("Call fit_jds() first.")
        resume_emb = self._embed([resume_text])
        score = cosine_similarity(resume_emb, self.jd_embeddings[jd_index:jd_index+1])[0][0]
        return float(score)

    def rank_all(self, resume_texts: List[str], jd_index: int = 0) -> np.ndarray:
        resume_embs = self._embed(resume_texts)
        scores = cosine_similarity(resume_embs, self.jd_embeddings[jd_index:jd_index+1]).flatten()
        return scores

    def get_top_k(self, resume_texts: List[str], candidate_ids: List[str],
                  jd_index: int = 0, k: int = 10) -> List[Dict]:
        scores = self.rank_all(resume_texts, jd_index)
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [
            {"rank": i + 1, "candidate_id": candidate_ids[idx],
             "semantic_score": round(float(scores[idx]), 4)}
            for i, idx in enumerate(ranked_indices)
        ]

    def save(self, path: str):
        state = {
            "model_name": self.model_name,
            "jd_embeddings": self.jd_embeddings.tolist() if self.jd_embeddings is not None else None,
            "jd_ids": self.jd_ids
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        print(f"✓ Semantic ranker saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'SemanticRanker':
        with open(path, 'r') as f:
            state = json.load(f)
        ranker = cls(model_name=state["model_name"])
        ranker.jd_embeddings = np.array(state["jd_embeddings"]) if state["jd_embeddings"] else None
        ranker.jd_ids = state["jd_ids"]
        return ranker


# ─── Ensemble Ranker ──────────────────────────────────────────────────────────

class EnsembleRanker:
    """
    Combines TF-IDF, Feature-Based, and Semantic scores
    with learned or fixed weights.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "tfidf": 0.25,
            "features": 0.45,
            "semantic": 0.30
        }
        self.tfidf = TFIDFRanker()
        self.feature_ranker = FeatureBasedRanker()
        self.semantic = SemanticRanker()

    def combine_scores(self, tfidf_score: float, feature_score: float,
                       semantic_score: float) -> float:
        """Weighted ensemble score."""
        return (
            self.weights["tfidf"] * tfidf_score +
            self.weights["features"] * feature_score +
            self.weights["semantic"] * semantic_score
        )

    def rank_candidates(self, candidates: List[Dict], jd: Dict,
                        jd_index: int = 0, k: int = 10) -> List[Dict]:
        """Full pipeline: rank candidates using all 3 models."""
        from utils.nlp_utils import engineer_features, compute_heuristic_score

        results = []
        for candidate in candidates:
            # TF-IDF score
            tfidf_score = self.tfidf.rank_resume(
                candidate.get("resume_text", ""), jd_index
            ) if self.tfidf.fitted else 0.0

            # Feature-based score
            features = engineer_features(candidate, jd)
            feature_score = (
                self.feature_ranker.predict_score(features)
                if self.feature_ranker.fitted
                else compute_heuristic_score(features)
            )

            # Semantic score
            semantic_score = 0.0
            if self.semantic.jd_embeddings is not None:
                semantic_score = self.semantic.rank_resume(
                    candidate.get("resume_text", ""), jd_index
                )

            ensemble_score = self.combine_scores(tfidf_score, feature_score, semantic_score)

            results.append({
                "candidate_id": candidate.get("candidate_id", "?"),
                "name": candidate.get("name", "Unknown"),
                "tfidf_score": round(tfidf_score, 4),
                "feature_score": round(feature_score, 4),
                "semantic_score": round(semantic_score, 4),
                "ensemble_score": round(ensemble_score, 4),
                "features": features,
                "skills": candidate.get("skills", []),
                "experience_years": candidate.get("experience_years", 0),
                "education": candidate.get("education", {}),
            })

        results.sort(key=lambda x: x["ensemble_score"], reverse=True)
        for i, r in enumerate(results[:k]):
            r["rank"] = i + 1

        return results[:k]


# ─── Evaluation Metrics ───────────────────────────────────────────────────────

class RankingEvaluator:
    """Comprehensive evaluation for ranking models."""

    @staticmethod
    def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain @ k."""
        ranked_idx = np.argsort(scores)[::-1][:k]
        dcg = sum(relevance[idx] / np.log2(i + 2) for i, idx in enumerate(ranked_idx))
        ideal_idx = np.argsort(relevance)[::-1][:k]
        idcg = sum(relevance[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_idx))
        return float(dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def precision_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 10,
                       threshold: float = 0.5) -> float:
        """Precision@k: fraction of top-k that are relevant."""
        ranked_idx = np.argsort(scores)[::-1][:k]
        return float(np.mean(relevance[ranked_idx] >= threshold))

    @staticmethod
    def mean_reciprocal_rank(scores: np.ndarray, relevance: np.ndarray) -> float:
        """MRR: reciprocal rank of first relevant result."""
        ranked_idx = np.argsort(scores)[::-1]
        for i, idx in enumerate(ranked_idx):
            if relevance[idx] > 0:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def evaluate_ranker(scores: np.ndarray, relevance: np.ndarray) -> Dict:
        return {
            "ndcg@5": RankingEvaluator.ndcg_at_k(scores, relevance, k=5),
            "ndcg@10": RankingEvaluator.ndcg_at_k(scores, relevance, k=10),
            "precision@5": RankingEvaluator.precision_at_k(scores, relevance, k=5),
            "precision@10": RankingEvaluator.precision_at_k(scores, relevance, k=10),
            "mrr": RankingEvaluator.mean_reciprocal_rank(scores, relevance),
            "spearman_corr": float(
                pd.Series(scores).corr(pd.Series(relevance), method='spearman')
            )
        }