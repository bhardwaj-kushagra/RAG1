from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from .tokenize import split_sentences, tokenize_words


@dataclass
class FeatureArtifacts:
    vectorizer: TfidfVectorizer


def build_vectorizer(docs: List[str], ngram_range=(1, 2), max_features: Optional[int] = 20000) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        tokenizer=tokenize_words,
        preprocessor=None,
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=True,
        min_df=2,
    )
    vec.fit(docs)
    return vec


def doc_sentence_matrix(text: str, vectorizer: TfidfVectorizer) -> Tuple[List[str], sparse.spmatrix, np.ndarray]:
    """
    Returns: (sentences, sent_tfidf, doc_tfidf_mean)
    sent_tfidf: shape [S, F]
    doc_tfidf_mean: shape [F]
    """
    sents = split_sentences(text)
    if not sents:
        n_feat = len(vectorizer.get_feature_names_out())
        return [], sparse.csr_matrix((0, n_feat), dtype=np.float32), np.zeros(n_feat, dtype=np.float32)
    X = vectorizer.transform(sents)
    # Document centroid as mean of sentence vectors (use dense for simplicity)
    if sparse.issparse(X):
        X_dense = X.toarray()  # type: ignore[attr-defined]
        doc_vec = X_dense.mean(axis=0).ravel()
    else:
        doc_vec = np.asarray(X.mean(axis=0)).ravel()  # type: ignore[attr-defined]
    return sents, X, doc_vec


def sentence_heuristics(sentences: List[str]) -> np.ndarray:
    # Basic per-sentence features: position (normalized), length, is_title_case
    n = max(1, len(sentences))
    feats = []
    for i, s in enumerate(sentences):
        words = tokenize_words(s)
        length = len(words)
        pos = i / n
        is_title = float(s[:1].isupper())
        feats.append([pos, length, is_title])
    return np.asarray(feats, dtype=np.float32)


def cosine_sim_matrix(A, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of A and vector b.
    Supports scipy.sparse CSR/CSC for A.
    """
    # Ensure b is 1D numpy array
    b = np.asarray(b).ravel()
    if A.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if sparse.issparse(A):
        # Row norms for sparse
        row_norms = np.sqrt(A.multiply(A).sum(axis=1)).A1 + 1e-8
        num = A.dot(b)
        # num may be matrix if A is sparse; flatten
        if sparse.issparse(num):
            num = num.A1
        else:
            num = np.asarray(num).ravel()
    else:
        row_norms = np.linalg.norm(A, axis=1) + 1e-8
        num = (A @ b).ravel()
    b_norm = float(np.linalg.norm(b) + 1e-8)
    sims = num / (row_norms * b_norm)
    return sims.astype(np.float32)
