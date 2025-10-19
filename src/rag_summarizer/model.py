from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os

import numpy as np
from sklearn.linear_model import Ridge
from joblib import dump, load
from rouge_score import rouge_scorer
from scipy import sparse

from .features import build_vectorizer, doc_sentence_matrix, sentence_heuristics, cosine_sim_matrix
from .mmr import mmr_select


@dataclass
class ModelArtifacts:
    vectorizer_path: str
    regressor_path: str


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _save_artifacts(vectorizer, regressor, out_dir: str) -> ModelArtifacts:
    os.makedirs(out_dir, exist_ok=True)
    vec_p = os.path.join(out_dir, "vectorizer.joblib")
    reg_p = os.path.join(out_dir, "regressor.joblib")
    dump(vectorizer, vec_p)
    dump(regressor, reg_p)
    return ModelArtifacts(vec_p, reg_p)


def _load_artifacts(model_dir: str):
    vec_p = os.path.join(model_dir, "vectorizer.joblib")
    reg_p = os.path.join(model_dir, "regressor.joblib")
    vectorizer = load(vec_p)
    regressor = load(reg_p)
    return vectorizer, regressor


def _reference_labels(sentences: List[str], reference: str) -> np.ndarray:
    # Compute ROUGE-1 F1 of each single sentence vs the reference summary as a soft supervisory signal
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    labels = []
    for s in sentences:
        score = scorer.score(reference, s)["rouge1"].fmeasure
        labels.append(score)
    if not labels:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(labels, dtype=np.float32)


def _build_features_for_sentences(sentences: List[str], sent_tfidf, doc_vec) -> np.ndarray:
    sims = cosine_sim_matrix(sent_tfidf, doc_vec)[:, None]
    heur = sentence_heuristics(sentences)
    # Combine dense features with a projection from TF-IDF: mean TF-IDF weight per sentence
    tfidf_means = np.asarray(sent_tfidf.mean(axis=1)).reshape(-1, 1)
    feats = np.concatenate([sims, heur, tfidf_means], axis=1)
    return feats.astype(np.float32)


def train_model(docs_dir: str, targets_dir: str, out_dir: str, alpha: float = 1.0) -> ModelArtifacts:
    # 1) Build vectorizer from all documents
    doc_paths = sorted([os.path.join(docs_dir, p) for p in os.listdir(docs_dir) if p.lower().endswith('.txt')])
    tgt_paths = {os.path.basename(p): os.path.join(targets_dir, os.path.basename(p)) for p in doc_paths}
    docs = [_load_text(p) for p in doc_paths]
    vectorizer = build_vectorizer(docs)

    X_list = []
    y_list = []
    for doc_path in doc_paths:
        doc_txt = _load_text(doc_path)
        ref_path = tgt_paths.get(os.path.basename(doc_path))
        if not ref_path or not os.path.exists(ref_path):
            continue
        ref_txt = _load_text(ref_path)
        sents, sent_mat, doc_vec = doc_sentence_matrix(doc_txt, vectorizer)
        labels = _reference_labels(sents, ref_txt)
        if len(sents) == 0 or labels.size == 0:
            continue
        feats = _build_features_for_sentences(sents, sent_mat, doc_vec)
        X_list.append(feats)
        y_list.append(labels)

    if not X_list:
        raise RuntimeError("No training instances found. Ensure docs and targets share filenames and are non-empty.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    reg = Ridge(alpha=alpha)
    reg.fit(X, y)

    arts = _save_artifacts(vectorizer, reg, out_dir)
    return arts


def summarize_text(text: str, vectorizer, regressor, max_words: Optional[int] = None, ratio: Optional[float] = None, lambda_div: float = 0.7) -> str:
    sents, sent_mat, doc_vec = doc_sentence_matrix(text, vectorizer)
    if not sents:
        return ""
    feats = _build_features_for_sentences(sents, sent_mat, doc_vec)
    scores = regressor.predict(feats)

    # Budget
    words = [len(s.split()) for s in sents]
    total_words = sum(words)
    if ratio is not None and (max_words is None):
        max_words = max(1, int(total_words * ratio))
    if max_words is None:
        max_words = max(1, int(total_words * 0.2))

    # Estimate k greedily: keep selecting until hitting budget
    order = np.argsort(-scores)
    pick_mask = np.zeros(len(sents), dtype=bool)
    word_count = 0
    for idx in order:
        if word_count + words[idx] <= max_words:
            pick_mask[idx] = True
            word_count += words[idx]
    # If nothing selected due to short budget, pick the top sentence
    if not pick_mask.any():
        pick_mask[np.argmax(scores)] = True

    # Use MMR among the candidates to ensure diversity
    cand_idx = np.where(pick_mask)[0]
    k = len(cand_idx)
    if k <= 1:
        selected = list(cand_idx)
    else:
        if sparse.issparse(sent_mat):
            csr = sent_mat.tocsr()  # type: ignore[attr-defined]
            sub = csr[cand_idx, :]  # type: ignore[index]
            sub_dense = np.asarray(sub.todense())  # type: ignore[attr-defined]
        else:
            sub_dense = np.asarray(sent_mat[cand_idx])  # type: ignore[index]
        selected = mmr_select(scores[cand_idx], sub_dense, k=k, lambda_div=lambda_div)
        selected = [int(cand_idx[i]) for i in selected]

    # Compose summary in original order
    summary = " ".join([sents[i] for i in sorted(selected)])
    return summary


def summarize_file(path: str, model_dir: Optional[str], words: Optional[int], ratio: Optional[float], lambda_div: float = 0.7) -> str:
    text = _load_text(path)
    if model_dir and os.path.exists(model_dir):
        vectorizer, regressor = _load_artifacts(model_dir)
    else:
        # Fallback: quick unsupervised setup - fit vectorizer on this single doc and use uniform weights
        vectorizer = build_vectorizer([text])
        from sklearn.linear_model import Ridge
        regressor = Ridge(alpha=1.0)
        # Train a dummy regressor to center around cosine/doc similarity + heuristics using pseudo labels (lead bias)
        sents, sent_mat, doc_vec = doc_sentence_matrix(text, vectorizer)
        feats = _build_features_for_sentences(sents, sent_mat, doc_vec)
        if len(sents) == 0:
            return ""
        # Lead-3 style soft labels
        y = np.linspace(1.0, 0.1, num=len(sents), dtype=np.float32)
        regressor.fit(feats, y)
    return summarize_text(text, vectorizer, regressor, max_words=words, ratio=ratio, lambda_div=lambda_div)
