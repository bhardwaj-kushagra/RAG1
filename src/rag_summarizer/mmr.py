from typing import List, Sequence

import numpy as np


def mmr_select(scores: np.ndarray, sent_vectors: np.ndarray, k: int, lambda_div: float = 0.7) -> List[int]:
    """
    Maximal Marginal Relevance selection.
    scores: relevance scores per sentence (higher is better)
    sent_vectors: [S, F] vectors to compute diversity via cosine
    k: number of sentences to pick
    Returns indices of selected sentences in original order of appearance.
    """
    S = len(scores)
    if S == 0 or k <= 0:
        return []
    selected = []
    candidates = list(range(S))
    norms = np.linalg.norm(sent_vectors, axis=1) + 1e-8
    while candidates and len(selected) < k:
        best_c = None
        best_val = -1e9
        for c in candidates:
            if selected:
                # diversity penalty: max cosine sim with already selected
                sims = []
                for s in selected:
                    num = float(sent_vectors[c].dot(sent_vectors[s]))
                    denom = float(norms[c] * norms[s])
                    sims.append(num / denom)
                div_pen = max(sims)
            else:
                div_pen = 0.0
            val = lambda_div * scores[c] - (1 - lambda_div) * div_pen
            if val > best_val:
                best_val = val
                best_c = c
        selected.append(best_c)
        candidates.remove(best_c)
    # Keep original order for readability
    return sorted(selected)
