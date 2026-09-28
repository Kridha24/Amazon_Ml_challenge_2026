"""
Matching models and threshold tuning for Amazon ML Challenge 2026.
Includes:
1. Simple Similarity Baseline (weighted rule-based score).
2. Lightweight ML Classifier (HistGradientBoostingClassifier).
3. Macro F_0.5 Threshold Tuning on tuning split with frozen parameter export.
"""

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import compute_pair_features
from src.metrics import evaluate_predictions

class SimpleSimilarityBaseline:
    """
    Deterministic similarity baseline combining token sort ratio and address similarity.
    """
    def score_pair(self, s1_entity, cand_record):
        # cand_record: (id, name, addr, country)
        feats = compute_pair_features(s1_entity, cand_record)
        name_sort = feats[1]
        has_missing = feats[6]
        addr_lev = feats[8]
        num_conflict = feats[10]
        
        if num_conflict > 0.5:
            return 0.0
            
        if has_missing > 0.5:
            return name_sort
        else:
            return 0.70 * name_sort + 0.30 * addr_lev

class LightweightClassifier:
    """
    Gradient boosted decision trees classifier for candidate pair verification.
    """
    def __init__(self, max_iter=100, min_samples_leaf=20, random_state=42):
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.is_fitted = False
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, filepath):
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.is_fitted = True
        return self

def tune_threshold(score_pairs_fn, s1_entities, candidate_lists, candidate_records, gt_map, thresholds=None):
    """
    Tune matching threshold on tuning split using per-Source-1 macro F_0.5.
    Evaluates every Source 1 entity, including those with zero candidates.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.02)]
        
    # Pre-score all candidate pairs to avoid recomputation
    scored_candidates = {}
    for s1 in s1_entities:
        s1_id = s1["entity_id"]
        cands = candidate_lists.get(s1_id, [])
        scored_list = []
        for cid, _ in cands:
            cand_rec = candidate_records[cid]
            prob = score_pairs_fn(s1, cand_rec)
            scored_list.append((cid, prob))
        scored_candidates[s1_id] = scored_list
        
    best_f05 = -1.0
    best_tau = 0.50
    best_eval = None
    tuning_history = []
    
    for tau in thresholds:
        preds = {}
        for s1 in s1_entities:
            s1_id = s1["entity_id"]
            scored = scored_candidates.get(s1_id, [])
            preds[s1_id] = {cid for cid, p in scored if p >= tau}
            
        eval_res = evaluate_predictions(gt_map, preds, s1_entities)
        f05 = eval_res["macro_f05"]
        tuning_history.append({
            "threshold": tau,
            "macro_f05": f05,
            "macro_precision": eval_res["macro_precision"],
            "macro_recall": eval_res["macro_recall"],
            "singleton_accuracy": eval_res["singleton_accuracy"]
        })
        
        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau
            best_eval = eval_res
            
    return {
        "best_threshold": best_tau,
        "best_macro_f05": best_f05,
        "best_evaluation": best_eval,
        "tuning_history": tuning_history,
        "scored_candidates": scored_candidates
    }
