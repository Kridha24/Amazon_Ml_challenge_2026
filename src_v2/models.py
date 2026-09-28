"""
Matching models and cached threshold tuning for V2.
Batches feature extraction and caches prediction probabilities for sub-second threshold tuning.
"""

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from src_v2.features import compute_pair_features_v2
from src_v2.metrics import evaluate_predictions

class LightweightClassifierV2:
    """
    Gradient boosted decision trees classifier (V2).
    """
    def __init__(self, max_iter=150, min_samples_leaf=20, random_state=42):
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

def batch_score_candidates(clf, s1_entities, candidate_lists, candidate_records):
    """
    Precomputes features and runs batched classifier inference across all candidate pairs.
    Returns cached mapping: {s1_id: [(cid, prob), ...]}.
    Executes in a fraction of a second via vectorized matrix operations.
    """
    pair_metadata = []
    feature_rows = []
    
    for s1 in s1_entities:
        s1_id = s1["entity_id"]
        cands = candidate_lists.get(s1_id, [])
        for cid, _ in cands:
            cand_rec = candidate_records[cid]
            feat = compute_pair_features_v2(s1, cand_rec)
            feature_rows.append(feat)
            pair_metadata.append((s1_id, cid))
            
    if not feature_rows:
        return {s1["entity_id"]: [] for s1 in s1_entities}
        
    X_matrix = np.array(feature_rows, dtype=np.float32)
    probabilities = clf.predict_proba(X_matrix)
    
    scored_candidates = {s1["entity_id"]: [] for s1 in s1_entities}
    for (s1_id, cid), prob in zip(pair_metadata, probabilities):
        scored_candidates[s1_id].append((cid, float(prob)))
        
    return scored_candidates

def tune_threshold_cached(scored_candidates, s1_entities, gt_map, thresholds=None):
    """
    Fast threshold sweep on cached prediction probabilities.
    Tunes threshold on tuning split using per-Source-1 macro F_0.5.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.02)]
        
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
        "tuning_history": tuning_history
    }
