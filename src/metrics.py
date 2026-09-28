"""
Evaluation metrics for Amazon ML Challenge 2026: Business Entity Resolution.
Implements the exact macro-averaged F_0.5 score per Source 1 entity,
accurately handling singletons, empty match lists, and country breakdowns.
"""

def evaluate_predictions(ground_truth_map, predictions_map, s1_entities_metadata):
    """
    Compute macro-averaged F_0.5 and auxiliary metrics across all evaluated Source 1 entities.
    
    Args:
        ground_truth_map: dict mapping s1_id -> set of true matched entity IDs
        predictions_map: dict mapping s1_id -> set of predicted matched entity IDs
        s1_entities_metadata: list or dict of S1 entities with 'entity_id' and 'country'
        
    Returns:
        dict containing:
            macro_f05: Overall macro-averaged F_0.5
            macro_precision: Overall macro-averaged precision
            macro_recall: Overall macro-averaged recall
            singleton_accuracy: Accuracy on true singletons (target has 0 matches)
            singleton_total: Total number of true singletons
            country_metrics: Breakdown of metrics by country (US, India, etc.)
            per_entity_scores: list of dicts with individual entity scores
    """
    scores = []
    precisions = []
    recalls = []
    
    singletons_correct = 0
    singletons_total = 0
    
    country_data = {}
    per_entity_scores = []
    
    for s1_meta in s1_entities_metadata:
        s1_id = s1_meta["entity_id"]
        country = s1_meta.get("country", "Unknown")
        
        if country not in country_data:
            country_data[country] = {
                "scores": [],
                "singletons_correct": 0,
                "singletons_total": 0,
                "total": 0
            }
        country_data[country]["total"] += 1
        
        true_matches = set(ground_truth_map.get(s1_id, set()))
        pred_matches = set(predictions_map.get(s1_id, set()))
        
        if len(true_matches) == 0:
            # Singleton entity
            singletons_total += 1
            country_data[country]["singletons_total"] += 1
            
            if len(pred_matches) == 0:
                f05 = 1.0
                p = 1.0
                r = 1.0
                singletons_correct += 1
                country_data[country]["singletons_correct"] += 1
            else:
                f05 = 0.0
                p = 0.0
                r = 0.0
        else:
            # Entity with true matches
            if len(pred_matches) == 0:
                f05 = 0.0
                p = 0.0
                r = 0.0
            else:
                tp = len(true_matches & pred_matches)
                if tp == 0:
                    f05 = 0.0
                    p = 0.0
                    r = 0.0
                else:
                    p = tp / len(pred_matches)
                    r = tp / len(true_matches)
                    denom = 0.25 * p + r
                    f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
                    
        scores.append(f05)
        precisions.append(p)
        recalls.append(r)
        country_data[country]["scores"].append(f05)
        
        per_entity_scores.append({
            "entity_id": s1_id,
            "country": country,
            "true_count": len(true_matches),
            "pred_count": len(pred_matches),
            "f05": round(f05, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "is_singleton": len(true_matches) == 0,
            "correct_match": f05 > 0.999
        })
        
    n = len(scores)
    macro_f05 = sum(scores) / n if n > 0 else 0.0
    macro_precision = sum(precisions) / n if n > 0 else 0.0
    macro_recall = sum(recalls) / n if n > 0 else 0.0
    singleton_acc = singletons_correct / singletons_total if singletons_total > 0 else 0.0
    
    country_breakdown = {}
    for c, c_info in country_data.items():
        c_scores = c_info["scores"]
        c_macro = sum(c_scores) / len(c_scores) if c_scores else 0.0
        c_sing_acc = (
            c_info["singletons_correct"] / c_info["singletons_total"]
            if c_info["singletons_total"] > 0 else 0.0
        )
        country_breakdown[c] = {
            "total_entities": c_info["total"],
            "macro_f05": round(c_macro, 4),
            "singleton_accuracy": round(c_sing_acc, 4),
            "singletons_total": c_info["singletons_total"]
        }
        
    return {
        "total_evaluated": n,
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "singleton_accuracy": round(singleton_acc, 4),
        "singleton_total": singletons_total,
        "singleton_correct": singletons_correct,
        "country_breakdown": country_breakdown,
        "per_entity_scores": per_entity_scores
    }
