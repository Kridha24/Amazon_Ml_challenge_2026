#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution Baseline Experiment Driver
Executes an end-to-end, memory-bounded, reproducible experiment:
1. Stratified S1 sampling (5,000 train, 2,000 tune, 2,000 holdout) with complete clusters.
2. Candidate pool construction with true matches, hard distractors, and random non-matches.
3. Multi-pass inverted index blocking and candidate recall evaluation at K in [5, 10, 20, 30, 50, 100].
4. Feature extraction on candidate pairs (excluding entity IDs).
5. Simple Similarity Baseline evaluation.
6. Lightweight Classifier training (HistGradientBoosting).
7. Threshold tuning on Tuning split using macro F_0.5.
8. Unbiased evaluation on Holdout split with frozen settings.
9. Country and singleton breakdown, error analysis, stage runtimes, and peak memory logging.
"""

import os
import sys
import time
import json
import resource
import platform
import numpy as np
import joblib
from collections import Counter

# Ensure workspace root is in python path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.sampling import create_stratified_splits, extract_split_samples_and_pools
from src.blocking import CandidateRetriever, evaluate_candidate_recall
from src.features import compute_pair_features, FEATURE_NAMES
from src.models import SimpleSimilarityBaseline, LightweightClassifier, tune_threshold
from src.metrics import evaluate_predictions

DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
EXPERIMENTS_DIR = os.path.join(WORKSPACE_DIR, "experiments")
SPLITS_DIR = os.path.join(EXPERIMENTS_DIR, "splits")
MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "models")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

def get_peak_memory_mb():
    """Returns peak memory in MB on macOS or Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024 * 1024), 2)  # bytes on Darwin
    else:
        return round(rss / 1024, 2)  # KB on Linux

def main():
    total_start_time = time.time()
    runtimes = {}
    
    print("=" * 80)
    print(" Amazon ML Challenge 2026: Business Entity Resolution Baseline Experiment")
    print("=" * 80)
    print(f"System: {platform.system()} {platform.machine()} | Python: {platform.python_version()}")
    print(f"Working Directory: {WORKSPACE_DIR}")
    print(f"Initial RAM RSS: {get_peak_memory_mb()} MB\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Stratified Sampling of Disjoint S1 Entities
    # -------------------------------------------------------------------------
    print("[Step 1/7] Performing Stratified Sampling of S1 Entities...")
    t0 = time.time()
    gt_path = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(DATASET_DIR, "train", "train_source1.tsv")
    
    splits_s1, gt_map, strata_manifest = create_stratified_splits(
        gt_path=gt_path,
        s1_path=s1_path,
        target_counts={"train": 5000, "tune": 2000, "holdout": 2000},
        seed=42
    )
    runtimes["stratified_sampling_s1"] = round(time.time() - t0, 2)
    print(f"  Train S1: {len(splits_s1['train'])}, Tune S1: {len(splits_s1['tune'])}, Holdout S1: {len(splits_s1['holdout'])}")
    print(f"  Stratified sampling finished in {runtimes['stratified_sampling_s1']}s")
    
    # -------------------------------------------------------------------------
    # Step 2: Streaming Sample & Candidate Pool Extraction (with Distractors)
    # -------------------------------------------------------------------------
    print("\n[Step 2/7] Streaming Source Files to Extract Samples & Candidate Pools...")
    t0 = time.time()
    s1_data, pools, pool_summary = extract_split_samples_and_pools(
        data_dir=DATASET_DIR,
        splits_s1=splits_s1,
        gt_map=gt_map,
        seed=42
    )
    runtimes["streaming_pool_extraction"] = round(time.time() - t0, 2)
    
    for split in ("train", "tune", "holdout"):
        ps = pool_summary[split]
        print(f"  [{split.upper()}] S1 entities: {ps['s1_count']}")
        print(f"    S2 Pool: {ps['s2_pool_total']} (True: {ps['s2_composition'].get('true_matches', 0)}, Hard: {ps['s2_composition'].get('hard_distractors', 0)}, Random: {ps['s2_composition'].get('random_distractors', 0)})")
        print(f"    S3 Pool: {ps['s3_pool_total']} (True: {ps['s3_composition'].get('true_matches', 0)}, Hard: {ps['s3_composition'].get('hard_distractors', 0)}, Random: {ps['s3_composition'].get('random_distractors', 0)})")
        print(f"    Total Candidate Pool: {ps['total_candidate_pool']:,} records")
        
    # Save Split Manifests
    for split in ("train", "tune", "holdout"):
        manifest_path = os.path.join(SPLITS_DIR, f"manifest_{split}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "split": split,
                "s1_count": len(s1_data[split]),
                "s1_ids": [x["entity_id"] for x in s1_data[split]],
                "pool_summary": pool_summary[split]
            }, f, indent=2)
    print(f"  Split manifests saved to: {SPLITS_DIR}/")
    print(f"  Pool extraction finished in {runtimes['streaming_pool_extraction']}s | RAM RSS: {get_peak_memory_mb()} MB")
    
    # -------------------------------------------------------------------------
    # Step 3: Candidate Generation (Blocking) & Recall Evaluation across K
    # -------------------------------------------------------------------------
    print("\n[Step 3/7] Building Inverted Indices & Evaluating Candidate Recall...")
    t0 = time.time()
    retrievers = {}
    candidate_lists = {}
    recall_analyses = {}
    
    for split in ("train", "tune", "holdout"):
        retriever = CandidateRetriever()
        retriever.build_index(pools[split]["S2"], pools[split]["S3"])
        retrievers[split] = retriever
        
        # Retrieve candidates for all S1 entities in split
        cand_map = {}
        for s1 in s1_data[split]:
            s1_id = s1["entity_id"]
            cands = retriever.retrieve_candidates_for_query(s1, max_candidates=100)
            cand_map[s1_id] = cands
        candidate_lists[split] = cand_map
        
        # Evaluate candidate recall
        recall_res = evaluate_candidate_recall(cand_map, gt_map, k_limits=[5, 10, 20, 30, 50, 100])
        recall_analyses[split] = recall_res
        
    runtimes["blocking_and_candidate_generation"] = round(time.time() - t0, 2)
    
    print("  Candidate Recall Curves (Tune & Holdout splits):")
    for split in ("tune", "holdout"):
        rec_k = recall_analyses[split]["recall_by_k"]
        print(f"    [{split.upper()}] Non-singletons: {recall_analyses[split]['non_singleton_evaluated']}, True matches: {recall_analyses[split]['total_true_matches']:,}, Avg cands/S1: {recall_analyses[split]['average_candidates_per_entity']}")
        print(f"      R@5: {rec_k['recall_at_5']:.1%} | R@10: {rec_k['recall_at_10']:.1%} | R@20: {rec_k['recall_at_20']:.1%} | R@30: {rec_k['recall_at_30']:.1%} | R@50: {rec_k['recall_at_50']:.1%} | R@100: {rec_k['recall_at_100']:.1%}")
        
    recall_out_path = os.path.join(RESULTS_DIR, "candidate_recall_analysis.json")
    with open(recall_out_path, "w", encoding="utf-8") as f:
        json.dump(recall_analyses, f, indent=2)
    print(f"  Recall analysis saved to: {recall_out_path}")
    print(f"  Blocking finished in {runtimes['blocking_and_candidate_generation']}s | RAM RSS: {get_peak_memory_mb()} MB")
    
    # -------------------------------------------------------------------------
    # Step 4: Feature Extraction on Candidate Pairs
    # -------------------------------------------------------------------------
    print("\n[Step 4/7] Extracting Features on Candidate Pairs...")
    t0 = time.time()
    
    # Extract training pairs (using top-30 candidates per S1 entity)
    X_train = []
    y_train = []
    train_cands_map = retrievers["train"].candidate_records
    
    for s1 in s1_data["train"]:
        s1_id = s1["entity_id"]
        true_set = set(gt_map.get(s1_id, []))
        cands = candidate_lists["train"][s1_id][:30]
        for cid, _ in cands:
            cand_rec = train_cands_map[cid]
            feat_vec = compute_pair_features(s1, cand_rec)
            label = 1 if cid in true_set else 0
            X_train.append(feat_vec)
            y_train.append(label)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    runtimes["feature_extraction"] = round(time.time() - t0, 2)
    
    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    print(f"  Training pairs generated: {len(X_train):,} (Positives: {pos_count:,}, Negatives: {neg_count:,})")
    print(f"  Feature dimensions: {X_train.shape[1]} features ({', '.join(FEATURE_NAMES[:4])}, ...)")
    print(f"  Feature extraction finished in {runtimes['feature_extraction']}s | RAM RSS: {get_peak_memory_mb()} MB")
    
    # -------------------------------------------------------------------------
    # Step 5: Model Training (Lightweight Classifier)
    # -------------------------------------------------------------------------
    print("\n[Step 5/7] Training Lightweight Classifier (HistGradientBoosting)...")
    t0 = time.time()
    clf = LightweightClassifier(max_iter=100, min_samples_leaf=20, random_state=42)
    clf.fit(X_train, y_train)
    runtimes["model_training"] = round(time.time() - t0, 2)
    
    model_save_path = os.path.join(MODELS_DIR, "baseline_classifier.joblib")
    clf.save(model_save_path)
    print(f"  Classifier trained successfully in {runtimes['model_training']}s")
    print(f"  Model artifact saved to: {model_save_path}")
    
    # -------------------------------------------------------------------------
    # Step 6: Threshold Tuning on Tuning Split (Macro F_0.5)
    # -------------------------------------------------------------------------
    print("\n[Step 6/7] Tuning Thresholds on Tuning Split (Per-Source-1 Macro F_0.5)...")
    t0 = time.time()
    tune_records = retrievers["tune"].candidate_records
    
    # Baseline scorer
    sim_baseline = SimpleSimilarityBaseline()
    tune_res_baseline = tune_threshold(
        score_pairs_fn=sim_baseline.score_pair,
        s1_entities=s1_data["tune"],
        candidate_lists=candidate_lists["tune"],
        candidate_records=tune_records,
        gt_map=gt_map,
        thresholds=[round(t, 2) for t in np.arange(0.20, 0.96, 0.02)]
    )
    
    # Classifier scorer
    def clf_score_pair(s1, cand_rec):
        feats = np.array([compute_pair_features(s1, cand_rec)], dtype=np.float32)
        return float(clf.predict_proba(feats)[0])
        
    tune_res_clf = tune_threshold(
        score_pairs_fn=clf_score_pair,
        s1_entities=s1_data["tune"],
        candidate_lists=candidate_lists["tune"],
        candidate_records=tune_records,
        gt_map=gt_map,
        thresholds=[round(t, 2) for t in np.arange(0.10, 0.96, 0.02)]
    )
    runtimes["threshold_tuning"] = round(time.time() - t0, 2)
    
    tau_base = tune_res_baseline["best_threshold"]
    tau_clf = tune_res_clf["best_threshold"]
    print(f"  [Tuning Split] Simple Baseline Best Threshold: tau = {tau_base:.2f} -> Macro F_0.5 = {tune_res_baseline['best_macro_f05']:.4f}")
    print(f"  [Tuning Split] Classifier Best Threshold:     tau = {tau_clf:.2f} -> Macro F_0.5 = {tune_res_clf['best_macro_f05']:.4f}")
    print(f"  Threshold tuning finished in {runtimes['threshold_tuning']}s | RAM RSS: {get_peak_memory_mb()} MB")
    
    # -------------------------------------------------------------------------
    # Step 7: Final Holdout Evaluation with Frozen Settings
    # -------------------------------------------------------------------------
    print("\n[Step 7/7] Evaluating Frozen Models on Holdout Split (Unbiased Test)...")
    t0 = time.time()
    holdout_records = retrievers["holdout"].candidate_records
    
    # 1. Simple Baseline on Holdout
    holdout_preds_base = {}
    for s1 in s1_data["holdout"]:
        s1_id = s1["entity_id"]
        cands = candidate_lists["holdout"].get(s1_id, [])
        matched = set()
        for cid, _ in cands:
            cand_rec = holdout_records[cid]
            if sim_baseline.score_pair(s1, cand_rec) >= tau_base:
                matched.add(cid)
        holdout_preds_base[s1_id] = matched
        
    eval_base_holdout = evaluate_predictions(gt_map, holdout_preds_base, s1_data["holdout"])
    
    # 2. Classifier on Holdout
    holdout_preds_clf = {}
    scored_holdout_details = {}
    for s1 in s1_data["holdout"]:
        s1_id = s1["entity_id"]
        cands = candidate_lists["holdout"].get(s1_id, [])
        matched = set()
        scored_pairs = []
        for cid, _ in cands:
            cand_rec = holdout_records[cid]
            prob = clf_score_pair(s1, cand_rec)
            scored_pairs.append((cid, prob))
            if prob >= tau_clf:
                matched.add(cid)
        holdout_preds_clf[s1_id] = matched
        scored_holdout_details[s1_id] = scored_pairs
        
    eval_clf_holdout = evaluate_predictions(gt_map, holdout_preds_clf, s1_data["holdout"])
    runtimes["holdout_evaluation"] = round(time.time() - t0, 2)
    
    # Error Analysis: Extract FP and FN cases
    fps = []
    fns = []
    s1_lookup = {x["entity_id"]: x for x in s1_data["holdout"]}
    
    for s1_id, preds in holdout_preds_clf.items():
        true_set = set(gt_map.get(s1_id, []))
        s1_rec = s1_lookup[s1_id]
        
        # False positives
        fp_ids = preds - true_set
        for cid in list(fp_ids)[:2]:
            cand_rec = holdout_records[cid]
            fps.append({
                "source1_id": s1_id,
                "source1_name": s1_rec["business_name"],
                "source1_address": s1_rec["business_address"],
                "source1_country": s1_rec["country"],
                "predicted_candidate_id": cid,
                "predicted_candidate_name": cand_rec[1],
                "predicted_candidate_address": cand_rec[2],
                "error_type": "False Positive (Unjustified Merge)"
            })
            
        # False negatives
        fn_ids = true_set - preds
        for cid in list(fn_ids)[:2]:
            cand_in_pool = cid in holdout_records
            cand_rec = holdout_records.get(cid, (cid, "NOT_IN_POOL", "", ""))
            fns.append({
                "source1_id": s1_id,
                "source1_name": s1_rec["business_name"],
                "source1_address": s1_rec["business_address"],
                "source1_country": s1_rec["country"],
                "missed_true_id": cid,
                "missed_name": cand_rec[1],
                "missed_address": cand_rec[2],
                "in_retrieval_pool": cand_in_pool,
                "error_type": "False Negative (Missed Link)"
            })
            
        if len(fps) >= 5 and len(fns) >= 5:
            break
            
    error_cases = {"false_positives": fps[:5], "false_negatives": fns[:5]}
    error_cases_path = os.path.join(RESULTS_DIR, "error_cases.json")
    with open(error_cases_path, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2)
        
    total_experiment_runtime = round(time.time() - total_start_time, 2)
    peak_mem_mb = get_peak_memory_mb()
    runtimes["total_experiment_runtime"] = total_experiment_runtime
    
    # Save overall experiment metrics
    experiment_metrics = {
        "benchmark_label": "Sampled-Pool Entity Resolution Benchmark (9,000 S1 Entities)",
        "hardware_environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "peak_memory_mb": peak_mem_mb
        },
        "stage_runtimes_seconds": runtimes,
        "sample_sizes": {
            "train_s1": len(s1_data["train"]),
            "tune_s1": len(s1_data["tune"]),
            "holdout_s1": len(s1_data["holdout"]),
            "train_candidate_pool": pool_summary["train"]["total_candidate_pool"],
            "tune_candidate_pool": pool_summary["tune"]["total_candidate_pool"],
            "holdout_candidate_pool": pool_summary["holdout"]["total_candidate_pool"]
        },
        "candidate_recall": {
            "tuning_r_at_20": recall_analyses["tune"]["recall_by_k"]["recall_at_20"],
            "tuning_r_at_50": recall_analyses["tune"]["recall_by_k"]["recall_at_50"],
            "holdout_r_at_20": recall_analyses["holdout"]["recall_by_k"]["recall_at_20"],
            "holdout_r_at_50": recall_analyses["holdout"]["recall_by_k"]["recall_at_50"]
        },
        "tuning_results": {
            "simple_baseline": {
                "frozen_threshold": tau_base,
                "tuning_macro_f05": tune_res_baseline["best_macro_f05"]
            },
            "classifier": {
                "frozen_threshold": tau_clf,
                "tuning_macro_f05": tune_res_clf["best_macro_f05"]
            }
        },
        "holdout_results": {
            "simple_baseline": {
                "macro_f05": eval_base_holdout["macro_f05"],
                "macro_precision": eval_base_holdout["macro_precision"],
                "macro_recall": eval_base_holdout["macro_recall"],
                "singleton_accuracy": eval_base_holdout["singleton_accuracy"],
                "country_breakdown": eval_base_holdout["country_breakdown"]
            },
            "classifier": {
                "macro_f05": eval_clf_holdout["macro_f05"],
                "macro_precision": eval_clf_holdout["macro_precision"],
                "macro_recall": eval_clf_holdout["macro_recall"],
                "singleton_accuracy": eval_clf_holdout["singleton_accuracy"],
                "country_breakdown": eval_clf_holdout["country_breakdown"]
            }
        }
    }
    
    metrics_path = os.path.join(RESULTS_DIR, "experiment_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metrics, f, indent=2)
        
    print("\n" + "=" * 80)
    print(" EXPERIMENT RESULTS & COMPARISON (HOLDOUT SPLIT, 2,000 S1 ENTITIES)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Simple Baseline':<20} | {'Lightweight Classifier':<25}")
    print("-" * 75)
    print(f"{'Frozen Threshold (tau)':<25} | {tau_base:<20.2f} | {tau_clf:<25.2f}")
    print(f"{'Macro F_0.5 Score':<25} | {eval_base_holdout['macro_f05']:<20.4f} | {eval_clf_holdout['macro_f05']:<25.4f}")
    print(f"{'Macro Precision':<25} | {eval_base_holdout['macro_precision']:<20.4f} | {eval_clf_holdout['macro_precision']:<25.4f}")
    print(f"{'Macro Recall':<25} | {eval_base_holdout['macro_recall']:<20.4f} | {eval_clf_holdout['macro_recall']:<25.4f}")
    print(f"{'Singleton Accuracy':<25} | {eval_base_holdout['singleton_accuracy']:<20.4f} | {eval_clf_holdout['singleton_accuracy']:<25.4f}")
    print("-" * 75)
    
    print("\nCountry Breakdown (Classifier on Holdout):")
    for c, c_m in eval_clf_holdout["country_breakdown"].items():
        print(f"  {c:<8}: Macro F_0.5 = {c_m['macro_f05']:.4f} | Singleton Acc = {c_m['singleton_accuracy']:.4f} (N={c_m['total_entities']})")
        
    print(f"\nStage Runtimes:")
    for k, v in runtimes.items():
        print(f"  {k:<35}: {v:>6.2f}s")
    print(f"  Total Experiment Runtime           : {total_experiment_runtime:>6.2f}s")
    print(f"  Peak RAM RSS                       : {peak_mem_mb:>6.2f} MB")
    print("=" * 80)
    print(f"\nArtifacts successfully persisted:")
    print(f"  - Metrics:      {metrics_path}")
    print(f"  - Recall Curve: {recall_out_path}")
    print(f"  - Error Cases:  {error_cases_path}")
    print(f"  - Model:        {model_save_path}")
    print(f"  - Manifests:    {SPLITS_DIR}/")

if __name__ == "__main__":
    main()
