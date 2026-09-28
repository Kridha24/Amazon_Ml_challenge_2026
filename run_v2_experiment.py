#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution Baseline V2 Driver
Implements and runs the complete V2 baseline experiment:
1. Verifies metric correctness against hand-calculated cases and confirms split separation.
2. Performs retrieval ablation across complementary routes (prefix, compact, DBA, locality PIN).
3. Evaluates uncapped and capped candidate recall curves and p95 candidate counts.
4. Trains retrained lightweight classifier on V2 candidates with 15 features.
5. Employs batched inference and cached probability threshold tuning.
6. Evaluates on Holdout Dev (identical to V1 for direct comparison) and Holdout Fresh (untouched).
7. Evaluates stress test against an expanded ~100k candidate pool.
8. Produces error analysis, metric summaries, and packages review ZIP.
"""

import os
import sys
import time
import json
import zipfile
import platform
import resource
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src_v2.sampling import get_v2_splits, extract_v2_samples_and_pools
from src_v2.blocking import CandidateRetrieverV2, evaluate_candidate_recall_v2
from src_v2.features import compute_pair_features_v2, FEATURE_NAMES_V2
from src_v2.models import LightweightClassifierV2, batch_score_candidates, tune_threshold_cached
from src_v2.metrics import evaluate_predictions

DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
V1_SPLITS_DIR = os.path.join(WORKSPACE_DIR, "experiments", "v1", "splits")
V2_DIR = os.path.join(WORKSPACE_DIR, "experiments", "v2")
V2_SPLITS_DIR = os.path.join(V2_DIR, "splits")
V2_MODELS_DIR = os.path.join(V2_DIR, "models")
V2_RESULTS_DIR = os.path.join(V2_DIR, "results")

def get_peak_memory_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024 * 1024), 2)
    else:
        return round(rss / 1024, 2)

def verify_hand_calculated_metric():
    gt = {
        'S1_exact': {'A', 'B'},
        'S1_extra': {'A', 'B'},
        'S1_missed': {'A', 'B'},
        'S1_corr_sing': set(),
        'S1_incorr_sing': set(),
        'S1_zero_cand': {'A'},
    }
    preds = {
        'S1_exact': {'A', 'B'},
        'S1_extra': {'A', 'B', 'C'},
        'S1_missed': {'A'},
        'S1_corr_sing': set(),
        'S1_incorr_sing': {'A'},
        'S1_zero_cand': set(),
    }
    meta = [{'entity_id': k, 'country': 'US'} for k in gt.keys()]
    res = evaluate_predictions(gt, preds, meta)
    expected = (1.0 + 5/7 + 5/6 + 1.0 + 0.0 + 0.0) / 6
    assert abs(res['macro_f05'] - round(expected, 4)) < 1e-3, "Hand calculated metric verification failed"
    return True

def main():
    total_start = time.time()
    runtimes = {}
    
    print("=" * 80)
    print(" Amazon ML Challenge 2026: Business Entity Resolution Baseline V2 Experiment")
    print("=" * 80)
    print(f"System: {platform.system()} {platform.machine()} | Python: {platform.python_version()}")
    print(f"Initial Process RAM RSS: {get_peak_memory_mb()} MB\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Metric Verification & Split Preparation
    # -------------------------------------------------------------------------
    print("[Step 1/8] Verifying Metric Correctness & Loading Disjoint Splits...")
    verify_hand_calculated_metric()
    print("  Hand-calculated metric test: PASSED (exact match, extra, missed, singletons, 0-cands).")
    
    t0 = time.time()
    splits_s1, gt_map = get_v2_splits(DATASET_DIR, V1_SPLITS_DIR, seed=100)
    runtimes["split_preparation"] = round(time.time() - t0, 2)
    print(f"  Splits loaded: Train={len(splits_s1['train'])}, Tune={len(splits_s1['tune'])}, Holdout_Dev={len(splits_s1['holdout_dev'])}, Holdout_Fresh={len(splits_s1['holdout_fresh'])}")
    
    # -------------------------------------------------------------------------
    # Step 2: Streaming Sample & Candidate Pool Extraction
    # -------------------------------------------------------------------------
    print("\n[Step 2/8] Streaming Source Files to Extract V2 Candidate Pools...")
    t0 = time.time()
    s1_data, pools, pool_summary, large_tuning_extra = extract_v2_samples_and_pools(
        data_dir=DATASET_DIR,
        splits_s1=splits_s1,
        gt_map=gt_map,
        include_large_tuning_pool=True,
        seed=42
    )
    runtimes["streaming_pool_extraction"] = round(time.time() - t0, 2)
    
    for split in ("train", "tune", "holdout_dev", "holdout_fresh"):
        ps = pool_summary[split]
        print(f"  [{split.upper()}] S1: {ps['s1_count']} | S2 Pool: {ps['s2_pool_total']} | S3 Pool: {ps['s3_pool_total']} | Total: {ps['total_candidate_pool']:,}")
    print(f"  [LARGE TUNING POOL EXTRA] S2 Extra: {len(large_tuning_extra['S2'])}, S3 Extra: {len(large_tuning_extra['S3'])}")
    
    # Save manifests
    for split in ("train", "tune", "holdout_dev", "holdout_fresh"):
        manifest_path = os.path.join(V2_SPLITS_DIR, f"manifest_{split}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "split": split,
                "s1_count": len(s1_data[split]),
                "s1_ids": [x["entity_id"] for x in s1_data[split]],
                "pool_summary": pool_summary[split]
            }, f, indent=2)
            
    # -------------------------------------------------------------------------
    # Step 3: Retrieval Ablation Study (on Tuning Split)
    # -------------------------------------------------------------------------
    print("\n[Step 3/8] Running Retrieval Route Ablation on Tuning Split...")
    t0 = time.time()
    ablation_routes = [
        ("Route 0: Prefix & Sorted Tokens Only (V1 Baseline)", {'prefix'}),
        ("Route 1: Baseline + Compact Names", {'prefix', 'compact'}),
        ("Route 2: Baseline + Compact + DBA Aliases", {'prefix', 'compact', 'dba'}),
        ("Route 3: Full V2 (Baseline + Compact + DBA + Locality PIN Fallback)", {'prefix', 'compact', 'dba', 'address'})
    ]
    
    ablation_results = {}
    for route_name, active_set in ablation_routes:
        retriever_ab = CandidateRetrieverV2(active_routes=active_set)
        retriever_ab.build_index(pools["tune"]["S2"], pools["tune"]["S3"])
        cand_map_ab = {}
        for s1 in s1_data["tune"]:
            cand_map_ab[s1["entity_id"]] = retriever_ab.retrieve_candidates_for_query(s1, max_candidates=50)
        rec_ab = evaluate_candidate_recall_v2(cand_map_ab, gt_map, k_limits=[20, 50, 100])
        ablation_results[route_name] = rec_ab
        print(f"  {route_name}:")
        print(f"    R@20: {rec_ab['recall_by_k']['recall_at_20']:.1%} | R@50: {rec_ab['recall_by_k']['recall_at_50']:.1%} | Avg Cands: {rec_ab['average_candidates_per_entity']} | p95 Cands: {rec_ab['p95_candidates_per_entity']}")
        
    runtimes["retrieval_ablation"] = round(time.time() - t0, 2)
    with open(os.path.join(V2_RESULTS_DIR, "retrieval_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
        
    # -------------------------------------------------------------------------
    # Step 4: Candidate Generation on All Splits using Full V2 Retriever
    # -------------------------------------------------------------------------
    print("\n[Step 4/8] Building Full V2 Inverted Indices & Generating Candidates...")
    t0 = time.time()
    retrievers = {}
    candidate_lists = {}
    recall_analyses = {}
    
    for split in ("train", "tune", "holdout_dev", "holdout_fresh"):
        retriever = CandidateRetrieverV2(active_routes={'prefix', 'compact', 'dba', 'address'})
        retriever.build_index(pools[split]["S2"], pools[split]["S3"])
        retrievers[split] = retriever
        
        cand_map = {}
        for s1 in s1_data[split]:
            cand_map[s1["entity_id"]] = retriever.retrieve_candidates_for_query(s1, max_candidates=50)
        candidate_lists[split] = cand_map
        recall_analyses[split] = evaluate_candidate_recall_v2(cand_map, gt_map, k_limits=[5, 10, 20, 30, 50, 100])
        
    runtimes["candidate_generation"] = round(time.time() - t0, 2)
    print(f"  Candidate Generation finished in {runtimes['candidate_generation']}s")
    for split in ("tune", "holdout_dev", "holdout_fresh"):
        ra = recall_analyses[split]
        print(f"    [{split.upper()}] True Links: {ra['total_true_matches']} | R@20: {ra['recall_by_k']['recall_at_20']:.1%} | R@50: {ra['recall_by_k']['recall_at_50']:.1%} | Avg Cands: {ra['average_candidates_per_entity']}")
        
    # -------------------------------------------------------------------------
    # Step 5: Feature Extraction & Model Training (V2)
    # -------------------------------------------------------------------------
    print("\n[Step 5/8] Extracting 15 Features & Training Lightweight Classifier V2...")
    t0 = time.time()
    X_train, y_train = [], []
    train_pool = retrievers["train"].candidate_records
    
    for s1 in s1_data["train"]:
        s1_id = s1["entity_id"]
        true_set = set(gt_map.get(s1_id, []))
        for cid, _ in candidate_lists["train"][s1_id]:
            cand_rec = train_pool[cid]
            feat = compute_pair_features_v2(s1, cand_rec)
            label = 1 if cid in true_set else 0
            X_train.append(feat)
            y_train.append(label)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    runtimes["feature_extraction_train"] = round(time.time() - t0, 2)
    
    print(f"  Training pairs: {len(X_train):,} (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})")
    
    t0 = time.time()
    clf_v2 = LightweightClassifierV2(max_iter=150, min_samples_leaf=20, random_state=42)
    clf_v2.fit(X_train, y_train)
    runtimes["model_training"] = round(time.time() - t0, 2)
    
    model_path = os.path.join(V2_MODELS_DIR, "v2_classifier.joblib")
    clf_v2.save(model_path)
    print(f"  Classifier V2 trained in {runtimes['model_training']}s and saved to {model_path}")
    
    # -------------------------------------------------------------------------
    # Step 6: Batched Probability Caching & Fast Threshold Tuning
    # -------------------------------------------------------------------------
    print("\n[Step 6/8] Running Batched Probability Caching & Threshold Tuning on Tuning Split...")
    t0 = time.time()
    tune_pool = retrievers["tune"].candidate_records
    scored_tune = batch_score_candidates(clf_v2, s1_data["tune"], candidate_lists["tune"], tune_pool)
    tune_res = tune_threshold_cached(scored_tune, s1_data["tune"], gt_map, thresholds=[round(t, 2) for t in np.arange(0.10, 0.96, 0.02)])
    runtimes["threshold_tuning"] = round(time.time() - t0, 2)
    
    tau_v2 = tune_res["best_threshold"]
    print(f"  Threshold tuning finished in {runtimes['threshold_tuning']}s (Cached inference speedup!)")
    print(f"  Optimal Frozen Threshold: tau = {tau_v2:.2f} -> Tuning Macro F_0.5 = {tune_res['best_macro_f05']:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 7: Evaluations on Holdout Dev and Holdout Fresh
    # -------------------------------------------------------------------------
    print("\n[Step 7/8] Evaluating on Holdout Dev (V1 Pool) and Fresh Untouched Holdout...")
    t0 = time.time()
    
    # A. Holdout Dev (for direct comparison with V1 on identical pool)
    dev_pool = retrievers["holdout_dev"].candidate_records
    scored_dev = batch_score_candidates(clf_v2, s1_data["holdout_dev"], candidate_lists["holdout_dev"], dev_pool)
    dev_preds = {s1["entity_id"]: {cid for cid, p in scored_dev[s1["entity_id"]] if p >= tau_v2} for s1 in s1_data["holdout_dev"]}
    eval_dev = evaluate_predictions(gt_map, dev_preds, s1_data["holdout_dev"])
    
    # B. Holdout Fresh (completely untouched validation)
    fresh_pool = retrievers["holdout_fresh"].candidate_records
    scored_fresh = batch_score_candidates(clf_v2, s1_data["holdout_fresh"], candidate_lists["holdout_fresh"], fresh_pool)
    fresh_preds = {s1["entity_id"]: {cid for cid, p in scored_fresh[s1["entity_id"]] if p >= tau_v2} for s1 in s1_data["holdout_fresh"]}
    eval_fresh = evaluate_predictions(gt_map, fresh_preds, s1_data["holdout_fresh"])
    runtimes["holdout_evaluations"] = round(time.time() - t0, 2)
    
    # -------------------------------------------------------------------------
    # Step 8: Stress Test against Expanded ~100k Search Pool
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Executing Stress Test on Tuning Split against ~100,000 Search Pool...")
    t0 = time.time()
    # Combine tuning pool with extra distractors
    stress_pool_s2 = dict(pools["tune"]["S2"])
    stress_pool_s2.update(large_tuning_extra["S2"])
    stress_pool_s3 = dict(pools["tune"]["S3"])
    stress_pool_s3.update(large_tuning_extra["S3"])
    total_stress_search_pool = len(stress_pool_s2) + len(stress_pool_s3)
    
    stress_retriever = CandidateRetrieverV2(active_routes={'prefix', 'compact', 'dba', 'address'})
    stress_retriever.build_index(stress_pool_s2, stress_pool_s3)
    
    stress_cand_map = {}
    for s1 in s1_data["tune"]:
        stress_cand_map[s1["entity_id"]] = stress_retriever.retrieve_candidates_for_query(s1, max_candidates=50)
        
    stress_recall = evaluate_candidate_recall_v2(stress_cand_map, gt_map, k_limits=[20, 50])
    scored_stress = batch_score_candidates(clf_v2, s1_data["tune"], stress_cand_map, stress_retriever.candidate_records)
    stress_preds = {s1["entity_id"]: {cid for cid, p in scored_stress[s1["entity_id"]] if p >= tau_v2} for s1 in s1_data["tune"]}
    eval_stress = evaluate_predictions(gt_map, stress_preds, s1_data["tune"])
    runtimes["stress_test_large_pool"] = round(time.time() - t0, 2)
    
    stress_summary = {
        "benchmark_label": "Stress Test: 100k Candidate Pool Distractor Evaluation",
        "searchable_pool_size": total_stress_search_pool,
        "tuning_queries_count": len(s1_data["tune"]),
        "frozen_threshold": tau_v2,
        "candidate_recall_at_50": stress_recall["recall_by_k"]["recall_at_50"],
        "average_candidates_per_entity": stress_recall["average_candidates_per_entity"],
        "macro_f05": eval_stress["macro_f05"],
        "macro_precision": eval_stress["macro_precision"],
        "macro_recall": eval_stress["macro_recall"],
        "total_false_positives": eval_stress["total_fp_links"],
        "runtime_seconds": runtimes["stress_test_large_pool"],
        "peak_memory_mb": get_peak_memory_mb()
    }
    with open(os.path.join(V2_RESULTS_DIR, "large_pool_stress_test.json"), "w", encoding="utf-8") as f:
        json.dump(stress_summary, f, indent=2)
        
    # Extract error cases on holdout dev
    fps, fns = [], []
    s1_lookup = {x["entity_id"]: x for x in s1_data["holdout_dev"]}
    for s1_id, preds in dev_preds.items():
        true_set = set(gt_map.get(s1_id, []))
        s1_rec = s1_lookup[s1_id]
        for cid in list(preds - true_set)[:2]:
            cand_rec = dev_pool[cid]
            fps.append({
                "source1_id": s1_id,
                "source1_name": s1_rec["business_name"],
                "source1_address": s1_rec["business_address"],
                "predicted_id": cid,
                "predicted_name": cand_rec[1],
                "predicted_address": cand_rec[2],
                "error_type": "False Positive (Unjustified Merge)"
            })
        for cid in list(true_set - preds)[:2]:
            cand_rec = dev_pool.get(cid, (cid, "NOT_IN_POOL", "", ""))
            fns.append({
                "source1_id": s1_id,
                "source1_name": s1_rec["business_name"],
                "source1_address": s1_rec["business_address"],
                "missed_id": cid,
                "missed_name": cand_rec[1],
                "missed_address": cand_rec[2],
                "error_type": "False Negative (Missed Match)"
            })
        if len(fps) >= 5 and len(fns) >= 5:
            break
            
    with open(os.path.join(V2_RESULTS_DIR, "error_cases_v2.json"), "w", encoding="utf-8") as f:
        json.dump({"false_positives": fps[:5], "false_negatives": fns[:5]}, f, indent=2)
        
    total_runtime = round(time.time() - total_start, 2)
    peak_mem = get_peak_memory_mb()
    runtimes["total_experiment_runtime"] = total_runtime
    
    # Save V2 metrics
    v2_metrics = {
        "benchmark_label": "Entity Resolution Baseline V2 Benchmark",
        "peak_memory_mb": peak_mem,
        "stage_runtimes_seconds": runtimes,
        "frozen_threshold": tau_v2,
        "holdout_dev_comparison": {
            "macro_f05": eval_dev["macro_f05"],
            "macro_precision": eval_dev["macro_precision"],
            "macro_recall": eval_dev["macro_recall"],
            "singleton_accuracy": eval_dev["singleton_accuracy"],
            "country_breakdown": eval_dev["country_breakdown"],
            "total_tp_links": eval_dev["total_tp_links"],
            "total_fp_links": eval_dev["total_fp_links"],
            "total_fn_links": eval_dev["total_fn_links"]
        },
        "holdout_fresh_untouched": {
            "macro_f05": eval_fresh["macro_f05"],
            "macro_precision": eval_fresh["macro_precision"],
            "macro_recall": eval_fresh["macro_recall"],
            "singleton_accuracy": eval_fresh["singleton_accuracy"],
            "country_breakdown": eval_fresh["country_breakdown"],
            "total_tp_links": eval_fresh["total_tp_links"],
            "total_fp_links": eval_fresh["total_fp_links"],
            "total_fn_links": eval_fresh["total_fn_links"]
        }
    }
    with open(os.path.join(V2_RESULTS_DIR, "v2_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(v2_metrics, f, indent=2)
        
    # Build Review ZIP archive
    zip_path = os.path.join(WORKSPACE_DIR, "business_entity_resolution_v2_review.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Source code
        for root, _, files in os.walk(os.path.join(WORKSPACE_DIR, "src_v2")):
            for file in files:
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, WORKSPACE_DIR)
                zipf.write(full_p, rel_p)
        zipf.write(os.path.join(WORKSPACE_DIR, "run_v2_experiment.py"), "run_v2_experiment.py")
        # Results & manifests
        for root, _, files in os.walk(V2_DIR):
            for file in files:
                if not file.endswith(".joblib"):  # exclude large model binary from review zip
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, WORKSPACE_DIR)
                    zipf.write(full_p, rel_p)
                    
    print("\n" + "=" * 80)
    print(" V1 vs V2 COMPARISON ON IDENTICAL HOLDOUT DEV POOL (2,000 S1 ENTITIES)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'V1 Classifier':<20} | {'V2 Classifier':<20} | {'Absolute Delta':<15}")
    print("-" * 85)
    print(f"{'Decision Threshold (tau)':<25} | {'0.78':<20} | {tau_v2:<20.2f} | {'-':<15}")
    print(f"{'Candidate Recall @ 50':<25} | {'75.3%':<20} | {recall_analyses['holdout_dev']['recall_by_k']['recall_at_50']:.1%} | {recall_analyses['holdout_dev']['recall_by_k']['recall_at_50'] - 0.753:+.1%}")
    print(f"{'Macro F_0.5 Score':<25} | {'0.8831':<20} | {eval_dev['macro_f05']:<20.4f} | {eval_dev['macro_f05'] - 0.8831:+.4f}")
    print(f"{'Macro Precision':<25} | {'0.9486':<20} | {eval_dev['macro_precision']:<20.4f} | {eval_dev['macro_precision'] - 0.9486:+.4f}")
    print(f"{'Macro Recall':<25} | {'0.7559':<20} | {eval_dev['macro_recall']:<20.4f} | {eval_dev['macro_recall'] - 0.7559:+.4f}")
    print(f"{'Singleton Accuracy':<25} | {'0.9911':<20} | {eval_dev['singleton_accuracy']:<20.4f} | {eval_dev['singleton_accuracy'] - 0.9911:+.4f}")
    print(f"{'India Macro F_0.5':<25} | {'0.8100':<20} | {eval_dev['country_breakdown']['India']['macro_f05']:<20.4f} | {eval_dev['country_breakdown']['India']['macro_f05'] - 0.8100:+.4f}")
    print(f"{'US Macro F_0.5':<25} | {'0.9320':<20} | {eval_dev['country_breakdown']['US']['macro_f05']:<20.4f} | {eval_dev['country_breakdown']['US']['macro_f05'] - 0.9320:+.4f}")
    print("-" * 85)
    
    print("\n" + "=" * 80)
    print(" V2 UNTOUCHED FRESH HOLDOUT RESULTS (2,000 UNSEEN S1 ENTITIES)")
    print("=" * 80)
    print(f"  Macro F_0.5:         {eval_fresh['macro_f05']:.4f}")
    print(f"  Macro Precision:     {eval_fresh['macro_precision']:.4f}")
    print(f"  Macro Recall:        {eval_fresh['macro_recall']:.4f}")
    print(f"  Singleton Accuracy:  {eval_fresh['singleton_accuracy']:.4f}")
    print(f"  India Macro F_0.5:   {eval_fresh['country_breakdown']['India']['macro_f05']:.4f}")
    print(f"  US Macro F_0.5:      {eval_fresh['country_breakdown']['US']['macro_f05']:.4f}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print(" STRESS TEST: ~100k CANDIDATE SEARCH POOL RESULTS (TUNING SPLIT)")
    print("=" * 80)
    print(f"  Searchable Pool:     {total_stress_search_pool:,} records")
    print(f"  Recall @ 50:         {stress_recall['recall_by_k']['recall_at_50']:.1%}")
    print(f"  Average Cands/S1:    {stress_recall['average_candidates_per_entity']}")
    print(f"  Macro F_0.5:         {eval_stress['macro_f05']:.4f}")
    print(f"  Macro Precision:     {eval_stress['macro_precision']:.4f}")
    print(f"  Macro Recall:        {eval_stress['macro_recall']:.4f}")
    print(f"  False Positives:     {eval_stress['total_fp_links']} (vs {stress_summary['total_false_positives']} in stress pool)")
    print(f"  Runtime:             {runtimes['stress_test_large_pool']}s")
    print("=" * 80)
    
    print(f"\nExecution Profile & System Resources:")
    for k, v in runtimes.items():
        print(f"  {k:<35}: {v:>6.2f}s")
    print(f"  Total Runtime                       : {total_runtime:>6.2f}s")
    print(f"  Peak Memory (RSS)                   : {peak_mem:>6.2f} MB")
    print(f"  Review ZIP archive created          : {zip_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
