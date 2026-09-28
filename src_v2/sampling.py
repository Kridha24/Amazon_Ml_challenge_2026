"""
Stratified sampling and candidate pool extraction for V2.
Preserves V1 train, tune, and holdout_dev splits for direct head-to-head comparison,
and samples a fresh, untouched holdout split (holdout_fresh) from unused clusters.
Extracts candidate pools and supports an expanded 100k distractor pool for stress testing.
"""

import os
import re
import json
import random
from collections import defaultdict, Counter

def clean_first_token(name):
    """Extract primary significant first word for distractor matching (preserving Unicode)."""
    text = name.lower()
    text = re.sub(r"['\u2019]s\b", '', text)
    text = re.sub(r'[^\w\s]+', ' ', text)
    stopwords = {
        'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
        'co', 'company', 'and', 'the', 'of', 'in', 'near', 'at', 'on', 'by', 'services'
    }
    tokens = [t for t in text.split() if t not in stopwords and len(t) >= 3]
    return tokens[0] if tokens else ""

def get_v2_splits(dataset_dir, splits_v1_dir, seed=100):
    """
    Loads V1 splits (train, tune, holdout_dev) and samples a fresh untouched holdout (holdout_fresh)
    from unused Source 1 clusters.
    """
    # 1. Load V1 manifests
    with open(os.path.join(splits_v1_dir, "manifest_train.json"), "r") as f:
        train_ids = json.load(f)["s1_ids"]
    with open(os.path.join(splits_v1_dir, "manifest_tune.json"), "r") as f:
        tune_ids = json.load(f)["s1_ids"]
    with open(os.path.join(splits_v1_dir, "manifest_holdout.json"), "r") as f:
        holdout_dev_ids = json.load(f)["s1_ids"]
        
    used_s1_ids = set(train_ids) | set(tune_ids) | set(holdout_dev_ids)
    
    # 2. Load ground truth
    gt_map = {}
    gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.partition("\t")
            s1 = s1.strip()
            rest = rest.strip()
            gt_map[s1] = [m.strip() for m in rest.split(",") if m.strip()] if rest else []
            
    # 3. Stratify unused S1 entities to sample holdout_fresh
    strata = defaultdict(list)
    s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            s1, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            if s1 in used_s1_ids:
                continue
                
            cnt = len(gt_map.get(s1, []))
            cat = "0" if cnt == 0 else ("1" if cnt == 1 else ("2" if cnt == 2 else ("3" if cnt == 3 else ("4" if cnt == 4 else "5+"))))
            strata[(country, cat)].append(s1)
            
    total_unused = sum(len(v) for v in strata.values())
    rng = random.Random(seed)
    holdout_fresh_ids = []
    
    for stratum_key, members in sorted(strata.items()):
        rng.shuffle(members)
        frac = len(members) / total_unused
        n_fresh = int(round(2000 * frac))
        holdout_fresh_ids.extend(members[:n_fresh])
        
    # Guarantee exactly 2000
    if len(holdout_fresh_ids) < 2000:
        remaining = [m for members in strata.values() for m in members if m not in holdout_fresh_ids]
        holdout_fresh_ids.extend(remaining[:2000 - len(holdout_fresh_ids)])
    elif len(holdout_fresh_ids) > 2000:
        holdout_fresh_ids = holdout_fresh_ids[:2000]
        
    splits_s1 = {
        "train": train_ids,
        "tune": tune_ids,
        "holdout_dev": holdout_dev_ids,
        "holdout_fresh": holdout_fresh_ids
    }
    
    return splits_s1, gt_map

def extract_v2_samples_and_pools(data_dir, splits_s1, gt_map, include_large_tuning_pool=True, seed=42):
    """
    Stream full datasets to extract:
    1. Full S1 records for all splits.
    2. Pools for train, tune, holdout_dev, holdout_fresh.
    3. Optional expanded distractor pool (~100k records) for tuning stress test.
    """
    s1_all_set = {s1: split for split, s1_list in splits_s1.items() for s1 in s1_list}
    true_id_to_split = {}
    for s1, split in s1_all_set.items():
        for mid in gt_map.get(s1, []):
            true_id_to_split[mid] = split
            
    # Load S1 full records
    s1_data = {split: [] for split in splits_s1}
    s1_keywords = {split: set() for split in splits_s1}
    
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            s1_id = parts[0]
            if s1_id in s1_all_set:
                split = s1_all_set[s1_id]
                rec = {
                    "entity_id": s1_id,
                    "business_name": parts[1],
                    "business_address": parts[2],
                    "country": parts[3]
                }
                s1_data[split].append(rec)
                kw = clean_first_token(parts[1])
                if kw:
                    s1_keywords[split].add(kw)
                    
    # Distractor quotas per source per split
    distractor_quotas = {
        "train": {"hard": 15000, "random": 10000},
        "tune": {"hard": 6000, "random": 4000},
        "holdout_dev": {"hard": 6000, "random": 4000},
        "holdout_fresh": {"hard": 6000, "random": 4000}
    }
    
    pools = {split: {"S2": {}, "S3": {}} for split in splits_s1}
    pool_stats = {split: {"S2": Counter(), "S3": Counter()} for split in splits_s1}
    
    # Large distractor pool for tuning stress test (target: +45,000 S2, +45,000 S3)
    large_tuning_pool_extra = {"S2": {}, "S3": {}}
    large_target_per_source = 45000 if include_large_tuning_pool else 0
    
    sources = [
        ("S2", os.path.join(data_dir, "train", "train_source2.tsv")),
        ("S3", os.path.join(data_dir, "train", "train_source3.tsv"))
    ]
    
    for src_name, src_path in sources:
        with open(src_path, "r", encoding="utf-8") as f:
            f.readline()
            for line_idx, line in enumerate(f):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    parts += [""] * (4 - len(parts))
                eid, bname, baddr, country = parts[0], parts[1], parts[2], parts[3]
                rec = (eid, bname, baddr, country)
                
                # Check if true match for any split
                if eid in true_id_to_split:
                    split = true_id_to_split[eid]
                    pools[split][src_name][eid] = rec
                    pool_stats[split][src_name]["true_matches"] += 1
                    continue
                    
                # Hard distractor check
                kw = clean_first_token(bname)
                assigned = False
                if kw:
                    for split in splits_s1:
                        if (kw in s1_keywords[split] and 
                            pool_stats[split][src_name]["hard_distractors"] < distractor_quotas[split]["hard"]):
                            pools[split][src_name][eid] = rec
                            pool_stats[split][src_name]["hard_distractors"] += 1
                            assigned = True
                            break
                            
                if assigned:
                    continue
                    
                # Random distractor check
                if line_idx % 60 == 0:
                    for split in splits_s1:
                        if pool_stats[split][src_name]["random_distractors"] < distractor_quotas[split]["random"]:
                            pools[split][src_name][eid] = rec
                            pool_stats[split][src_name]["random_distractors"] += 1
                            assigned = True
                            break
                            
                if assigned:
                    continue
                    
                # Extra distractors for large tuning stress test
                if len(large_tuning_pool_extra[src_name]) < large_target_per_source:
                    if line_idx % 25 == 0:
                        large_tuning_pool_extra[src_name][eid] = rec
                        
    summary = {}
    for split in splits_s1:
        summary[split] = {
            "s1_count": len(s1_data[split]),
            "s2_pool_total": len(pools[split]["S2"]),
            "s2_composition": dict(pool_stats[split]["S2"]),
            "s3_pool_total": len(pools[split]["S3"]),
            "s3_composition": dict(pool_stats[split]["S3"]),
            "total_candidate_pool": len(pools[split]["S2"]) + len(pools[split]["S3"])
        }
        
    return s1_data, pools, summary, large_tuning_pool_extra
