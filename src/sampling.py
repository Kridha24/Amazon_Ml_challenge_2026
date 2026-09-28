"""
Stratified sampling and streaming dataset extraction for Amazon ML Challenge 2026.
Creates deterministic, disjoint splits of Source 1 entities and realistic candidate pools
with true matches, hard distractors, and random non-matches.
"""

import os
import re
import json
import time
import random
from collections import defaultdict, Counter

def clean_first_token(name):
    """Extract primary significant first word for distractor matching."""
    text = name.lower()
    text = re.sub(r"['\u2019]s\b", '', text)
    text = re.sub(r'[^a-z0-9\u0900-\u097f\u00c0-\u017f]+', ' ', text)
    stopwords = {'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private', 'co', 'company', 'and', 'the', 'of', 'in', 'near'}
    tokens = [t for t in text.split() if t not in stopwords and len(t) >= 3]
    return tokens[0] if tokens else ""

def create_stratified_splits(gt_path, s1_path, target_counts={"train": 5000, "tune": 2000, "holdout": 2000}, seed=42):
    """
    Stratifies Source 1 entities by (country, match_count_category).
    Returns disjoint sets of S1 IDs for each split.
    """
    rng = random.Random(seed)
    
    # 1. Load ground truth match sets and counts
    gt_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.partition("\t")
            s1 = s1.strip()
            rest = rest.strip()
            gt_map[s1] = [m.strip() for m in rest.split(",") if m.strip()] if rest else []
            
    # 2. Group all S1 IDs into strata
    strata = defaultdict(list)
    s1_records = {}
    
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            s1, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            
            cnt = len(gt_map.get(s1, []))
            cat = "0" if cnt == 0 else ("1" if cnt == 1 else ("2" if cnt == 2 else ("3" if cnt == 3 else ("4" if cnt == 4 else "5+"))))
            strata[(country, cat)].append(s1)
            
    total_population = sum(len(v) for v in strata.values())
    
    # 3. Proportional sampling per stratum
    splits_s1 = {"train": [], "tune": [], "holdout": []}
    strata_manifest = {}
    
    for stratum_key, members in sorted(strata.items()):
        rng.shuffle(members)
        frac = len(members) / total_population
        n_tr = int(round(target_counts["train"] * frac))
        n_tu = int(round(target_counts["tune"] * frac))
        n_ho = int(round(target_counts["holdout"] * frac))
        
        # Slices
        tr_slice = members[:n_tr]
        tu_slice = members[n_tr : n_tr + n_tu]
        ho_slice = members[n_tr + n_tu : n_tr + n_tu + n_ho]
        
        splits_s1["train"].extend(tr_slice)
        splits_s1["tune"].extend(tu_slice)
        splits_s1["holdout"].extend(ho_slice)
        
        strata_manifest[f"{stratum_key[0]}_{stratum_key[1]}"] = {
            "population": len(members),
            "train": len(tr_slice),
            "tune": len(tu_slice),
            "holdout": len(ho_slice)
        }
        
    return splits_s1, gt_map, strata_manifest

def extract_split_samples_and_pools(data_dir, splits_s1, gt_map, seed=42):
    """
    Stream full datasets to extract:
    1. Full S1 records for each split.
    2. Candidate pools (true matches + hard distractors + random distractors) for S2 and S3 per split.
    Guarantees strict isolation across splits.
    """
    rng = random.Random(seed)
    
    s1_all_set = {s1: split for split, s1_list in splits_s1.items() for s1 in s1_list}
    
    # Map each true match ID to its split
    true_id_to_split = {}
    for s1, split in s1_all_set.items():
        for mid in gt_map.get(s1, []):
            true_id_to_split[mid] = split
            
    # Load S1 full records
    s1_data = {"train": [], "tune": [], "holdout": []}
    s1_keywords = {"train": set(), "tune": set(), "holdout": set()}
    
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
                    
    # Target distractor quotas per source per split
    distractor_quotas = {
        "train": {"hard": 15000, "random": 10000},
        "tune": {"hard": 6000, "random": 4000},
        "holdout": {"hard": 6000, "random": 4000}
    }
    
    pools = {
        "train": {"S2": {}, "S3": {}},
        "tune": {"S2": {}, "S3": {}},
        "holdout": {"S2": {}, "S3": {}}
    }
    
    pool_stats = {
        "train": {"S2": Counter(), "S3": Counter()},
        "tune": {"S2": Counter(), "S3": Counter()},
        "holdout": {"S2": Counter(), "S3": Counter()}
    }
    
    # Helper to stream source 2 and source 3
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
                
                # Check if true match
                if eid in true_id_to_split:
                    split = true_id_to_split[eid]
                    pools[split][src_name][eid] = rec
                    pool_stats[split][src_name]["true_matches"] += 1
                    continue
                    
                # Otherwise, consider as distractor
                kw = clean_first_token(bname)
                assigned = False
                
                # Try assigning as hard distractor
                if kw:
                    for split in ("train", "tune", "holdout"):
                        if (kw in s1_keywords[split] and 
                            pool_stats[split][src_name]["hard_distractors"] < distractor_quotas[split]["hard"]):
                            pools[split][src_name][eid] = rec
                            pool_stats[split][src_name]["hard_distractors"] += 1
                            assigned = True
                            break
                            
                if assigned:
                    continue
                    
                # Try assigning as random distractor (sample 1 in every 100)
                if line_idx % 70 == 0:
                    for split in ("train", "tune", "holdout"):
                        if pool_stats[split][src_name]["random_distractors"] < distractor_quotas[split]["random"]:
                            pools[split][src_name][eid] = rec
                            pool_stats[split][src_name]["random_distractors"] += 1
                            break
                            
    # Format return summary
    summary = {}
    for split in ("train", "tune", "holdout"):
        summary[split] = {
            "s1_count": len(s1_data[split]),
            "s2_pool_total": len(pools[split]["S2"]),
            "s2_composition": dict(pool_stats[split]["S2"]),
            "s3_pool_total": len(pools[split]["S3"]),
            "s3_composition": dict(pool_stats[split]["S3"]),
            "total_candidate_pool": len(pools[split]["S2"]) + len(pools[split]["S3"])
        }
        
    return s1_data, pools, summary
