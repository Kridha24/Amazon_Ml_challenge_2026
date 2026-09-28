#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Local Dataset & Environment Audit Script
Performs a comprehensive, streaming, memory-bounded audit of:
1. Environment and system resources (Python, CPU, RAM, Disk).
2. All train and test TSV files and ground-truth files (paths, sizes, headers, row counts).
3. Missing fields, duplicate entity IDs, and country distributions.
4. Ground-truth match integrity (S1/S2/S3 existence, duplicate links, singleton rate, match degree distribution).

Preserves all IDs as strings, handles empty match lists correctly, and uses O(1) or bounded memory.
"""

import os
import sys
import time
import json
import shutil
import platform
import subprocess
from collections import Counter

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
OUTPUT_JSON_PATH = os.path.join(WORKSPACE_DIR, "audit_results.json")

def format_bytes(n):
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"

def inspect_environment():
    """Inspect Python version, CPU architecture, RAM, and Disk space."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    
    # Disk usage
    disk = shutil.disk_usage(WORKSPACE_DIR)
    env["disk_total_bytes"] = disk.total
    env["disk_free_bytes"] = disk.free
    env["disk_used_bytes"] = disk.used
    env["disk_total_formatted"] = format_bytes(disk.total)
    env["disk_free_formatted"] = format_bytes(disk.free)
    env["disk_used_formatted"] = format_bytes(disk.used)
    
    # Memory (macOS specific sysctl & vm_stat fallback)
    try:
        total_ram = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode().strip())
        env["ram_total_bytes"] = total_ram
        env["ram_total_formatted"] = format_bytes(total_ram)
    except Exception as e:
        env["ram_total_error"] = str(e)
        
    try:
        vm_stat_out = subprocess.check_output(['vm_stat']).decode()
        vm_lines = vm_stat_out.strip().split('\n')
        vm_dict = {}
        for line in vm_lines[1:]:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().rstrip('.')
                if val.isdigit():
                    vm_dict[key] = int(val)
        
        page_size = 16384  # standard for Apple Silicon (16 KB)
        free_pages = vm_dict.get('Pages free', 0)
        speculative_pages = vm_dict.get('Pages speculative', 0)
        inactive_pages = vm_dict.get('Pages inactive', 0)
        available_bytes = (free_pages + speculative_pages + inactive_pages) * page_size
        env["ram_available_estimated_bytes"] = available_bytes
        env["ram_available_estimated_formatted"] = format_bytes(available_bytes)
        env["vm_stat_summary"] = {k: v for k, v in vm_dict.items() if 'Pages' in k}
    except Exception as e:
        env["vm_stat_error"] = str(e)
        
    return env

def audit_source_tsv(path, expected_prefix):
    """
    Stream and audit a business entity source TSV file.
    Validates headers, row count, missing fields, duplicates, prefixes, and countries.
    """
    filename = os.path.basename(path)
    rel_path = os.path.relpath(path, WORKSPACE_DIR)
    file_size = os.path.getsize(path)
    
    t0 = time.time()
    seen_ids = set()
    dup_ids = set()
    invalid_prefixes = set()
    missing_fields = Counter()
    countries = Counter()
    total_rows = 0
    header_list = []
    
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return {
                "file": rel_path,
                "size_bytes": file_size,
                "error": "File is completely empty"
            }
        header_list = header_line.rstrip("\r\n").split("\t")
        
        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            total_rows += 1
            
            # Row length check
            if len(parts) < 4:
                parts = parts + [""] * (4 - len(parts))
            elif len(parts) > 4:
                missing_fields["_malformed_columns_count"] += 1
                
            entity_id, b_name, b_addr, country = parts[0], parts[1], parts[2], parts[3]
            
            # ID check
            if entity_id in seen_ids:
                dup_ids.add(entity_id)
            else:
                seen_ids.add(entity_id)
                
            if not entity_id.startswith(expected_prefix):
                invalid_prefixes.add(entity_id)
                
            # Missing checks (whitespace or empty)
            if not entity_id.strip():
                missing_fields["entity_id"] += 1
            if not b_name.strip():
                missing_fields["business_name"] += 1
            if not b_addr.strip():
                missing_fields["business_address"] += 1
            if not country.strip():
                missing_fields["country"] += 1
            else:
                countries[country.strip()] += 1
                
    elapsed = time.time() - t0
    
    return {
        "file": rel_path,
        "size_bytes": file_size,
        "size_formatted": format_bytes(file_size),
        "header": header_list,
        "total_rows": total_rows,
        "unique_entity_ids": len(seen_ids),
        "duplicate_entity_ids_count": len(dup_ids),
        "duplicate_entity_ids_samples": list(dup_ids)[:5],
        "invalid_prefix_count": len(invalid_prefixes),
        "invalid_prefix_samples": list(invalid_prefixes)[:5],
        "missing_fields": dict(missing_fields),
        "country_distribution": dict(countries),
        "audit_duration_seconds": round(elapsed, 2)
    }

def audit_ground_truth(path):
    """
    Stream and audit train_ground_truth.tsv.
    Extracts match cardinality, singletons, duplicate IDs, and sets of matched S2 and S3 IDs.
    """
    rel_path = os.path.relpath(path, WORKSPACE_DIR)
    file_size = os.path.getsize(path)
    t0 = time.time()
    
    seen_s1 = set()
    dup_s1 = set()
    intra_list_dupes = 0
    empty_matches = 0
    total_rows = 0
    header_list = []
    
    s2_matched_set = set()
    s3_matched_set = set()
    other_prefix_matches = set()
    
    match_cardinality = Counter() # number of matches per entity
    sources_breakdown = Counter() # singleton, S2 only, S3 only, both S2 and S3
    
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return {"file": rel_path, "size_bytes": file_size, "error": "File is empty"}, set(), set(), set()
        header_list = header_line.rstrip("\r\n").split("\t")
        
        for line in f:
            total_rows += 1
            s1, tab, rest = line.partition("\t")
            s1 = s1.strip()
            
            if s1 in seen_s1:
                dup_s1.add(s1)
            else:
                seen_s1.add(s1)
                
            matched_str = rest.rstrip("\r\n").strip()
            if not matched_str:
                empty_matches += 1
                match_cardinality[0] += 1
                sources_breakdown["singleton_zero_matches"] += 1
                continue
                
            raw_ids = [x.strip() for x in matched_str.split(",") if x.strip()]
            n_ids = len(raw_ids)
            match_cardinality[n_ids] += 1
            
            unique_ids = set(raw_ids)
            if len(unique_ids) < n_ids:
                intra_list_dupes += 1
                
            has_s2 = False
            has_s3 = False
            for mid in unique_ids:
                if mid.startswith("S2-"):
                    has_s2 = True
                    s2_matched_set.add(mid)
                elif mid.startswith("S3-"):
                    has_s3 = True
                    s3_matched_set.add(mid)
                else:
                    other_prefix_matches.add(mid)
                    
            if has_s2 and has_s3:
                sources_breakdown["both_S2_and_S3"] += 1
            elif has_s2:
                sources_breakdown["S2_only"] += 1
            elif has_s3:
                sources_breakdown["S3_only"] += 1
                
    elapsed = time.time() - t0
    
    return {
        "file": rel_path,
        "size_bytes": file_size,
        "size_formatted": format_bytes(file_size),
        "header": header_list,
        "total_rows": total_rows,
        "unique_s1_ids": len(seen_s1),
        "duplicate_s1_rows": len(dup_s1),
        "empty_matches_count (singletons)": empty_matches,
        "empty_matches_percentage": round(empty_matches / total_rows * 100, 2) if total_rows else 0,
        "intra_list_duplicates_count": intra_list_dupes,
        "total_unique_s2_matched": len(s2_matched_set),
        "total_unique_s3_matched": len(s3_matched_set),
        "invalid_prefix_matches_count": len(other_prefix_matches),
        "match_cardinality_distribution": dict(sorted(match_cardinality.items())),
        "sources_breakdown": dict(sources_breakdown),
        "audit_duration_seconds": round(elapsed, 2)
    }, seen_s1, s2_matched_set, s3_matched_set

def verify_ground_truth_cross_references(gt_s1_set, gt_s2_set, gt_s3_set):
    """
    Verifies that all ground truth S1 IDs exist in train_source1.tsv,
    all matched S2 IDs exist in train_source2.tsv,
    and all matched S3 IDs exist in train_source3.tsv.
    Uses set.discard() to avoid unbounded memory growth.
    """
    t0 = time.time()
    results = {}
    
    # Check S1
    s1_src_path = os.path.join(DATASET_DIR, "train", "train_source1.tsv")
    s1_remaining = set(gt_s1_set)
    s1_source_total = 0
    with open(s1_src_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1_source_total += 1
            s1_id, _, _ = line.partition("\t")
            s1_remaining.discard(s1_id.strip())
            
    results["gt_s1_total"] = len(gt_s1_set)
    results["train_source1_total_rows"] = s1_source_total
    results["gt_s1_missing_from_source1_count"] = len(s1_remaining)
    results["gt_s1_missing_samples"] = list(s1_remaining)[:5]
    del s1_remaining
    
    # Check S2
    s2_src_path = os.path.join(DATASET_DIR, "train", "train_source2.tsv")
    s2_remaining = set(gt_s2_set)
    s2_source_total = 0
    with open(s2_src_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s2_source_total += 1
            s2_id, _, _ = line.partition("\t")
            s2_remaining.discard(s2_id.strip())
            
    results["gt_s2_unique_matched"] = len(gt_s2_set)
    results["train_source2_total_rows"] = s2_source_total
    results["gt_s2_missing_from_source2_count"] = len(s2_remaining)
    results["gt_s2_missing_samples"] = list(s2_remaining)[:5]
    results["train_source2_unmatched_singletons"] = s2_source_total - len(gt_s2_set)
    del s2_remaining
    
    # Check S3
    s3_src_path = os.path.join(DATASET_DIR, "train", "train_source3.tsv")
    s3_remaining = set(gt_s3_set)
    s3_source_total = 0
    with open(s3_src_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s3_source_total += 1
            s3_id, _, _ = line.partition("\t")
            s3_remaining.discard(s3_id.strip())
            
    results["gt_s3_unique_matched"] = len(gt_s3_set)
    results["train_source3_total_rows"] = s3_source_total
    results["gt_s3_missing_from_source3_count"] = len(s3_remaining)
    results["gt_s3_missing_samples"] = list(s3_remaining)[:5]
    results["train_source3_unmatched_singletons"] = s3_source_total - len(gt_s3_set)
    del s3_remaining
    
    results["cross_check_duration_seconds"] = round(time.time() - t0, 2)
    return results

def main():
    print("=" * 70)
    print(" Amazon ML Challenge 2026 - Dataset & Environment Audit")
    print("=" * 70)
    start_time = time.time()
    
    audit_data = {}
    
    # 1. Environment
    print("\n[1/4] Inspecting System & Hardware Environment...")
    env_info = inspect_environment()
    audit_data["environment"] = env_info
    print(f"  Python: {env_info['python_version'].split()[0]}")
    print(f"  Platform: {env_info['platform']}")
    print(f"  Architecture: {env_info['machine']} ({env_info['processor']})")
    print(f"  RAM Total: {env_info.get('ram_total_formatted', 'N/A')}")
    print(f"  RAM Available (Est): {env_info.get('ram_available_estimated_formatted', 'N/A')}")
    print(f"  Disk Total: {env_info['disk_total_formatted']} | Free: {env_info['disk_free_formatted']}")
    
    # 2. Source TSV Files Audit
    print("\n[2/4] Auditing Train and Test TSV Source Files (Streaming)...")
    source_files = [
        (os.path.join(DATASET_DIR, "train", "train_source1.tsv"), "S1-"),
        (os.path.join(DATASET_DIR, "train", "train_source2.tsv"), "S2-"),
        (os.path.join(DATASET_DIR, "train", "train_source3.tsv"), "S3-"),
        (os.path.join(DATASET_DIR, "test", "test_source1.tsv"), "S1-"),
        (os.path.join(DATASET_DIR, "test", "test_source2.tsv"), "S2-"),
        (os.path.join(DATASET_DIR, "test", "test_source3.tsv"), "S3-"),
    ]
    
    audit_data["source_files"] = {}
    for path, prefix in source_files:
        fname = os.path.basename(path)
        print(f"  Auditing {fname}...")
        res = audit_source_tsv(path, prefix)
        audit_data["source_files"][fname] = res
        print(f"    -> Rows: {res['total_rows']:,}, Size: {res['size_formatted']}, Header: {res['header']}")
        print(f"    -> Duplicates: {res['duplicate_entity_ids_count']}, Missing Fields: {res['missing_fields']}")
        print(f"    -> Countries: {res['country_distribution']} ({res['audit_duration_seconds']}s)")
        
    # 3. Ground Truth Audit
    print("\n[3/4] Auditing Training Ground Truth File...")
    gt_path = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")
    gt_res, gt_s1, gt_s2, gt_s3 = audit_ground_truth(gt_path)
    audit_data["ground_truth"] = gt_res
    print(f"  Rows: {gt_res['total_rows']:,}, Size: {gt_res['size_formatted']}")
    print(f"  Singletons (no match): {gt_res['empty_matches_count (singletons)']:,} ({gt_res['empty_matches_percentage']}%)")
    print(f"  Unique S2 Matches: {gt_res['total_unique_s2_matched']:,}")
    print(f"  Unique S3 Matches: {gt_res['total_unique_s3_matched']:,}")
    print(f"  Intra-list duplicates: {gt_res['intra_list_duplicates_count']}")
    print(f"  Breakdown: {gt_res['sources_breakdown']}")
    
    # 4. Cross-Reference Verification
    print("\n[4/4] Verifying Ground Truth Integrity Against Source 1, 2, 3...")
    cross_res = verify_ground_truth_cross_references(gt_s1, gt_s2, gt_s3)
    audit_data["cross_references"] = cross_res
    print(f"  Missing S1 from Source 1: {cross_res['gt_s1_missing_from_source1_count']}")
    print(f"  Missing S2 from Source 2: {cross_res['gt_s2_missing_from_source2_count']}")
    print(f"  Missing S3 from Source 3: {cross_res['gt_s3_missing_from_source3_count']}")
    print(f"  Source 2 Unmatched / Singletons: {cross_res['train_source2_unmatched_singletons']:,}")
    print(f"  Source 3 Unmatched / Singletons: {cross_res['train_source3_unmatched_singletons']:,}")
    
    total_duration = time.time() - start_time
    audit_data["total_audit_runtime_seconds"] = round(total_duration, 2)
    print(f"\nAudit completed in {total_duration:.2f} seconds.")
    
    # Save findings to JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Audit results successfully written to: {OUTPUT_JSON_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
