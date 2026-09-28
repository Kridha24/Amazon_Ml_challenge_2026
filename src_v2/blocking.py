"""
Multi-pass complementary candidate generation (blocking) for V2.
Includes:
1. Multi-token name prefixes and sorted tokens.
2. Compact name variants (spaces/symbols/.com stripped).
3. DBA and trade-name alias decomposition.
4. Address locality, PIN, and building-number fallback.
5. Unicode-preserving Indic and accented Latin tokenization.
6. Deterministic, input-field-based candidate ranking with meaningful tie-breaking.
"""

import re
import hashlib
from collections import defaultdict
from rapidfuzz import fuzz

def clean_tokens(text):
    """Normalize and tokenize text into significant lowercase words while preserving Unicode."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"['\u2019]s\b", '', text)
    # Strip URL domains and www
    text = re.sub(r'https?://\S+|www\.\S+|\.(com|in|org|net|co|io)\b', ' ', text)
    # Strip non-alphanumeric/non-unicode symbols
    text = re.sub(r'[^\w\s]+', ' ', text)
    stopwords = {
        'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
        'co', 'company', 'and', 'the', 'of', 'in', 'near', 'at', 'on', 'by', 'services',
        'sa', 'sas', 'sarl', 'gmbh'
    }
    return [t for t in text.split() if t not in stopwords and len(t) > 1]

def extract_name_aliases(name):
    """
    Extract DBA / trade-name variants from business name.
    e.g. 'Wexfaye d/b/a One Investments' -> ['wexfaye d/b/a one investments', 'wexfaye', 'one investments']
    """
    raw_lower = name.lower()
    variants = [raw_lower]
    # Check DBA patterns
    dba_patterns = [r'\bd/?b/?a\b', r'\bt/?a\b', r'\btrading as\b', r'\bdoing business as\b', r'\bc/?o\b', r'\baka\b']
    for pat in dba_patterns:
        parts = re.split(pat, raw_lower)
        if len(parts) > 1:
            for p in parts:
                p_clean = p.strip()
                if len(p_clean) >= 3:
                    variants.append(p_clean)
            break
    return variants

def extract_compact_name(name):
    """Strip spaces, punctuation, and legal suffixes to produce a compact alphanumeric string."""
    text = name.lower()
    text = re.sub(r'\.(com|in|org|net|co|io)\b', '', text)
    text = re.sub(r'[^\w]+', '', text)
    return text

def extract_blocking_keys_v2(name, addr, country, active_routes=None):
    """
    Extract multiple complementary blocking keys based on active routes:
    - 'prefix': first 2 tokens & sorted tokens
    - 'compact': compact name prefix (spaces removed)
    - 'dba': trade-name / DBA variants
    - 'address': PIN code + building number / locality
    """
    if active_routes is None:
        active_routes = {'prefix', 'compact', 'dba', 'address'}
        
    keys = []
    
    # 1. Primary and DBA variants
    aliases = extract_name_aliases(name) if 'dba' in active_routes else [name.lower()]
    
    for alias_idx, alias in enumerate(aliases):
        toks = clean_tokens(alias)
        if not toks:
            continue
            
        is_dba = (alias_idx > 0)
        route_prefix = f"{country}_dba_" if is_dba else f"{country}_"
        
        if 'prefix' in active_routes:
            if len(toks) >= 2:
                keys.append(f"{route_prefix}p2_{toks[0]}_{toks[1]}")
                keys.append(f"{route_prefix}s2_{min(toks[0], toks[1])}_{max(toks[0], toks[1])}")
            else:
                keys.append(f"{route_prefix}p1_{toks[0]}")
                
            if len(toks[0]) >= 3:
                keys.append(f"{route_prefix}w1_{toks[0]}")
                
    # 2. Compact name route
    if 'compact' in active_routes:
        compact = extract_compact_name(name)
        if len(compact) >= 6:
            keys.append(f"{country}_cmp_{compact[:8]}")
            
    # 3. Address-based fallback routes (retaining building/postal numbers)
    if 'address' in active_routes and addr:
        addr_clean = addr.strip()
        pins = re.findall(r'\b\d{5,6}\b', addr_clean)
        # Building numbers: e.g. '6A', '1795', '1669', '3/115'
        bldg_matches = re.findall(r'\b\d+[/a-zA-Z0-9-]*\b', addr_clean)
        
        primary_tok = clean_tokens(name)
        name_init = primary_tok[0][:2] if primary_tok else "xx"
        
        if pins:
            pin = pins[0]
            # Key with PIN + name initials
            keys.append(f"{country}_pin_{pin}_{name_init}")
            
            # Key with PIN + primary building number (for cross-script matching)
            if bldg_matches:
                bldg = bldg_matches[0].lower().replace('/', '_')
                if bldg != pin:
                    keys.append(f"{country}_pb_{pin}_{bldg[:6]}")
                    
    return list(set(keys))

class CandidateRetrieverV2:
    """
    Multi-pass inverted index retriever for V2.
    """
    def __init__(self, max_posting_size=1000, active_routes=None):
        self.index = defaultdict(list)
        self.candidate_records = {}
        self.max_posting_size = max_posting_size
        self.active_routes = active_routes
        
    def build_index(self, pool_s2, pool_s3):
        """Build inverted index over candidate pool."""
        self.candidate_records.clear()
        self.index.clear()
        
        all_candidates = list(pool_s2.values()) + list(pool_s3.values())
        for cand in all_candidates:
            cid, cname, caddr, ccountry = cand[0], cand[1], cand[2], cand[3]
            self.candidate_records[cid] = cand
            keys = extract_blocking_keys_v2(cname, caddr, ccountry, self.active_routes)
            for k in keys:
                self.index[k].append(cid)
                
    def retrieve_candidates_for_query(self, s1_entity, max_candidates=50):
        """
        Retrieve and rank candidate records with deterministic tie-breaking.
        Returns list of (candidate_id, pre_rank_score).
        """
        s1_id = s1_entity["entity_id"]
        s1_name = s1_entity["business_name"]
        s1_addr = s1_entity["business_address"]
        s1_country = s1_entity["country"]
        
        keys = extract_blocking_keys_v2(s1_name, s1_addr, s1_country, self.active_routes)
        candidate_ids = set()
        
        for k in keys:
            postings = self.index.get(k, [])
            if len(postings) <= self.max_posting_size:
                candidate_ids.update(postings)
            elif "p2_" in k or "s2_" in k or "pb_" in k or "cmp_" in k:
                # Always trust highly specific keys
                candidate_ids.update(postings[:self.max_posting_size])
                
        if not candidate_ids:
            return []
            
        # Meaningful input-field ranking with deterministic tie-breaking
        ranked = []
        for cid in candidate_ids:
            cand = self.candidate_records[cid]
            cname = cand[1]
            caddr = cand[2]
            
            n_sort = fuzz.token_sort_ratio(s1_name, cname) / 100.0
            n_ratio = fuzz.ratio(s1_name, cname) / 100.0
            
            has_addr = bool(s1_addr.strip() and caddr.strip())
            a_sort = fuzz.token_sort_ratio(s1_addr, caddr) / 100.0 if has_addr else 0.0
            
            # Composite rank score
            if has_addr:
                composite = 0.70 * n_sort + 0.15 * n_ratio + 0.15 * a_sort
            else:
                composite = 0.85 * n_sort + 0.15 * n_ratio
                
            len_diff = abs(len(s1_name) - len(cname))
            
            # Deterministic tie-breaker using hash of canonical names and addresses
            det_tie_breaker = hashlib.md5(f"{cname.lower()}|{caddr.lower()}|{cid}".encode("utf-8")).hexdigest()
            
            # Sort tuple: (-composite, -n_ratio, len_diff, det_tie_breaker)
            ranked.append((cid, composite, n_ratio, len_diff, det_tie_breaker))
            
        # Sort deterministically
        ranked.sort(key=lambda x: (-round(x[1], 4), -round(x[2], 4), x[3], x[4]))
        
        return [(x[0], round(x[1], 4)) for x in ranked[:max_candidates]]

def evaluate_candidate_recall_v2(candidate_lists, ground_truth_map, k_limits=[5, 10, 20, 30, 50, 100]):
    """
    Measure candidate recall at multiple cutoff thresholds K BEFORE matching.
    Reports average and p95 candidate counts.
    """
    import numpy as np
    candidate_counts = []
    total_true_matches = 0
    non_singleton_entities = 0
    
    for s1_id, cand_list in candidate_lists.items():
        candidate_counts.append(len(cand_list))
        true_matches = set(ground_truth_map.get(s1_id, []))
        if true_matches:
            non_singleton_entities += 1
            total_true_matches += len(true_matches)
            
    recall_by_k = {}
    for k in k_limits:
        found_matches = 0
        for s1_id, cand_list in candidate_lists.items():
            true_matches = set(ground_truth_map.get(s1_id, []))
            if true_matches:
                k_cands = set([c[0] for c in cand_list[:k]])
                found_matches += len(true_matches & k_cands)
                
        rec = found_matches / total_true_matches if total_true_matches > 0 else 0.0
        recall_by_k[f"recall_at_{k}"] = round(rec, 4)
        
    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    p95_candidates = float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0
    
    return {
        "non_singleton_evaluated": non_singleton_entities,
        "total_true_matches": total_true_matches,
        "average_candidates_per_entity": round(avg_candidates, 2),
        "p95_candidates_per_entity": round(p95_candidates, 2),
        "recall_by_k": recall_by_k
    }
