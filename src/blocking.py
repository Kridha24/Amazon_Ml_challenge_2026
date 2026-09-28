"""
Multi-pass complementary candidate generation (blocking) and recall evaluation.
Implements inverted index blocking across normalized prefixes, sorted tokens,
and locality signals, with candidate recall evaluation at multiple limits K.
"""

import re
from collections import defaultdict
from rapidfuzz import fuzz

def clean_tokens(text):
    """Normalize and tokenize text into significant lowercase words."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"['\u2019]s\b", '', text)
    text = re.sub(r'[^a-z0-9\u0900-\u097f\u00c0-\u017f]+', ' ', text)
    stopwords = {
        'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
        'co', 'company', 'and', 'the', 'of', 'in', 'near', 'at', 'on', 'by', 'services'
    }
    return [t for t in text.split() if t not in stopwords and len(t) > 1]

def extract_blocking_keys(name, addr, country):
    """
    Extract multiple complementary blocking keys:
    1. Prefix (first 2 tokens)
    2. Sorted tokens (word reordering invariant)
    3. Significant primary token
    4. Address postal/PIN code + name prefix
    """
    keys = []
    toks = clean_tokens(name)
    if not toks:
        raw_pre = re.sub(r'[^a-z0-9]+', '', name.lower())[:5]
        return [f"{country}_raw_{raw_pre}"] if raw_pre else [f"{country}_empty"]
        
    # 1. First 2 tokens
    if len(toks) >= 2:
        keys.append(f"{country}_p2_{toks[0]}_{toks[1]}")
        keys.append(f"{country}_s2_{min(toks[0], toks[1])}_{max(toks[0], toks[1])}")
    else:
        keys.append(f"{country}_p1_{toks[0]}")
        
    # 2. Significant first word
    if len(toks[0]) >= 3:
        keys.append(f"{country}_w1_{toks[0]}")
        
    # 3. Address PIN / Zip code + name initials (if available)
    if addr:
        pins = re.findall(r'\b\d{5,6}\b', addr)
        if pins:
            keys.append(f"{country}_pin_{pins[0]}_{toks[0][:2]}")
            
    return keys

class CandidateRetriever:
    """
    Inverted index retriever for business entity blocking.
    """
    def __init__(self, max_posting_size=800):
        self.index = defaultdict(list)
        self.candidate_records = {}
        self.max_posting_size = max_posting_size
        
    def build_index(self, pool_s2, pool_s3):
        """Build inverted index over candidate pool."""
        self.candidate_records.clear()
        self.index.clear()
        
        all_candidates = list(pool_s2.values()) + list(pool_s3.values())
        for cand in all_candidates:
            cid, cname, caddr, ccountry = cand[0], cand[1], cand[2], cand[3]
            self.candidate_records[cid] = cand
            keys = extract_blocking_keys(cname, caddr, ccountry)
            for k in keys:
                self.index[k].append(cid)
                
    def retrieve_candidates_for_query(self, s1_entity, max_candidates=50):
        """
        Retrieve and rank candidate records for a single Source 1 query entity.
        Returns list of (candidate_id, pre_rank_score).
        """
        s1_id = s1_entity["entity_id"]
        s1_name = s1_entity["business_name"]
        s1_addr = s1_entity["business_address"]
        s1_country = s1_entity["country"]
        
        keys = extract_blocking_keys(s1_name, s1_addr, s1_country)
        candidate_ids = set()
        
        for k in keys:
            postings = self.index.get(k, [])
            if len(postings) <= self.max_posting_size:
                candidate_ids.update(postings)
            elif "p2_" in k or "s2_" in k or "pin_" in k:
                # Always trust highly specific keys
                candidate_ids.update(postings[:self.max_posting_size])
                
        if not candidate_ids:
            return []
            
        # Fast lexical pre-ranking
        ranked = []
        for cid in candidate_ids:
            cand = self.candidate_records[cid]
            cname = cand[1]
            score = fuzz.token_sort_ratio(s1_name, cname) / 100.0
            ranked.append((cid, score))
            
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:max_candidates]

def evaluate_candidate_recall(candidate_lists, ground_truth_map, k_limits=[5, 10, 20, 30, 50, 100]):
    """
    Measure candidate recall at multiple cutoff thresholds K BEFORE matching.
    Only evaluates non-singleton entities that have true matches.
    """
    recall_by_k = {}
    candidate_counts = []
    
    total_true_matches = 0
    non_singleton_entities = 0
    
    for s1_id, cand_list in candidate_lists.items():
        candidate_counts.append(len(cand_list))
        true_matches = set(ground_truth_map.get(s1_id, []))
        if true_matches:
            non_singleton_entities += 1
            total_true_matches += len(true_matches)
            
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
    
    return {
        "non_singleton_evaluated": non_singleton_entities,
        "total_true_matches": total_true_matches,
        "average_candidates_per_entity": round(avg_candidates, 2),
        "recall_by_k": recall_by_k
    }
