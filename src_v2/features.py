"""
Feature engineering for candidate pairs in Business Entity Resolution (V2).
Includes compact name similarity, PIN matching, numeric conflict detection,
and missing address handling without treating empty fields as positive evidence.
Strictly excludes entity IDs as features.
"""

import re
from rapidfuzz import fuzz, distance

FEATURE_NAMES_V2 = [
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_char_3gram_jaccard",
    "name_length_diff_ratio",
    "compact_name_ratio",
    "has_missing_addr",
    "both_missing_addr",
    "addr_token_jaccard",
    "addr_levenshtein_ratio",
    "pin_code_match",
    "numeric_overlap",
    "numeric_conflict",
    "is_source_3"
]

def extract_numbers(text):
    """Extract numeric sequences from text for numeric agreement check."""
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))

def extract_compact_string(text):
    """Strip spaces, punctuation, and URLs."""
    t = text.lower()
    t = re.sub(r'https?://\S+|www\.\S+|\.(com|in|org|net|co|io)\b', '', t)
    t = re.sub(r'[^\w]+', '', t)
    return t

def compute_pair_features_v2(s1_entity, cand_record):
    """
    Compute pairwise similarity features between an S1 entity and a candidate record.
    s1_entity: dict or tuple with (id, name, addr, country)
    cand_record: tuple (id, name, addr, country)
    """
    if isinstance(s1_entity, dict):
        s1_name = s1_entity["business_name"]
        s1_addr = s1_entity["business_address"]
    else:
        s1_name = s1_entity[1]
        s1_addr = s1_entity[2]
        
    c_id = cand_record[0]
    c_name = cand_record[1]
    c_addr = cand_record[2]
    
    # 1. Name features
    r_ratio = fuzz.ratio(s1_name, c_name) / 100.0
    r_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    r_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    r_jw = distance.JaroWinkler.similarity(s1_name, c_name)
    
    # Character 3-gram Jaccard
    s1_3g = {s1_name[i:i+3].lower() for i in range(len(s1_name)-2)} if len(s1_name) >= 3 else {s1_name.lower()}
    c_3g = {c_name[i:i+3].lower() for i in range(len(c_name)-2)} if len(c_name) >= 3 else {c_name.lower()}
    u_3g = len(s1_3g | c_3g)
    g3_jaccard = len(s1_3g & c_3g) / u_3g if u_3g else 0.0
    
    len_diff = abs(len(s1_name) - len(c_name)) / max(len(s1_name), len(c_name), 1)
    
    # Compact name similarity
    s1_cmp = extract_compact_string(s1_name)
    c_cmp = extract_compact_string(c_name)
    cmp_ratio = fuzz.ratio(s1_cmp, c_cmp) / 100.0
    
    # 2. Address features & missingness
    s1_addr_clean = s1_addr.strip()
    c_addr_clean = c_addr.strip()
    
    s1_empty = not bool(s1_addr_clean)
    c_empty = not bool(c_addr_clean)
    
    has_missing_addr = 1.0 if (s1_empty or c_empty) else 0.0
    both_missing_addr = 1.0 if (s1_empty and c_empty) else 0.0
    
    if s1_empty or c_empty:
        addr_jaccard = 0.0
        addr_lev = 0.0
        pin_match = 0.0
    else:
        s1_toks = set(s1_addr_clean.lower().split())
        c_toks = set(c_addr_clean.lower().split())
        u_toks = len(s1_toks | c_toks)
        addr_jaccard = len(s1_toks & c_toks) / u_toks if u_toks else 0.0
        addr_lev = fuzz.token_sort_ratio(s1_addr_clean, c_addr_clean) / 100.0
        
        # PIN code matching
        s1_pins = set(re.findall(r'\b\d{5,6}\b', s1_addr_clean))
        c_pins = set(re.findall(r'\b\d{5,6}\b', c_addr_clean))
        if s1_pins and c_pins:
            pin_match = 1.0 if bool(s1_pins & c_pins) else -1.0
        else:
            pin_match = 0.0
            
    # 3. Numeric agreement / conflict
    s1_nums = extract_numbers(s1_name + " " + s1_addr)
    c_nums = extract_numbers(c_name + " " + c_addr)
    if s1_nums and c_nums:
        num_u = len(s1_nums | c_nums)
        num_overlap = len(s1_nums & c_nums) / num_u if num_u else 0.0
        num_conflict = 1.0 if len(s1_nums & c_nums) == 0 else 0.0
    else:
        num_overlap = 0.0
        num_conflict = 0.0
        
    # 4. Source indicator (S2 vs S3)
    is_s3 = 1.0 if c_id.startswith("S3-") else 0.0
    
    return [
        r_ratio, r_sort, r_set, r_jw, g3_jaccard, len_diff,
        cmp_ratio, has_missing_addr, both_missing_addr,
        addr_jaccard, addr_lev, pin_match,
        num_overlap, num_conflict, is_s3
    ]
