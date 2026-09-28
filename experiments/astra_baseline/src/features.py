import pandas as pd
import difflib

def str_similarity(a, b):
    if pd.isna(a) or pd.isna(b) or not a or not b: return 0.0
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()

def token_jaccard(a, b):
    if pd.isna(a) or pd.isna(b) or not a or not b: return 0.0
    t1 = set(str(a).split())
    t2 = set(str(b).split())
    if not t1 and not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def extract_features(s1_row, c_row):
    features = {}
    
    # Names
    n1 = s1_row.get('norm_name', '')
    n2 = c_row.get('norm_name', '')
    features['name_exact'] = int(n1 == n2) if n1 and n2 else 0
    features['name_sim'] = str_similarity(n1, n2)
    features['name_jaccard'] = token_jaccard(n1, n2)
    features['name_len_diff'] = abs(len(n1) - len(n2)) if n1 and n2 else -1
    
    # Addresses
    a1 = s1_row.get('norm_address', '')
    a2 = c_row.get('norm_address', '')
    features['addr_exact'] = int(a1 == a2) if a1 and a2 else 0
    features['addr_sim'] = str_similarity(a1, a2)
    features['addr_jaccard'] = token_jaccard(a1, a2)
    features['addr_len_diff'] = abs(len(a1) - len(a2)) if a1 and a2 else -1
    
    # Countries
    c1 = s1_row.get('country', '')
    c2 = c_row.get('country', '')
    features['country_exact'] = int(c1.lower() == c2.lower()) if c1 and c2 and not pd.isna(c1) and not pd.isna(c2) else 0
    
    # Cross features
    features['both_exact'] = features['name_exact'] * features['addr_exact']
    
    # Missingness
    features['missing_name'] = int(not n1 or not n2)
    features['missing_addr'] = int(not a1 or not a2)
    
    return features
