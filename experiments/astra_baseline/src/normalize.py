import pandas as pd
import re
import unicodedata

def unicode_normalize(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def remove_punctuation(text):
    if not text: return ""
    return re.sub(r'[^\w\s]', ' ', text)

def normalize_whitespace(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def normalize_legal_suffixes(text):
    if not text: return ""
    text = f" {text} "
    suffixes = [
        (" pvt ltd ", " private limited "),
        (" pvt ", " private "),
        (" ltd ", " limited "),
        (" inc ", " incorporated "),
        (" corp ", " corporation "),
        (" co ", " company "),
        (" llc ", " limited liability company ")
    ]
    for k, v in suffixes:
        text = text.replace(k, v)
    return text.strip()

def extract_numeric(text):
    if not text: return ""
    nums = re.findall(r'\d+', text)
    return " ".join(nums)

def get_name_representations(name):
    raw = name if not pd.isna(name) else ""
    lowercase = str(raw).lower() if raw else ""
    unicode_norm = unicode_normalize(lowercase)
    punct_norm = remove_punctuation(unicode_norm)
    ws_norm = normalize_whitespace(punct_norm)
    legal_norm = normalize_legal_suffixes(ws_norm)
    
    tokens = ws_norm.split()
    sorted_tokens = " ".join(sorted(tokens))
    compact = re.sub(r'\s+', '', ws_norm)
    
    # 3-grams
    if len(compact) >= 3:
        ngrams = [compact[i:i+3] for i in range(len(compact)-2)]
    else:
        ngrams = [compact] if compact else []
        
    return {
        "raw": raw,
        "lowercase": lowercase,
        "unicode_normalized": unicode_norm,
        "punctuation_normalized": punct_norm,
        "whitespace_normalized": ws_norm,
        "legal_suffix_normalized": legal_norm,
        "tokens": tokens,
        "sorted_tokens": sorted_tokens,
        "compact": compact,
        "char_ngrams": ngrams
    }

def get_address_representations(address):
    raw = address if not pd.isna(address) else ""
    lowercase = str(raw).lower() if raw else ""
    unicode_norm = unicode_normalize(lowercase)
    punct_norm = remove_punctuation(unicode_norm)
    ws_norm = normalize_whitespace(punct_norm)
    
    tokens = ws_norm.split()
    numeric_tokens = extract_numeric(ws_norm)
    compact = re.sub(r'\s+', '', ws_norm)
    
    if len(compact) >= 3:
        ngrams = [compact[i:i+3] for i in range(len(compact)-2)]
    else:
        ngrams = [compact] if compact else []
        
    return {
        "raw": raw,
        "lowercase": lowercase,
        "unicode_normalized": unicode_norm,
        "punctuation_normalized": punct_norm,
        "whitespace_normalized": ws_norm,
        "tokens": tokens,
        "numeric_tokens": numeric_tokens.split(),
        "compact": compact,
        "char_ngrams": ngrams
    }

def normalize_df(df):
    def norm(x):
        if pd.isna(x): return ""
        return normalize_whitespace(remove_punctuation(unicode_normalize(str(x).lower())))
    
    df['norm_name'] = df['business_name'].apply(norm)
    df['norm_address'] = df['business_address'].apply(norm)
    return df

if __name__ == "__main__":
    name = "Amazon.com, Inc. (Seattle)"
    print(get_name_representations(name))
    
    addr = "123 Main St, Apt 4B, New York, NY 10001!"
    print(get_address_representations(addr))
