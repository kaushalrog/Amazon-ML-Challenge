"""Normalization v2.

Fixes the fatal defect in ``normalize.py``: NFKD + ``encode('ascii','ignore')``
deletes every non-Latin script outright, erasing ~9% of Source-2 and ~5% of
Source-3 business names (and ~6.7% of all true-match rows) before matching can
see them.  Source 1 is 100% Latin, so those records became unmatchable by name.

Here we instead (a) fold Latin diacritics only, and (b) romanize the nine Indic
scripts present in the data with a single table.  The Indic Unicode blocks are
ISCII-aligned -- Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and
Malayalam sit at fixed offsets parallel to Devanagari -- so one Devanagari table
plus a block offset romanizes all of them.  The result is not scholarly
transliteration; it only has to be consistent enough for fuzzy comparison.
"""

import re
import unicodedata

VIRAMA = "्"

_CONSONANTS = {
    "क": "k",  "ख": "kh", "ग": "g",  "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh","ज": "j",  "झ": "jh", "ञ": "n",
    "ट": "t",  "ठ": "th", "ड": "d",  "ढ": "dh", "ण": "n",
    "त": "t",  "थ": "th", "द": "d",  "ध": "dh", "न": "n",
    "ऩ": "n",  "प": "p",  "फ": "ph", "ब": "b",  "भ": "bh",
    "म": "m",  "य": "y",  "र": "r",  "ऱ": "r",  "ल": "l",
    "ळ": "l",  "ऴ": "l",  "व": "v",  "श": "sh", "ष": "sh",
    "स": "s",  "ह": "h",  "य़": "y",
}

_VOWELS = {
    "अ": "a",  "आ": "aa", "इ": "i",  "ई": "ee", "उ": "u",
    "ऊ": "oo", "ऋ": "ri", "ए": "e",  "ऐ": "ai", "ओ": "o",
    "औ": "au", "ऍ": "e",  "ऑ": "o",  "ऒ": "o",  "ऎ": "e",
}

# Dependent vowel signs (matras) replace the inherent "a" of the preceding consonant.
_MATRAS = {
    "ा": "aa", "ि": "i",  "ी": "ee", "ु": "u",  "ू": "oo",
    "ृ": "ri", "े": "e",  "ै": "ai", "ो": "o",  "ौ": "au",
    "ॅ": "e",  "ॉ": "o",  "ॊ": "o",  "ॆ": "e",
}

# Anusvara / chandrabindu / visarga.
_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}

_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

# Indic block bases that are ISCII-aligned with Devanagari (0x0900).
_BLOCK_BASES = (0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)


def _to_devanagari(ch):
    """Map a character in any ISCII-aligned Indic block onto its Devanagari twin."""
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F:
        return ch
    for base in _BLOCK_BASES:
        if base <= cp <= base + 0x7F:
            return chr(cp - base + 0x0900)
    return ch


_HAS_INDIC = re.compile(r"[ऀ-ൿ]")


def indic_to_latin(text):
    """Romanize any Indic text in ``text``; non-Indic characters pass through."""
    if not text or not _HAS_INDIC.search(text):
        return text
    out = []
    for raw in text:
        ch = _to_devanagari(raw)
        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            out.append("a")          # inherent vowel, possibly overridden below
        elif ch in _MATRAS:
            if out and out[-1] == "a":
                out[-1] = _MATRAS[ch]
            else:
                out.append(_MATRAS[ch])
        elif ch == VIRAMA:
            if out and out[-1] == "a":
                out.pop()
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _SIGNS:
            out.append(_SIGNS[ch])
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
        elif ch in ("़", "‌", "‍"):   # nukta, ZWNJ, ZWJ
            continue
        else:
            out.append(raw)
    return "".join(out)


def fold_diacritics(text):
    """Strip Latin combining marks (cafe<-café) without deleting whole scripts."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


_NON_ALNUM = re.compile(r"[^0-9a-zऀ-ൿ]+")
_WS = re.compile(r"\s+")


def basic_norm(text, romanize=True):
    """Lowercase, romanize Indic text, fold diacritics, reduce to alnum tokens."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    if romanize:
        t = indic_to_latin(t)
    t = fold_diacritics(t).lower()
    t = _NON_ALNUM.sub(" ", t)
    return _WS.sub(" ", t).strip()


if __name__ == "__main__":
    samples = [
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "आदित्य प्रॉपर्टीज एलएलपी",
        "ஸ்ரீ டெக்னாலஜிஸ்",
        "ಶ್ರೀ ಎಂಟರ್‌ಪ್ರೈಸಸ್",
        "<< Team Ecole",
        "175 Boulevard du Président Franklin Roosevelt, Bordeaux",
    ]
    for s in samples:
        print(f"{s!r}\n  -> {basic_norm(s)!r}")
