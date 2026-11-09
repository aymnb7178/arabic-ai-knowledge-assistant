"""Arabic-aware text normalisation and tokenisation (stdlib only)."""

from __future__ import annotations

import re
import unicodedata

# Tashkeel (harakat), Quranic annotation marks and the dagger alef.
_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"

_LETTER_MAP = {
    # alef variants -> bare alef
    "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا",
    # alef maksura -> yaa
    "ى": "ي",
    # taa marbuta -> haa
    "ة": "ه",
    # hamza on waw / yaa -> carrier letter
    "ؤ": "و", "ئ": "ي",
}
# Arabic-Indic and Eastern Arabic-Indic digits -> ASCII digits.
_DIGIT_MAP = {0x0660 + i: str(i) for i in range(10)}
_DIGIT_MAP.update({0x06F0 + i: str(i) for i in range(10)})
_TABLE = str.maketrans({**{ord(k): v for k, v in _LETTER_MAP.items()}, **_DIGIT_MAP})

_TOKEN = re.compile(r"[^\W_]+")
_ARABIC_CHAR = re.compile("[؀-ۿ]")


def normalize_text(text: str) -> str:
    """Normalise Arabic orthography and fold case; safe to apply to English too."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_TABLE).casefold()
    return re.sub(r"\s+", " ", text).strip()


def arabic_ratio(text: str) -> float:
    """Share of letters that are Arabic-script (0.0 for empty text)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _ARABIC_CHAR.match(c)) / len(letters)


def is_arabic(text: str) -> bool:
    return arabic_ratio(text) >= 0.5


_RAW_STOPWORDS = """
the a an of to in on for and or is are was were be been with by as at from that this it its
what which who how why when where do does can should i you we they your
في من على إلى الى عن أن ان إن ما ماذا هل هو هي هذا هذه ذلك تلك التي الذي كل أو او مع كيف لماذا متى أين اين
هم هن نحن أنا انا أي اي لا لم لن قد كان كانت يكون تكون ثم بين حتى عند بعد قبل
"""
STOPWORDS = frozenset(normalize_text(w) for w in _RAW_STOPWORDS.split())

_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_AR_SUFFIXES = ("ات", "ون", "ين", "ها")


def light_stem(token: str) -> str:
    """Very light stemmer: Arabic definite-article prefixes / plural suffixes, English plural -s."""
    if _ARABIC_CHAR.search(token):
        for p in _AR_PREFIXES:
            if token.startswith(p) and len(token) - len(p) >= 3:
                token = token[len(p):]
                break
        for s in _AR_SUFFIXES:
            if token.endswith(s) and len(token) - len(s) >= 3:
                token = token[: -len(s)]
                break
        return token
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, *, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Normalise, split into word tokens, drop stopwords, lightly stem."""
    out = []
    for tok in _TOKEN.findall(normalize_text(text)):
        if drop_stopwords and tok in STOPWORDS:
            continue
        out.append(light_stem(tok) if stem else tok)
    return out
