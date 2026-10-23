import re
import unicodedata

# one normaliser for training wer and evaluation wer, so the numbers match.
# urdu text from different sources uses different unicode for the same
# letter (arabic vs farsi/urdu forms), which counts as a word error if
# we dont unify it first

CHAR_MAP = {
    "ي": "ی",  # arabic yeh -> farsi/urdu yeh
    "ى": "ی",  # alef maksura -> urdu yeh
    "ك": "ک",  # arabic kaf -> keheh (urdu kaf)
    "ه": "ہ",  # arabic heh -> heh goal (urdu heh)
    "ۀ": "ۂ",  # heh with yeh above, split so hamza is dropped below
    "ة": "ۃ",  # teh marbuta -> urdu teh marbuta goal
    "ە": "ہ",  # ae -> heh goal
}

# urdu (extended arabic-indic) and arabic-indic digits -> 0-9
DIGITS = {chr(0x06F0 + i): str(i) for i in range(10)}
DIGITS.update({chr(0x0660 + i): str(i) for i in range(10)})

# harakat, superscript alef, hamza above/below, tatweel - spelling marks
# that are written inconsistently in transcripts
DIACRITICS_RE = re.compile("[ً-ٰٟـۖ-ۭ]")

# zero width joiners show up in urdu text and are invisible
ZERO_WIDTH_RE = re.compile("[​-‏⁠﻿]")


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = ZERO_WIDTH_RE.sub("", text)
    text = "".join(CHAR_MAP.get(ch, ch) for ch in text)
    text = "".join(DIGITS.get(ch, ch) for ch in text)
    text = DIACRITICS_RE.sub("", text)
    text = text.lower()
    # any unicode punctuation or symbol (english and urdu: ، ۔ ؟ etc) -> space
    text = "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch
        for ch in text
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()
