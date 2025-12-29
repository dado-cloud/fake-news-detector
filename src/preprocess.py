import re
import string
import contractions
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# download required resources (safe to call multiple times)
nltk.download("stopwords", quiet=True)


# initialize reusable objects once
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """
    Clean and normalize a single news article text.

    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Expand contractions
    4. Remove stopwords
    5. Apply stemming

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned text
    """

    if not isinstance(text, str):
        return ""

    # 1. lowercase
    text = text.lower()

    # 2. remove punctuation
    text = text.translate(PUNCT_TABLE)

    # 3. expand contractions (e.g. can't -> cannot)
    text = contractions.fix(text)

    # 4. remove stopwords + non-alphabetic tokens
    tokens = [
        word for word in text.split()
        if word.isalpha() and word not in STOPWORDS
    ]

    # 5. stemming
    tokens = [STEMMER.stem(word) for word in tokens]

    return " ".join(tokens)
