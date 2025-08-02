import re
import ftfy
from bs4 import BeautifulSoup
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import os
import langid
from joblib import dump, load, Parallel, delayed

# Lazy-initialized NLP components
_NLP = None           # spaCy language model
_STOPS = None         # NLTK stopword set
_STEMMER = None       # Porter stemmer

def _init():
    """
    Initialize spaCy model, stopword list, and stemmer once.
    Called internally before any text processing.
    """
    global _NLP, _STOPS, _STEMMER
    if _NLP is None:
        # Load small English spaCy model without parser or NER for speed
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    if _STOPS is None:
        # Load English stopwords from NLTK
        _STOPS = set(stopwords.words("english"))
    if _STEMMER is None:
        # Initialize Porter stemmer
        _STEMMER = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Clean and normalize a single text string:
      1. Fix encoding issues and lowercase
      2. Remove URLs
      3. Strip HTML tags
      4. Remove non-letter characters
      5. Tokenize, remove stopwords, and lemmatize

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned, lemmatized text joined by spaces.
    """
    _init()
    # Fix unicode anomalies and lowercase
    text = ftfy.fix_text(text).lower()
    # Remove HTTP/S and WWW URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Keep only letters and whitespace
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = []
    for token in _NLP(text):
        # Skip spaCy stopwords and NLTK stopwords
        if token.is_stop or token.text in _STOPS:
            continue
        lemma = token.lemma_.strip()
        if lemma:
            tokens.append(lemma)
    # Return cleaned text as single string
    return " ".join(tokens)

def fast_language_filter(series: pd.Series, allowed: set = {"en"}) -> pd.Series:
    """
    Quickly detect languages in parallel using langid and keep only allowed ones.

    Args:
        series (pd.Series): Series of raw text strings.
        allowed (set): Set of language codes to retain (default {"en"}).

    Returns:
        pd.Series: Subset containing only texts detected in allowed languages.
    """
    # Determine number of parallel jobs
    n_jobs = os.cpu_count() or 1
    # Classify each text in parallel
    langs = Parallel(n_jobs=n_jobs)(
        delayed(lambda txt: langid.classify(txt)[0])(txt) for txt in series
    )
    mask = pd.Series(langs, index=series.index).isin(allowed)
    print(f"→ Language filter: keeping {mask.sum()} of {len(series)} rows")
    return series[mask]

def preprocess_series(series: pd.Series) -> pd.Series:
    """
    Apply the clean_text function to every element of the Series.

    Args:
        series (pd.Series): Series of raw text strings.

    Returns:
        pd.Series: Series of cleaned and lemmatized text strings.
    """
    return series.apply(clean_text)
