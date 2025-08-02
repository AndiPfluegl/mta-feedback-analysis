import pandas as pd
from config import DATA_PATH
from langdetect import detect, DetectorFactory
from spellchecker import SpellChecker
from collections import Counter

# Seed the language detector for reproducible results
DetectorFactory.seed = 42

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV file, filter down to complaints only, and remove
    empty or missing 'Issue Detail' entries.

    Args:
        path (str): File path to the CSV dataset (default from config).

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid complaints.
    """
    df = pd.read_csv(path)
    # Keep only rows labeled as complaints (case-insensitive)
    df = df[df['Commendation or Complaint'].str.lower() == 'complaint']
    # Remove rows with missing or blank issue details
    df = df[df['Issue Detail'].notna() & df['Issue Detail'].str.strip().astype(bool)]
    return df

def sample_data(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """
    Draw a reproducible random sample of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        size (int): Number of rows to sample.

    Returns:
        pd.DataFrame: Sampled subset of the original DataFrame.
    """
    return df.sample(size, random_state=42)

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the count and percentage of missing values per column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Two-column report with 'missing_count' and 'missing_pct',
                      sorted by descending percentage missing.
    """
    missing_count = df.isna().sum()
    missing_pct = missing_count / len(df) * 100
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct
    })
    return report.sort_values("missing_pct", ascending=False)

def duplicate_report(df: pd.DataFrame, subset: list) -> pd.Series:
    """
    Count exact duplicates based on a subset of columns, excluding
    the first occurrence.

    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list): List of column names to check for duplicates.

    Returns:
        pd.Series: Series with total records, duplicate count, and percent.
    """
    mask = df.duplicated(subset=subset, keep="first")
    return pd.Series({
        "total_records":     len(df),
        "duplicate_records": int(mask.sum()),
        "duplicate_pct":     float(mask.mean() * 100)
    })

def row_duplicate_report(df: pd.DataFrame) -> pd.Series:
    """
    Count fully identical row duplicates, excluding the first instance.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Series with total rows, duplicate row count, and percent.
    """
    mask = df.duplicated(keep="first")
    return pd.Series({
        "total_rows":     len(df),
        "duplicate_rows": int(mask.sum()),
        "duplicate_pct":  float(mask.mean() * 100)
    })

def text_uniqueness(series: pd.Series) -> pd.Series:
    """
    Measure how many unique text entries exist compared to total.

    Args:
        series (pd.Series): Series of text strings.

    Returns:
        pd.Series: Series with total texts, unique count, and percent unique.
    """
    total = len(series)
    unique = series.nunique()
    return pd.Series({
        "total_texts":  total,
        "unique_texts": unique,
        "unique_pct":   float(unique / total * 100)
    })

def language_filter(series: pd.Series, langs: set = {"en"}) -> pd.Series:
    """
    Detect language of each text and keep only those in the allowed set.

    Args:
        series (pd.Series): Series of text strings.
        langs (set): Allowed language codes (default {'en'}).

    Returns:
        pd.Series: Subset containing only texts detected as allowed languages.
    """
    def is_allowed(text: str) -> bool:
        try:
            return detect(text) in langs
        except:
            # If detection fails, drop the entry
            return False

    mask = series.map(is_allowed)
    print(f"→ Language filter: keeping {mask.sum()} of {len(series)} rows")
    return series[mask]

def length_filter(series: pd.Series, min_len: int = 20, max_len: int = 2000) -> pd.Series:
    """
    Remove texts that are too short or too long, likely noise.

    Args:
        series (pd.Series): Series of text strings.
        min_len (int): Minimum character length to keep.
        max_len (int): Maximum character length to keep.

    Returns:
        pd.Series: Subset of series with lengths within [min_len, max_len].
    """
    lengths = series.str.len()
    mask = (lengths >= min_len) & (lengths <= max_len)
    print(f"→ Length filter: keeping {mask.sum()} of {len(series)} rows "
          f"({min_len}-{max_len} chars)")
    return series[mask]

def spellcheck_report(series: pd.Series, n_samples: int = 1000) -> pd.DataFrame:
    """
    Perform simple spell-checking on a random sample, reporting most common errors.

    Args:
        series (pd.Series): Series of text strings.
        n_samples (int): Max number of rows to sample for spell-check.

    Returns:
        pd.DataFrame: Top-20 misspelled tokens and their counts.
    """
    sample = series.sample(min(n_samples, len(series)), random_state=42)
    sp = SpellChecker()
    misspelled = []
    for text in sample:
        # Keep only alphabetic tokens
        words = [w for w in text.split() if w.isalpha()]
        misspelled.extend(sp.unknown(words))
    freq = Counter(misspelled).most_common(20)
    return pd.DataFrame(freq, columns=["token", "count"])

def token_statistics(series: pd.Series) -> (pd.DataFrame, pd.DataFrame):
    """
    Compute basic token-level metrics:
      - Vocabulary size
      - Average tokens per document
      - Top-20 tokens by frequency

    Args:
        series (pd.Series): Series of text strings.

    Returns:
        tuple:
          - DataFrame with 'vocab_size' and 'avg_tokens_per_doc'
          - DataFrame of top-20 tokens and counts
    """
    all_tokens = []
    for text in series:
        all_tokens.extend(text.split())

    vocab = set(all_tokens)
    avg_len = sum(len(text.split()) for text in series) / len(series)
    top20 = Counter(all_tokens).most_common(20)
    df_top = pd.DataFrame(top20, columns=["token", "count"])

    metrics = pd.DataFrame({
        "vocab_size": [len(vocab)],
        "avg_tokens_per_doc": [avg_len]
    })
    return metrics, df_top
