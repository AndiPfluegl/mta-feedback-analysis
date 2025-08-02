"""
Configuration constants for the customer feedback analysis pipeline:
- DATA_PATH: path to the raw CSV dataset
- SAMPLE_SIZE: number of records to sample for faster experimentation
- N_TOPICS: default number of topics to extract
- TFIDF_PARAMS: parameters controlling TF-IDF vectorization
- W2V_PARAMS: parameters for training Word2Vec embeddings
- TRANSFORMER_MODEL: HuggingFace transformer model for embeddings
"""
DATA_PATH = "../data/MTA_Customer_Feedback_Data.csv"  # CSV file location
SAMPLE_SIZE = 10000  # Limit on number of rows to load initially
N_TOPICS = 8  # Number of topics to extract in topic modeling
TFIDF_PARAMS = dict(
    max_df=0.8,     # Discard terms in >80% of docs as too common
    min_df=5,       # Discard terms in <5 docs as too rare
    ngram_range=(1,2)  # Include unigrams and bigrams
)
W2V_PARAMS = dict(
    vector_size=100,  # Dimensionality of word vectors
    window=5,         # Context window size for skip-gram
    min_count=5       # Minimum term frequency threshold
)
TRANSFORMER_MODEL = "all-mpnet-base-v2"  # Sentence-transformer model identifier