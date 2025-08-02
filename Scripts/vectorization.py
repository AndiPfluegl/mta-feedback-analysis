from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import numpy as np
from config import TFIDF_PARAMS, W2V_PARAMS, TRANSFORMER_MODEL
from umap import UMAP

def vectorize_tfidf(corpus: list) -> (np.ndarray, TfidfVectorizer):
    """
    Convert a list of raw text documents into TF-IDF feature vectors.

    Args:
        corpus (list of str): Preprocessed text documents.

    Returns:
        X (sparse matrix): TF-IDF feature matrix of shape (n_docs, n_terms).
        vec (TfidfVectorizer): Fitted TF-IDF vectorizer instance.
    """
    # Initialize TF-IDF vectorizer with parameters from config
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    # Fit to the corpus and transform it into a document-term matrix
    X = vec.fit_transform(corpus)
    return X, vec

def vectorize_word2vec(token_lists: list) -> (np.ndarray, Word2Vec):
    """
    Train a Word2Vec model on tokenized text and compute document embeddings
    by averaging word vectors.

    Args:
        token_lists (list of list of str): Tokenized sentences/documents.

    Returns:
        X (ndarray): Dense matrix of shape (n_docs, vector_size) where each row
                     is the mean of its token vectors.
        w2v (Word2Vec): Trained Gensim Word2Vec model.
    """
    # Train Word2Vec with parameters from config
    w2v = Word2Vec(sentences=token_lists, **W2V_PARAMS)

    def doc_vector(tokens: list) -> np.ndarray:
        """
        Compute the average Word2Vec embedding for a list of tokens.
        """
        # Collect vectors for tokens present in the vocabulary
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if vecs:
            # Average along the token dimension
            return np.mean(vecs, axis=0)
        else:
            # If no tokens are in vocab, return a zero vector
            return np.zeros(w2v.vector_size)

    # Build document-by-vector matrix
    X = np.vstack([doc_vector(toks) for toks in token_lists])
    return X, w2v

def vectorize_transformer(corpus: list, device: str = "cuda") -> (np.ndarray, SentenceTransformer):
    """
    Encode text documents into fixed-size embeddings using a pre-trained
    transformer model.

    Args:
        corpus (list of str): Raw or preprocessed text documents.
        device (str): Torch device to run the model on (e.g., "cpu" or "cuda").

    Returns:
        X_emb (ndarray): Array of shape (n_docs, embedding_dim).
        model (SentenceTransformer): Fitted sentence transformer instance.
    """
    # Load a sentence-transformer model specified in config
    model = SentenceTransformer(TRANSFORMER_MODEL, device=device)
    # Encode the corpus; show progress and use batching for efficiency
    X_emb = model.encode(
        corpus,
        show_progress_bar=True,
        device=device,
        batch_size=64
    )
    return X_emb, model

def reduce_umap(
    X_emb: np.ndarray,
    n_neighbors: int = 15,
    n_components: int = 5,
    metric: str = "cosine",
    random_state: int = 42
) -> (np.ndarray, UMAP):
    """
    Reduce high-dimensional embeddings to a lower-dimensional space using UMAP.

    Args:
        X_emb (ndarray): Input embeddings of shape (n_docs, embedding_dim).
        n_neighbors (int): Number of nearest neighbors for UMAP graph construction.
        n_components (int): Target embedding dimension.
        metric (str): Distance metric for UMAP.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_umap (ndarray): Reduced embeddings of shape (n_docs, n_components).
        umap (UMAP): Fitted UMAP instance.
    """
    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    # Fit UMAP to the embeddings and transform them
    X_umap = umap.fit_transform(X_emb)
    return X_umap, umap
