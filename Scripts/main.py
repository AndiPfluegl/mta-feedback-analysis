import os
from joblib import dump, load, Parallel, delayed
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF as SKNMF
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from data import (
    load_data,
    missing_value_report,
    duplicate_report,
    row_duplicate_report,
    text_uniqueness,
    length_filter,
    token_statistics,
    spellcheck_report,
)
from preprocessing import preprocess_series, fast_language_filter
from vectorization import (
    vectorize_tfidf,
    vectorize_word2vec,
    vectorize_transformer,
    reduce_umap,
)
from bertopic import BERTopic
from config import N_TOPICS

# Directories for caching cleaned data and saving models
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "df_cleaned.pkl")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -----------------------
# Section: Data Handling
# -----------------------

def load_and_clean_data() -> 'pd.DataFrame':
    """
    Load raw data, produce quality reports, filter and preprocess text,
    cache the cleaned DataFrame to speed up subsequent runs.
    Returns:
        pd.DataFrame: Cleaned and filtered data for modeling.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # If a cached cleaned DataFrame exists, load it and skip preprocessing
    if os.path.exists(CACHE_FILE):
        print(f"Loading cleaned DataFrame from cache: {CACHE_FILE}")
        return load(CACHE_FILE)

    # Otherwise, load raw data and report basic statistics
    df = load_data()
    print(f"Loaded {len(df)} records.")

    # Print various data quality reports
    print(missing_value_report(df))                     # Missing value counts per column
    print(duplicate_report(df, subset=["Issue Detail"]))  # Exact duplicates in specific column
    print(row_duplicate_report(df))                     # Entire row duplicates
    print(text_uniqueness(df["Issue Detail"]))          # Uniqueness of text entries

    # Remove duplicate issue descriptions
    df_unique = df.drop_duplicates(subset=["Issue Detail"])
    texts = df_unique["Issue Detail"]

    # Keep only English entries and appropriate lengths
    texts = fast_language_filter(texts, allowed={"en"})
    texts = length_filter(texts, min_len=3, max_len=2000)

    # Print token-level statistics and perform a spellcheck on random samples
    print(token_statistics(texts)[0].to_string(index=False))
    print(spellcheck_report(texts, n_samples=500).to_string(index=False))

    # Subset the DataFrame to only valid indices after filtering
    df_clean = df_unique.loc[texts.index].copy()

    # Cache the cleaned DataFrame for future reuse
    dump(df_clean, CACHE_FILE)
    df_clean.to_csv(os.path.join(CACHE_DIR, "df_cleaned.csv"), index=False)
    print("Cached cleaned DataFrame.")

    return df_clean

# -----------------------
# Section: Modeling
# -----------------------

def hyperparameter_sweep(clean_texts: list):
    """
    Perform a grid search over a range of topic counts (k) for various models:
    - LDA (Latent Dirichlet Allocation)
    - NMF (Non-negative Matrix Factorization)
    - KMeans clustering on word2vec embeddings
    - Gaussian Mixture Models on word2vec embeddings

    For each k, compute coherence (for LDA/NMF) and silhouette scores (for clustering),
    and return aggregated results along with learned vectorizers and embeddings.
    """
    # Vectorize texts using TF-IDF and Word2Vec
    tfidf_X, tfidf_vec = vectorize_tfidf(clean_texts)
    w2v_X, w2v_model = vectorize_word2vec([txt.split() for txt in clean_texts])

    ks = range(2, 16)  # Evaluate k from 2 to 15
    dictionary = Dictionary([txt.split() for txt in clean_texts])
    feature_names = tfidf_vec.get_feature_names_out()

    def eval_k(k):
        # Fit each model for a given k
        lda = LatentDirichletAllocation(n_components=k, random_state=42).fit(tfidf_X)
        nmf = SKNMF(n_components=k, random_state=42).fit(tfidf_X)
        km = KMeans(n_clusters=k, random_state=42).fit(w2v_X)
        gm = GaussianMixture(n_components=k, random_state=42).fit(w2v_X)

        # Extract top-k terms per topic/component for coherence calculation
        topics_lda = [[feature_names[i] for i in comp.argsort()[-k:]] for comp in lda.components_]
        topics_nmf = [[feature_names[i] for i in comp.argsort()[-k:]] for comp in nmf.components_]

        # Compute coherence for LDA and NMF
        coh_lda = CoherenceModel(
            topics=topics_lda,
            texts=[t.split() for t in clean_texts],
            dictionary=dictionary,
            coherence='c_v',
            processes=1
        ).get_coherence()
        coh_nmf = CoherenceModel(
            topics=topics_nmf,
            texts=[t.split() for t in clean_texts],
            dictionary=dictionary,
            coherence='c_v',
            processes=1
        ).get_coherence()

        # Silhouette scores for clustering quality
        sil_km = silhouette_score(w2v_X, km.labels_)
        sil_gm = silhouette_score(w2v_X, gm.predict(w2v_X))
        inertia = km.inertia_  # KMeans inertia for elbow plot

        return k, coh_lda, coh_nmf, sil_km, sil_gm, inertia

    # Parallelize the sweep over multiple CPU cores
    results = Parallel(n_jobs=-1)(delayed(eval_k)(k) for k in ks)
    return zip(*results), tfidf_vec, w2v_model, w2v_X

# -----------------------
# Section: Reporting
# -----------------------

def plot_sweep_results(ks, lda_coh, nmf_coh, km_sil, gm_sil, inertias):
    """
    Plot coherence, silhouette scores, and KMeans inertia across different k values.
    """
    plt.figure(figsize=(12,5))

    # Plot coherence for LDA vs NMF
    plt.subplot(1,2,1)
    plt.plot(ks, lda_coh, marker='o', label='LDA')
    plt.plot(ks, nmf_coh, marker='x', label='NMF')
    plt.legend(); plt.title('Coherence vs k')

    # Plot silhouette for KMeans vs GMM
    plt.subplot(1,2,2)
    plt.plot(ks, km_sil, marker='o', label='KMeans')
    plt.plot(ks, gm_sil, marker='x', label='GMM')
    plt.legend(); plt.title('Silhouette vs k')

    plt.tight_layout(); plt.show()

    # Plot elbow curve for KMeans
    plt.figure(figsize=(6,4))
    plt.plot(ks, inertias, marker='o')
    plt.title('Elbow Plot (KMeans)'); plt.show()

# -----------------------
# Section: Topic Display with Metrics
# -----------------------

def display_final_topics_with_scores(
    lda, nmf, kmeans, gmm, w2v_model, w2v_X, clean_texts, tfidf_vec
):
    """
    Print final coherence and silhouette metrics, followed by the top N terms for each model's topics/components.
    """
    feature_names = tfidf_vec.get_feature_names_out()
    dictionary = Dictionary([t.split() for t in clean_texts])

    # Compute final coherence scores
    topics_lda = [[feature_names[i] for i in comp.argsort()[-lda.n_components:]] for comp in lda.components_]
    coh_lda = CoherenceModel(
        topics=topics_lda,
        texts=[t.split() for t in clean_texts],
        dictionary=dictionary,
        coherence='c_v',
        processes=1
    ).get_coherence()

    topics_nmf = [[feature_names[i] for i in comp.argsort()[-nmf.n_components:]] for comp in nmf.components_]
    coh_nmf = CoherenceModel(
        topics=topics_nmf,
        texts=[t.split() for t in clean_texts],
        dictionary=dictionary,
        coherence='c_v',
        processes=1
    ).get_coherence()

    # Compute silhouette scores for clustering
    sil_km = silhouette_score(w2v_X, kmeans.labels_)
    sil_gm = silhouette_score(w2v_X, gmm.predict(w2v_X))

    # Print metrics
    print(f"LDA Coherence (c_v): {coh_lda:.3f}")
    print(f"NMF Coherence (c_v): {coh_nmf:.3f}")
    print(f"KMeans Silhouette: {sil_km:.3f}")
    print(f"GMM Silhouette: {sil_gm:.3f}\n")

    # Display top N topics for each model
    print("-- LDA Topics --")
    for i, comp in enumerate(lda.components_):
        top = [feature_names[j] for j in comp.argsort()[-N_TOPICS:]][::-1]
        print(f" {i}: {', '.join(top)}")

    print("-- NMF Topics --")
    for i, comp in enumerate(nmf.components_):
        top = [feature_names[j] for j in comp.argsort()[-N_TOPICS:]][::-1]
        print(f" {i}: {', '.join(top)}")

    print("-- KMeans Clusters --")
    for i, ctr in enumerate(kmeans.cluster_centers_):
        top = [w for w, _ in w2v_model.wv.similar_by_vector(ctr, topn=N_TOPICS)]
        print(f" {i}: {', '.join(top)}")

    print("-- GMM Components --")
    for i, ctr in enumerate(gmm.means_):
        top = [w for w, _ in w2v_model.wv.similar_by_vector(ctr, topn=N_TOPICS)]
        print(f" {i}: {', '.join(top)}")

# -----------------------
# Section: BERTopic
# -----------------------

def run_and_show_bertopic(texts: list):
    """
    Fit BERTopic model, display coherence and top N topics with their counts.
    """
    model = BERTopic()
    topics, probs = model.fit_transform(texts)

    info = model.get_topic_info().head(N_TOPICS + 1)
    print(info[['Topic','Count']])
    for tid in info.Topic:
        if tid == -1:
            continue
        terms = [w for w, _ in model.get_topic(tid)][:N_TOPICS]
        print(f"{tid}: {', '.join(terms)}")

    return model

# -----------------------
# Section: Persistence
# -----------------------

def save_all(models: dict, vectorizers: dict):
    """
    Save all trained models and vectorizers to disk, using `.save` if available or joblib otherwise.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, obj in {**models, **vectorizers}.items():
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if hasattr(obj, 'save'):
            obj.save(path)
        else:
            dump(obj, path)
    print("Saved models and vectorizers.")

# -----------------------
# Main procedure
# -----------------------
def main():
    """
    Full workflow:
      1. Load and clean data
      2. Preprocess text
      3. Hyperparameter sweep
      4. Plot results
      5. Train final models with best k
      6. Display topics and metrics
      7. Run BERTopic
      8. Persist all artifacts
    """
    # Step 1: Data
    df_clean = load_and_clean_data()
    clean_texts = preprocess_series(df_clean["Issue Detail"])

    # Step 2: Grid search for best number of topics
    (ks, lda_coh, nmf_coh, km_sil, gm_sil, inertias), tfidf_vec, w2v_model, w2v_X = \
        hyperparameter_sweep(clean_texts.tolist())
    plot_sweep_results(ks, lda_coh, nmf_coh, km_sil, gm_sil, inertias)

    # Step 3: Train final models based on chosen ks
    best_k = {'lda':8,'nmf':8,'km':5,'gm':5}
    lda = LatentDirichletAllocation(n_components=best_k['lda'], random_state=42).fit(
        tfidf_vec.transform(clean_texts.tolist())
    )
    nmf = SKNMF(n_components=best_k['nmf'], random_state=42).fit(
        tfidf_vec.transform(clean_texts.tolist())
    )
    kmeans = KMeans(n_clusters=best_k['km'], random_state=42).fit(w2v_X)
    gmm = GaussianMixture(n_components=best_k['gm'], random_state=42).fit(w2v_X)

    # Step 4: Reporting
    display_final_topics_with_scores(
        lda, nmf, kmeans, gmm, w2v_model, w2v_X, clean_texts.tolist(), tfidf_vec
    )
    bt_model = run_and_show_bertopic(clean_texts.tolist())

    # Step 5: Save artifacts
    save_all(
        models={'lda':lda,'nmf':nmf,'kmeans':kmeans,'gmm':gmm,'bertopic':bt_model},
        vectorizers={'tfidf':tfidf_vec, 'w2v':w2v_model}
    )

if __name__ == "__main__":
    main()
