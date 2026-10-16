import re

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .stopwords import FRENCH_STOP_WORDS

NOISE = -1
_TOKEN = r'(?u)\b\w\w+\b'
# Split like the vectorizers' tokenizer, so entries such as "c'est" or "au-delà" match
STOP_WORDS = sorted({t for w in FRENCH_STOP_WORDS for t in re.findall(_TOKEN, w.lower())} | ENGLISH_STOP_WORDS)


def syntax_vectors(texts: list[str], analyzer: str = 'word') -> np.ndarray:
    """TF-IDF reduced by LSA (TruncatedSVD), L2-normalised: dense, low-dim, suited to k-means."""
    n = len(texts)
    if analyzer == 'char':
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True,
                                     min_df=2 if n >= 50 else 1, max_features=100_000)
    else:
        vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True,
                                     min_df=2 if n >= 50 else 1, max_df=0.9 if n >= 20 else 1.0,
                                     max_features=100_000)
    tfidf = vectorizer.fit_transform(texts)
    components = min(100, tfidf.shape[1] - 1, n - 1)
    if components < 2:
        return normalize(tfidf.toarray())
    return normalize(TruncatedSVD(components, random_state=0).fit_transform(tfidf))


def kmeans_labels(vectors: np.ndarray, nb_clusters: int) -> np.ndarray:
    k = max(1, min(nb_clusters, len(vectors)))
    # A single k-means++ run often merges two topics on sparse text vectors
    n_init = 10 if len(vectors) <= 20_000 else 3
    return KMeans(k, n_init=n_init, random_state=0).fit_predict(vectors)


def density_labels(vectors: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """Automatic number of clusters (BERTopic recipe): UMAP to 5 dims, then HDBSCAN; -1 is noise."""
    n = len(vectors)
    min_cluster_size = max(2, min_cluster_size)
    if n < 2 * min_cluster_size:
        return np.zeros(n, dtype=int)
    reduced = _reduce(vectors, 5, fallback_dims=50)
    return HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(reduced)


def _reduce(vectors: np.ndarray, dims: int, fallback_dims: int = None) -> np.ndarray:
    n = len(vectors)
    try:
        import umap
        return umap.UMAP(n_components=dims, n_neighbors=min(15, n - 1), min_dist=0.0,
                         metric='cosine', random_state=0, n_jobs=1).fit_transform(vectors)
    except ImportError:
        return PCA(min(fallback_dims or dims, n - 1, vectors.shape[1]), random_state=0).fit_transform(vectors)


def keywords(texts: list[str], labels: np.ndarray, top_n: int = 5) -> dict[int, list[str]]:
    """Most representative words of each cluster with class-based TF-IDF (as BERTopic)."""
    clusters = sorted(c for c in set(labels.tolist()) if c != NOISE)
    if not clusters:
        return {}
    docs = [' '.join(t for t, l in zip(texts, labels) if l == c) for c in clusters]
    try:
        counts = CountVectorizer(stop_words=STOP_WORDS).fit(docs)
    except ValueError:  # only stop words
        return {c: [] for c in clusters}
    tf = counts.transform(docs).toarray().astype(float)
    words = counts.get_feature_names_out()
    tf /= np.maximum(tf.sum(1, keepdims=True), 1)
    avg = tf.sum() / len(clusters)
    idf = np.log(1 + avg / np.maximum(tf.sum(0), 1e-12))
    scores = tf * idf
    if len(clusters) > 2:
        scores[:, (tf > 0).all(0)] = 0  # words shared by every cluster name none of them
    return {c: [words[i] for i in np.argsort(-scores[row])[:top_n] if scores[row, i] > 0]
            for row, c in enumerate(clusters)}


def cohesion(vectors: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Mean cosine similarity of each cluster's members to their centroid (vectors are normalised)."""
    res = {}
    for c in set(labels.tolist()) - {NOISE}:
        members = vectors[labels == c]
        centroid = normalize(members.mean(0, keepdims=True))[0]
        res[c] = float((members @ centroid).mean())
    return res


def top_similar(matrix: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = matrix @ query
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def map_coordinates(vectors: np.ndarray) -> np.ndarray:
    return _reduce(vectors, 2)
