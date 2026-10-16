# PanopticText

Panoptic plugin to cluster, compare, search and map images by a text property (OCR, transcriptions, captions...).

# Installation

From Panoptic's plugin settings, with the repository URL `https://github.com/Tyrannas/PanopticText`.

# Functions

## Text embeddings
Text embeddings are stored as Panoptic vector types (Vector settings), one per text property and model:
- `create_vector_type`: choose the text property and a multilingual model (`MiniLM`: fast, `mpnet` / `e5`: better, `bge-m3`: long texts) or any HuggingFace model.
- `compute_vectors`: computes the missing vectors in a background task. Only edited texts are computed again.

Long texts are split into windows of the model's length and averaged. Images sharing a sha1 share one text: their instance values are joined.

## Clusters
- `cluster_text_syntax`: shared vocabulary, model-free. TF-IDF (words, or character n-grams for noisy OCR) reduced by LSA, then k-means.
- `cluster_text_semantic`: meaning, with text embeddings. A number of clusters runs k-means; `0` finds it automatically (UMAP + HDBSCAN, as BERTopic) and leaves outliers apart.

Groups are named by their keywords (class-based TF-IDF) and scored by their cohesion. With `save_as_property`, each image gets its cluster in the tag property `<property> clusters`.

## Similarity, search, map
- `find_similar_texts`: images whose text is closest to the selected ones.
- `search_text`: semantic search, also available as a mode of the search bar.
- `text_map`: 2D map of the texts (UMAP).

# Tests

```
PYTHONPATH=<Panoptic>/panoptic_back python -m pytest tests
```
The real-model tests run only if `paraphrase-multilingual-MiniLM-L12-v2` is in the HuggingFace cache.
