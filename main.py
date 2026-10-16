import threading
from enum import Enum

import msgspec
import numpy as np

from panoptic.core.databases.data.models import DataCommit, Property, Sha1Value, Tag
from panoptic.core.databases.entity_schema import OP_CREATE
from panoptic.core.databases.media.models import Map, VectorType
from panoptic.core.plugin.plugin import APlugin
from panoptic.models.action_models import (
    ActionContext, ActionResult, Group, Notif, NotifType, OwnVectorType, PropertyId, Score, ScoreList,
)

from . import clustering
from .compute_task import ComputeTextVectorsTask, TextHashes, text_hash
from .embedder import EmbedderManager


class TextModel(Enum):
    multilingual_minilm = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    multilingual_mpnet = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    multilingual_e5 = 'intfloat/multilingual-e5-base'
    bge_m3 = 'BAAI/bge-m3'


class Analyzer(Enum):
    words = 'word'
    characters = 'char'


class PanopticText(APlugin):
    """
    Clustering, similarity, search and maps on a text property.
    Text embeddings are stored as Panoptic vector types; TF-IDF clustering needs no model.
    """

    def __init__(self, name: str, project, plugin_path: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.embedders = EmbedderManager()
        self.hashes = TextHashes(self.data_path / 'text_hashes.db')
        self._property_lock = threading.Lock()

        self.add_action_easy(self.create_vector_type, ['vector_type'])
        self.add_action_easy(self.compute_vectors, ['vector'])
        self.add_action_easy(self.cluster_text_syntax, ['group'])
        self.add_action_easy(self.cluster_text_semantic, ['group'])
        self.add_action_easy(self.find_similar_texts, ['similar'])
        self.add_action_easy(self.search_text, ['text_search', 'execute'])
        self.add_action_easy(self.text_map, ['map', 'execute'])

    def _stop(self) -> None:
        self.embedders.clear()

    # ------------------------------------------------------------------
    # Vectors
    # ------------------------------------------------------------------

    def create_vector_type(self, context: ActionContext, text_property: PropertyId,
                           model: TextModel = TextModel.multilingual_minilm,
                           custom_model: str = '') -> ActionResult:
        """Create text embeddings of a text property.
        @text_property: the text property to embed
        @model: multilingual embedding model (MiniLM: fast; mpnet / e5: better; bge-m3: long texts, slow)
        @custom_model: HuggingFace identifier of another model, overrides the model above
        """
        prop = self._get_text_property(text_property)
        if prop is None:
            return _error('NotATextProperty', 'Choose a text property')
        model_name = custom_model.strip() or model.value
        vt = self.project.upsert_vector_type(VectorType(id=-1, source=self.name, params={
            'model': model_name, 'property': prop.id, 'property_name': prop.name,
        }))
        self.vector_types.append(vt)
        return ActionResult(value=msgspec.structs.asdict(vt))

    def compute_vectors(self, context: ActionContext, vec_type: OwnVectorType) -> ActionResult:
        """Compute the missing or outdated text embeddings.
        @vec_type: the text vector type to compute
        """
        texts = self._texts(vec_type.params['property'], self._selected_sha1s(context))
        task = self._enqueue(vec_type, self._outdated(vec_type, texts))
        if task is None:
            return ActionResult(notifs=[Notif(NotifType.INFO, name='UpToDate',
                                              message='All text vectors are up to date')])
        return ActionResult(task_id=task.id)

    def _enqueue(self, vec_type: VectorType, texts: dict[str, str]) -> ComputeTextVectorsTask | None:
        if not texts:
            return None
        return self.project.add_task(ComputeTextVectorsTask(self, vec_type, texts))

    def _outdated(self, vec_type: VectorType, texts: dict[str, str]) -> dict[str, str]:
        stored = self.hashes.get(vec_type.id)
        return {s: t for s, t in texts.items() if stored.get(s) != text_hash(t)}

    def _load_vectors(self, vec_type: VectorType, texts: dict[str, str]):
        """(sha1s, matrix) of up-to-date vectors, or an ActionResult launching their computation."""
        outdated = self._outdated(vec_type, texts)
        if outdated:
            task = self._enqueue(vec_type, outdated)
            return ActionResult(task_id=task.id, notifs=[Notif(
                NotifType.WARNING, name='VectorsComputing',
                message=f'Computing {len(outdated)} missing or outdated text vectors, run again once done',
            )])
        vectors = self.project.get_vectors(vec_type.id, sha1s=list(texts))
        if not vectors:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [v.sha1 for v in vectors], np.stack([v.data for v in vectors]).astype(np.float32)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_text_syntax(self, context: ActionContext, text_property: PropertyId, nb_clusters: int = 10,
                            analyzer: Analyzer = Analyzer.words,
                            save_as_property: bool = False) -> ActionResult:
        """Cluster on shared vocabulary: TF-IDF + LSA + k-means, fast and model-free.
        @text_property: the text property to cluster on
        @nb_clusters: number of clusters
        @analyzer: words, or character n-grams (robust to OCR errors and spelling variants)
        @save_as_property: store each image's cluster as a tag property
        """
        prop = self._get_text_property(text_property)
        if prop is None:
            return _error('NotATextProperty', 'Choose a text property')
        texts = self._texts(prop.id, self._selected_sha1s(context))
        if not texts:
            return _error('NoText', f'No text found in {prop.name}')
        sha1s, docs = list(texts), list(texts.values())
        vectors = clustering.syntax_vectors(docs, analyzer.value)
        labels = clustering.kmeans_labels(vectors, nb_clusters)
        return self._clusters_result(prop, sha1s, docs, vectors, labels, save_as_property)

    def cluster_text_semantic(self, context: ActionContext, vec_type: OwnVectorType, nb_clusters: int = 0,
                              min_cluster_size: int = 5, save_as_property: bool = False) -> ActionResult:
        """Cluster on meaning with text embeddings, groups named by their keywords.
        @vec_type: the text vector type to use
        @nb_clusters: number of clusters (k-means); 0 finds it automatically (UMAP + HDBSCAN, leaves outliers apart)
        @min_cluster_size: smallest cluster when the number of clusters is automatic
        @save_as_property: store each image's cluster as a tag property
        """
        prop = self._get_text_property(vec_type.params['property'])
        if prop is None:
            return _error('NotATextProperty', 'The text property of this vector type was deleted')
        texts = self._texts(prop.id, self._selected_sha1s(context))
        if not texts:
            return _error('NoText', f'No text found in {prop.name}')
        loaded = self._load_vectors(vec_type, texts)
        if isinstance(loaded, ActionResult):
            return loaded
        sha1s, vectors = loaded
        if nb_clusters > 0:
            labels = clustering.kmeans_labels(vectors, nb_clusters)
        else:
            labels = clustering.density_labels(vectors, min_cluster_size)
        docs = [texts[s] for s in sha1s]
        return self._clusters_result(prop, sha1s, docs, vectors, labels, save_as_property)

    def _clusters_result(self, prop: Property, sha1s: list[str], docs: list[str], vectors: np.ndarray,
                         labels: np.ndarray, save_as_property: bool) -> ActionResult:
        words = clustering.keywords(docs, labels)
        scores = clustering.cohesion(vectors, labels)
        clusters = sorted(set(labels.tolist()) - {clustering.NOISE}, key=lambda c: -np.sum(labels == c))
        names = _unique_names([', '.join(words.get(c, [])) or f'Cluster {i}' for i, c in enumerate(clusters)])
        sha1_array = np.asarray(sha1s)
        groups = [Group(sha1s=sha1_array[labels == c].tolist(), name=name,
                        score=Score(value=round(scores[c], 3), min=0, max=1, max_is_best=True,
                                    description='Mean similarity to the cluster centre'))
                  for c, name in zip(clusters, names)]
        if np.any(labels == clustering.NOISE):
            groups.append(Group(sha1s=sha1_array[labels == clustering.NOISE].tolist(), name='Outliers'))
        if save_as_property:
            label_names = dict(zip(clusters, names))
            self._save_clusters(prop, {s: label_names.get(l) for s, l in zip(sha1s, labels.tolist())})
        return ActionResult(groups=groups)

    def _save_clusters(self, source: Property, sha1_names: dict[str, str | None]) -> None:
        name = f'{source.name} clusters'
        with self._property_lock:
            prop = next((p for p in self.project.get_properties(name=name) if p.dtype == 'tag'), None) \
                or self.project.add_property(name, 'tag', 'sha1')
            tags = {t.value: t for t in self.project.get_tags(list_id=prop.tag_list_id)}
            new = [Tag(id=-1, list_id=prop.tag_list_id, value=v, parents=[], color=i % 12, operation=OP_CREATE)
                   for i, v in enumerate(dict.fromkeys(n for n in sha1_names.values() if n and n not in tags))]
            if new:
                with self.project._project_db() as pdb:
                    pdb.allocate_tags(new)
                tags.update({t.value: t for t in new})
            self.project.apply_commit(DataCommit(tags=new, sha1_values=[
                Sha1Value(property_id=prop.id, sha1=s, value=[tags[n].id] if n else None)
                for s, n in sha1_names.items()
            ]))

    # ------------------------------------------------------------------
    # Similarity, search, map
    # ------------------------------------------------------------------

    def find_similar_texts(self, context: ActionContext, vec_type: OwnVectorType,
                           max_results: int = 200) -> ActionResult:
        """Find the images whose text is closest in meaning to the selected ones.
        @vec_type: the text vector type to use
        @max_results: maximum number of results
        """
        selected = self._selected_sha1s(context) if context.instance_ids else []
        texts = self._texts(vec_type.params['property'])
        loaded = self._load_vectors(vec_type, texts)
        if isinstance(loaded, ActionResult):
            return loaded
        sha1s, matrix = loaded
        index = {s: i for i, s in enumerate(sha1s)}
        rows = [index[s] for s in selected if s in index]
        if not rows:
            return _error('NoText', 'The selected images have no text')
        query = matrix[rows].mean(0)
        return self._similarity_group(sha1s, matrix, query / np.linalg.norm(query), max_results, 'Similar texts')

    def search_text(self, context: ActionContext, vec_type: OwnVectorType, text: str = '',
                    min_similarity: float = 0.3) -> ActionResult:
        """Search images whose text matches a query in meaning, not only in words.
        @vec_type: the text vector type to search in
        @text: the query
        @min_similarity: minimum cosine similarity (0-1)
        """
        if not text.strip():
            return _error('EmptySearchText', 'Please provide a non-empty search text')
        texts = self._texts(vec_type.params['property'], self._selected_sha1s(context))
        loaded = self._load_vectors(vec_type, texts)
        if isinstance(loaded, ActionResult):
            return loaded
        sha1s, matrix = loaded
        if not sha1s:
            return _error('NoText', 'No text to search in')
        query = self.embedders.get(vec_type.params['model']).embed([text], is_query=True)[0]
        return self._similarity_group(sha1s, matrix, query, len(sha1s), f'Text Search: {text}', min_similarity)

    @staticmethod
    def _similarity_group(sha1s, matrix, query, max_results, name, min_similarity=None) -> ActionResult:
        idx, sims = clustering.top_similar(matrix, query, max_results)
        if min_similarity is not None:
            keep = sims >= min_similarity
            idx, sims = idx[keep], sims[keep]
        return ActionResult(groups=[Group(
            sha1s=[sha1s[i] for i in idx], name=name,
            scores=ScoreList(min=0, max=1, max_is_best=True, values=np.clip(sims, 0, 1).round(3).tolist(),
                             description='Cosine similarity of the texts'),
        )])

    def text_map(self, context: ActionContext, vec_type: OwnVectorType, map_name: str = '') -> ActionResult:
        """Map the images by the meaning of their text (UMAP).
        @vec_type: the text vector type to use
        @map_name: name of the map
        """
        texts = self._texts(vec_type.params['property'], self._selected_sha1s(context))
        loaded = self._load_vectors(vec_type, texts)
        if isinstance(loaded, ActionResult):
            return loaded
        sha1s, matrix = loaded
        if len(sha1s) < 4:
            return _error('NotEnoughData', 'A map needs at least 4 images with text')
        coords = clustering.map_coordinates(matrix)
        coords = coords - coords.mean(0)
        coords *= 100 / max(np.linalg.norm(coords, axis=1).max(), 1e-12)
        data = [x for s, (cx, cy) in zip(sha1s, coords.tolist()) for x in (s, cx, cy)]
        name = map_name or f"Text {vec_type.params.get('property_name', '')}"
        point_map = self.project.upsert_map(Map(id=-1, source=self.name, name=name, key='sha1',
                                                count=len(sha1s), data=data))
        return ActionResult(value=msgspec.structs.asdict(point_map))

    # ------------------------------------------------------------------
    # Texts
    # ------------------------------------------------------------------

    def _get_text_property(self, property_id: int) -> Property | None:
        props = self.project.get_properties(id=property_id)
        return props[0] if props and props[0].dtype == 'text' else None

    def _selected_sha1s(self, context: ActionContext) -> list[str] | None:
        """sha1s of the selection, None for the whole project."""
        if not context.instance_ids:
            return None
        return list(dict.fromkeys(i.sha1 for i in self.project.get_instances(id=context.instance_ids)))

    def _texts(self, property_id: int, sha1s: list[str] | None = None) -> dict[str, str]:
        """Non-empty text per sha1. Instance or file values sharing an image are joined."""
        prop = self._get_text_property(property_id)
        if prop is None:
            return {}
        per_sha1: dict[str, dict[str, None]] = {}

        def add(sha1, value):
            if sha1 and isinstance(value, str) and value.strip():
                per_sha1.setdefault(sha1, {})[value.strip()] = None

        if prop.mode == 'sha1':
            for v in self.project.get_sha1_values(property_id=prop.id, **_filter('sha1', sha1s)):
                add(v.sha1, v.value)
        else:
            instances = self.project.get_instances(**_filter('sha1', sha1s))
            if prop.mode == 'file':
                values = {v.file_id: v.value for v in self.project.get_file_values(property_id=prop.id)}
                for i in instances:
                    add(i.sha1, values.get(i.file_id))
            else:
                values = {v.instance_id: v.value for v in self.project.get_instance_values(property_id=prop.id)}
                for i in instances:
                    add(i.sha1, values.get(i.id))
        return {s: '\n'.join(v) for s, v in per_sha1.items()}


def _filter(key: str, values: list | None) -> dict:
    return {} if values is None else {key: values}


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    res = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        res.append(n if seen[n] == 1 else f'{n} ({seen[n]})')
    return res


def _error(name: str, message: str) -> ActionResult:
    return ActionResult(notifs=[Notif(NotifType.ERROR, name=name, message=message)])
