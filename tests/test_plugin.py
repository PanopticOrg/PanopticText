"""PanopticText inside a real Panoptic project.

Run from the plugin folder, with panoptic importable:
    PYTHONPATH=<Panoptic>/panoptic_back python -m pytest tests
The real-model test runs only if the default model is in the HuggingFace cache.
"""
from __future__ import annotations

import os
import random
import re
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from panoptic.core.databases.data.models import DataCommit, Sha1Value
from panoptic.core.databases.panoptic.models import PluginKey
from panoptic.core.project.project import Project
from panoptic.models.action_models import ActionContext

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PLUGIN_ID = 'PanopticText'
DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

TOPICS = {
    'cuisine': ['recette', 'farine', 'beurre', 'four', 'gâteau', 'sucre', 'cuisson', 'pâte'],
    'football': ['match', 'ballon', 'équipe', 'but', 'arbitre', 'stade', 'joueur', 'championnat'],
    'astronomie': ['étoile', 'planète', 'télescope', 'galaxie', 'orbite', 'comète', 'lune', 'nébuleuse'],
}
DOCS_PER_TOPIC = 12


def wait_until(predicate, timeout: float = 60.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def make_docs(rng: random.Random) -> list[tuple[str, str]]:
    docs = []
    for topic, words in TOPICS.items():
        for _ in range(DOCS_PER_TOPIC):
            docs.append((topic, f"Le texte parle de {' et de '.join(rng.sample(words, 4))}."))
    return docs


class HashEmbedder:
    """Bag-of-words hashing embedder: fast, deterministic, no model download."""
    dim = 256

    def embed(self, texts, is_query=False, batch_size=32):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r'\w\w+', t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


class PanopticTextTest(unittest.TestCase):
    fake_embedder = True

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        images = root / 'images'
        images.mkdir()
        rng = random.Random(0)
        self.docs = make_docs(rng)
        for i in range(len(self.docs)):
            Image.new('RGB', (8, 8), (i, 255 - i, (i * 7) % 256)).save(images / f'{i:03}.png')

        key = PluginKey(id=PLUGIN_ID, install_path=str(PLUGIN_DIR), source_type='local',
                        source_path=str(PLUGIN_DIR))
        self.project = Project(root / 'project', plugin_keys=[key])
        self.project.start()
        self.assertTrue(wait_until(lambda: self.project.is_plugin_loaded(PLUGIN_ID)))
        self.plugin = next(p for p in self.project.plugins if p.name == PLUGIN_ID)
        if self.fake_embedder:
            self.plugin.embedders.get = lambda name: HashEmbedder()

        task = self.project.import_folder(str(images))
        self.assertTrue(wait_until(lambda: task.state.finished))
        instances = sorted(self.project.get_instances(), key=lambda i: i.file_id)
        self.assertEqual(len(instances), len(self.docs))

        self.text = self.plugin.project.add_property('ocr', 'text', 'sha1')
        self.topic_of = {}
        values = []
        for inst, (topic, doc) in zip(instances, self.docs):
            self.topic_of[inst.sha1] = topic
            values.append(Sha1Value(property_id=self.text.id, sha1=inst.sha1, value=doc))
        self.plugin.project.apply_commit(DataCommit(sha1_values=values))
        self.instances = instances

    def tearDown(self):
        self.project.close()
        sys.modules.pop(PLUGIN_ID, None)
        self.tmp.cleanup()

    def call(self, fn: str, instance_ids=None, **inputs):
        return self.project.action.call(f'{PLUGIN_ID}.{fn}',
                                        ActionContext(instance_ids=instance_ids, ui_inputs=inputs))

    def create_vectors(self):
        res = self.call('create_vector_type', text_property=self.text.id)
        vt = res.value
        res = self.call('compute_vectors', vec_type=vt)
        self.assertIsNotNone(res.task_id)
        self.assertTrue(wait_until(lambda: len(self.plugin.project.get_vectors(vt['id'])) == len(self.docs)))
        self.assertEqual(self.call('compute_vectors', vec_type=vt).notifs[0].name, 'UpToDate')
        return vt

    def assert_topic_purity(self, groups, min_purity=0.9):
        named = [g for g in groups if g.name != 'Outliers']
        total = sum(len(g.sha1s) for g in named)
        majority = sum(max(sum(self.topic_of[s] == t for s in g.sha1s) for t in TOPICS) for g in named)
        self.assertGreaterEqual(majority / total, min_purity, [(g.name, len(g.sha1s)) for g in groups])

    def test_actions_are_registered(self):
        ids = {i for i, a in self.project.action.actions.items() if a.owner == PLUGIN_ID}
        self.assertEqual(ids, {f'{PLUGIN_ID}.{n}' for n in (
            'create_vector_type', 'compute_vectors', 'cluster_text_syntax', 'cluster_text_semantic',
            'find_similar_texts', 'search_text', 'text_map')})

    def test_syntax_clustering(self):
        for analyzer in ('word', 'char'):
            res = self.call('cluster_text_syntax', text_property=self.text.id, nb_clusters=3, analyzer=analyzer)
            self.assertEqual(len(res.groups), 3)
            self.assertEqual(sum(len(g.sha1s) for g in res.groups), len(self.docs))
            self.assert_topic_purity(res.groups)

    def test_selection_and_empty_texts(self):
        selected = [i.id for i in self.instances[:DOCS_PER_TOPIC + 3]]
        res = self.call('cluster_text_syntax', selected, text_property=self.text.id, nb_clusters=2)
        self.assertEqual(sum(len(g.sha1s) for g in res.groups), len(selected))

        res = self.call('cluster_text_syntax', text_property=self.plugin.project.add_property('x', 'text').id)
        self.assertEqual(res.notifs[0].name, 'NoText')

    def test_semantic_clustering_and_saved_property(self):
        vt = self.create_vectors()
        res = self.call('cluster_text_semantic', vec_type=vt, nb_clusters=3, save_as_property=True)
        self.assertEqual(len(res.groups), 3)
        self.assert_topic_purity(res.groups)

        prop = next(p for p in self.plugin.project.get_properties(name='ocr clusters'))
        tags = {t.id: t.value for t in self.plugin.project.get_tags(list_id=prop.tag_list_id)}
        self.assertEqual(sorted(tags.values()), sorted(g.name for g in res.groups))

        # Running again reuses the property and its tags
        self.call('cluster_text_semantic', vec_type=vt, nb_clusters=3, save_as_property=True)
        self.assertEqual(len(self.plugin.project.get_properties(name='ocr clusters')), 1)
        self.assertEqual(len(self.plugin.project.get_tags(list_id=prop.tag_list_id)), len(tags))

        res = self.call('cluster_text_semantic', vec_type=vt, nb_clusters=0, min_cluster_size=5)
        self.assertGreaterEqual(len(res.groups), 2)

    def test_outdated_vectors_are_recomputed(self):
        vt = self.create_vectors()
        sha1 = self.instances[0].sha1
        self.plugin.project.apply_commit(DataCommit(sha1_values=[
            Sha1Value(property_id=self.text.id, sha1=sha1, value='Un tout autre sujet')]))
        res = self.call('cluster_text_semantic', vec_type=vt, nb_clusters=3)
        self.assertEqual(res.notifs[0].name, 'VectorsComputing')
        self.assertIsNotNone(res.task_id)
        self.assertTrue(wait_until(lambda: self.call('cluster_text_semantic', vec_type=vt,
                                                     nb_clusters=3).groups is not None))

    def test_similarity_search_and_map(self):
        vt = self.create_vectors()
        first = self.instances[0]
        res = self.call('find_similar_texts', [first.id], vec_type=vt, max_results=DOCS_PER_TOPIC)
        group = res.groups[0]
        self.assertEqual(group.sha1s[0], first.sha1)
        self.assertGreaterEqual(sum(self.topic_of[s] == self.topic_of[first.sha1] for s in group.sha1s),
                                DOCS_PER_TOPIC * 0.8)

        res = self.call('search_text', vec_type=vt, text='galaxie télescope', min_similarity=0.2)
        self.assertTrue(res.groups[0].sha1s)
        self.assertEqual(self.topic_of[res.groups[0].sha1s[0]], 'astronomie')

        res = self.call('text_map', vec_type=vt)
        self.assertEqual(res.value['count'], len(self.docs))
        coords = np.asarray(res.value['data'], dtype=object).reshape(-1, 3)[:, 1:].astype(float)
        self.assertLessEqual(np.linalg.norm(coords, axis=1).max(), 100 + 1e-6)


def _model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(DEFAULT_MODEL, 'config.json'), str)
    except Exception:
        return False


@unittest.skipUnless(_model_cached(), f'{DEFAULT_MODEL} not in the HuggingFace cache')
class PanopticTextRealModelTest(PanopticTextTest):
    fake_embedder = False

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

    def test_long_texts_are_chunked(self):
        from PanopticText.embedder import TextEmbedder
        embedder = self.plugin.embedders.get(DEFAULT_MODEL)
        self.assertIsInstance(embedder, TextEmbedder)
        long = ' '.join(['Les étoiles et les galaxies du ciel nocturne.'] * 200)
        vec = embedder.embed([long, 'court'])
        self.assertEqual(vec.shape[0], 2)
        np.testing.assert_allclose(np.linalg.norm(vec, axis=1), 1, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
