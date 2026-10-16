import hashlib
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from panoptic.core.databases.media.models import Vector, VectorType
from panoptic.core.task.task import Task

logger = logging.getLogger(__name__)

BATCH_SIZE = 64


def text_hash(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


class TextHashes:
    """Hash of the text each stored vector was computed from, to detect edited texts."""

    def __init__(self, path: Path):
        self._path = str(path)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS hashes '
                       '(type_id INTEGER, sha1 TEXT, hash TEXT, PRIMARY KEY (type_id, sha1))')

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self._path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, type_id: int) -> dict[str, str]:
        with self._connect() as db:
            return dict(db.execute('SELECT sha1, hash FROM hashes WHERE type_id = ?', (type_id,)))

    def set(self, type_id: int, hashes: dict[str, str]) -> None:
        with self._lock, self._connect() as db:
            db.executemany('INSERT OR REPLACE INTO hashes VALUES (?, ?, ?)',
                           [(type_id, s, h) for s, h in hashes.items()])


class ComputeTextVectorsTask(Task):
    def __init__(self, plugin, vec_type: VectorType, texts: dict[str, str]):
        super().__init__()
        self.plugin = plugin
        self.vec_type = vec_type
        self.vector_type_id = vec_type.id
        self.texts = texts
        self.name = f"Text vectors: {vec_type.params.get('property_name', '')}"

    def start(self):
        items = list(self.texts.items())
        self.state.total = len(items)
        self.set_step('Loading model', self.vec_type.params['model'])
        self._notify()
        embedder = self.plugin.embedders.get(self.vec_type.params['model'])
        self.set_step('Embedding')
        for start in range(0, len(items), BATCH_SIZE):
            if self.is_cancelled():
                break
            batch = items[start:start + BATCH_SIZE]
            try:
                data = embedder.embed([t for _, t in batch])
            except Exception:
                logger.exception('PanopticText: embedding failed')
                self.state.failed += len(batch)
                self._notify()
                continue
            self.plugin.project.upsert_vectors([Vector(type_id=self.vec_type.id, sha1=s, data=d)
                                                for (s, _), d in zip(batch, data)])
            self.plugin.hashes.set(self.vec_type.id, {s: text_hash(t) for s, t in batch})
            self.state.done += len(batch)
            self._notify()
