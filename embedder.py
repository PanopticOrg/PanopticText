import threading
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    max_length: int
    pooling: str = 'mean'  # 'mean' | 'cls'
    query_prefix: str = ''
    passage_prefix: str = ''


MODEL_SPECS = {
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': ModelSpec(128),
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': ModelSpec(128),
    'intfloat/multilingual-e5-base': ModelSpec(512, query_prefix='query: ', passage_prefix='passage: '),
    'BAAI/bge-m3': ModelSpec(1024, pooling='cls'),
}
DEFAULT_SPEC = ModelSpec(512)

# Long texts are split in windows of max_length tokens; beyond this many windows the rest is ignored
MAX_CHUNKS = 32


def _device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class TextEmbedder:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.spec = MODEL_SPECS.get(model_name, DEFAULT_SPEC)
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        if self.device == 'cuda':
            self.model = self.model.half()
        self.max_length = min(self.spec.max_length, self.tokenizer.model_max_length)
        self._lock = threading.Lock()
        self._torch = torch

    def embed(self, texts: list[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        """L2-normalised float32 embeddings, one row per text (mean of its chunks)."""
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        enc = self.tokenizer([prefix + t for t in texts], truncation=True, max_length=self.max_length,
                             return_overflowing_tokens=True, add_special_tokens=True)
        mapping = enc['overflow_to_sample_mapping']
        keep = [i for i, n in enumerate(_chunk_ranks(mapping)) if n < MAX_CHUNKS]

        dim = self.model.config.hidden_size
        sums = np.zeros((len(texts), dim), dtype=np.float32)
        for start in range(0, len(keep), batch_size):
            rows = keep[start:start + batch_size]
            batch = self.tokenizer.pad({'input_ids': [enc['input_ids'][r] for r in rows],
                                        'attention_mask': [enc['attention_mask'][r] for r in rows]},
                                       return_tensors='pt')
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self._lock, self._torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state.float()
            if self.spec.pooling == 'cls':
                pooled = hidden[:, 0]
            else:
                mask = batch['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = self._torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
            np.add.at(sums, [mapping[r] for r in rows], pooled)
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        return sums / np.maximum(norms, 1e-12)


def _chunk_ranks(mapping: list[int]) -> list[int]:
    seen: dict[int, int] = {}
    ranks = []
    for m in mapping:
        ranks.append(seen.get(m, 0))
        seen[m] = ranks[-1] + 1
    return ranks


class EmbedderManager:
    def __init__(self):
        self._embedders: dict[str, TextEmbedder] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str) -> TextEmbedder:
        with self._lock:
            if model_name not in self._embedders:
                self._embedders[model_name] = TextEmbedder(model_name)
            return self._embedders[model_name]

    def clear(self) -> None:
        with self._lock:
            self._embedders.clear()
