# store.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import hnswlib


class BaseCollection(ABC):
    dim: int
    metric: str

    @abstractmethod
    def upsert(self, points: List[dict]) -> int:
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        ...

    @abstractmethod
    def query(self, vector: List[float], top_k: int) -> List[dict]:
        ...


class BruteCollection(BaseCollection):
    def __init__(self, dim: int, metric: str = "cosine"):
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.dim = dim
        self.metric = metric  # "cosine" or "l2"
        self._id2idx: Dict[str, int] = {}
        self._ids: List[str] = []
        self._meta: List[Optional[dict]] = []
        self._vecs = np.empty((0, dim), dtype=np.float32)

    def upsert(self, points: List[dict]) -> int:
        for p in points:
            vid = p["id"]
            vec = np.asarray(p["vector"], dtype=np.float32)
            if vec.shape != (self.dim,):
                raise ValueError(f"vector dim {vec.shape} != {self.dim}")
            meta = p.get("metadata")
            if vid in self._id2idx:
                i = self._id2idx[vid]
                self._vecs[i] = vec
                self._meta[i] = meta
            else:
                i = len(self._ids)
                self._id2idx[vid] = i
                self._ids.append(vid)
                self._meta.append(meta)
                self._vecs = np.vstack([self._vecs, vec[None, :]])
        return len(points)

    def delete(self, ids: List[str]) -> int:
        deleted = 0
        for vid in ids:
            if vid not in self._id2idx:
                continue
            i = self._id2idx.pop(vid)
            last_i = len(self._ids) - 1
            if i != last_i:
                # move last into i
                self._vecs[i] = self._vecs[last_i]
                self._ids[i] = self._ids[last_i]
                self._meta[i] = self._meta[last_i]
                self._id2idx[self._ids[i]] = i
            # shrink
            self._vecs = self._vecs[:-1]
            self._ids.pop()
            self._meta.pop()
            deleted += 1
        return deleted

    def _scores(self, q: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            A = self._vecs
            if len(A) == 0:
                return np.empty((0,), dtype=np.float32)
            Aq = np.linalg.norm(A, axis=1, keepdims=True)
            Aq[Aq == 0] = 1.0
            qn = q / (np.linalg.norm(q) + 1e-12)
            An = A / Aq
            return (An @ qn).astype(np.float32)
        else:  # l2 distance -> convert to negative distance so higher is better
            if len(self._vecs) == 0:
                return np.empty((0,), dtype=np.float32)
            d = np.linalg.norm(self._vecs - q[None, :], axis=1)
            return (-d).astype(np.float32)

    def query(self, vector: List[float], top_k: int) -> List[dict]:
        if top_k <= 0:
            return []
        if len(self._vecs) == 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query dim {q.shape} != {self.dim}")
        scores = self._scores(q)
        k = min(top_k, scores.shape[0])
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        hits = []
        for i in idxs:
            hits.append({
                "id": self._ids[i],
                "score": float(scores[i]),
                "metadata": self._meta[i]
            })
        return hits


class HNSWCollection(BaseCollection):
    """
    HNSW-backed collection using `hnswlib`.

    Semantics:
      - score is "higher is better"
        * cosine: score = 1 - distance  (distance in [0, 2])
        * l2:     score = -distance     (distance is L2^2 in hnswlib)
    """

    def __init__(self, dim: int, metric: str = "cosine",
                 max_elements: int = 1024, M: int = 16,
                 ef_construction: int = 200, ef: int = 200):
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.dim = dim
        self.metric = metric

        space = "cosine" if metric == "cosine" else "l2"
        self._index = hnswlib.Index(space=space, dim=dim)
        self._max_elements = max_elements
        self._index.init_index(max_elements=max_elements,
                               ef_construction=ef_construction,
                               M=M)
        self._index.set_ef(ef)

        # Map string ids -> int labels used by HNSW
        self._id2label: Dict[str, int] = {}
        self._label2id: Dict[int, str] = {}
        self._meta: Dict[str, Optional[dict]] = {}

        self._next_label: int = 0  # next free label

    def _ensure_capacity(self, n_new: int) -> None:
        needed = self._next_label + n_new
        if needed > self._max_elements:
            new_max = max(needed, self._max_elements * 2)
            self._index.resize_index(new_max)
            self._max_elements = new_max

    def upsert(self, points: List[dict]) -> int:
        if not points:
            return 0

        # Ensure capacity for new items
        n_new_ids = sum(1 for p in points if p["id"] not in self._id2label)
        if n_new_ids > 0:
            self._ensure_capacity(n_new_ids)

        vecs = []
        labels = []

        for p in points:
            vid = p["id"]
            vec = np.asarray(p["vector"], dtype=np.float32)
            if vec.shape != (self.dim,):
                raise ValueError(f"vector dim {vec.shape} != {self.dim}")
            meta = p.get("metadata")

            if vid in self._id2label:
                label = self._id2label[vid]
            else:
                label = self._next_label
                self._next_label += 1
                self._id2label[vid] = label
                self._label2id[label] = vid

            self._meta[vid] = meta
            vecs.append(vec)
            labels.append(label)

        if vecs:
            vecs_arr = np.stack(vecs, axis=0)
            labels_arr = np.asarray(labels, dtype=np.int64)
            # hnswlib allows adding new vectors with existing labels to update them
            self._index.add_items(vecs_arr, labels_arr)

        return len(points)

    def delete(self, ids: List[str]) -> int:
        deleted = 0
        for vid in ids:
            label = self._id2label.pop(vid, None)
            if label is None:
                continue
            # mark_deleted keeps the node but excludes it from future queries
            self._index.mark_deleted(label)
            self._label2id.pop(label, None)
            self._meta.pop(vid, None)
            deleted += 1
        return deleted

    def query(self, vector: List[float], top_k: int) -> List[dict]:
        if top_k <= 0:
            return []
        if self._index.get_current_count() == 0:
            return []

        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query dim {q.shape} != {self.dim}")

        k = min(top_k, self._index.get_current_count())
        labels, distances = self._index.knn_query(q[None, :], k=k)
        labels = labels[0]
        distances = distances[0]

        hits: List[dict] = []
        for label, dist in zip(labels, distances):
            vid = self._label2id.get(int(label))
            if vid is None:
                continue  # should not normally happen
            if self.metric == "cosine":
                score = float(1.0 - dist)  # hnswlib cosine distance = 1 - cos
            else:  # "l2"
                score = float(-dist)       # hnswlib returns L2^2, we negate
            hits.append({
                "id": vid,
                "score": score,
                "metadata": self._meta.get(vid)
            })
        return hits


class Registry:
    """
    Polymorphic registry: each collection can choose its backend ("brute" or "hnsw"),
    but the API code just sees `BaseCollection`.
    """
    def __init__(self):
        self._cols: Dict[str, BaseCollection] = {}

    def create(self, name: str, dim: int, metric: str, backend: str = "brute") -> None:
        if name in self._cols:
            raise ValueError("collection exists")

        backend = backend.lower()
        if backend == "brute":
            col: BaseCollection = BruteCollection(dim, metric)
        elif backend == "hnsw":
            col = HNSWCollection(dim, metric)
        elif backend == "sharded_hnsw":
            from sharded_hnsw import ShardedHNSWCollection
            col = ShardedHNSWCollection(dim, metric)
        else:
            raise ValueError(f"unknown backend '{backend}' (use 'brute' or 'hnsw')")

        self._cols[name] = col

    def get(self, name: str) -> BaseCollection:
        if name not in self._cols:
            raise KeyError("collection not found")
        return self._cols[name]
