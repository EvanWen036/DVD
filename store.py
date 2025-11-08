from typing import Dict, List, Optional, Any
import numpy as np

class Collection:
    def __init__(self, dim: int, metric: str = "cosine"):
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
        # Lazy delete: swap with last row for O(1) removal
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
            # cosine similarity in [-1,1]; higher is better
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
        if len(self._vecs) == 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query dim {q.shape} != {self.dim}")
        scores = self._scores(q)
        k = min(top_k, scores.shape[0])
        idxs = np.argpartition(-scores, k-1)[:k]
        # sort top-k
        idxs = idxs[np.argsort(-scores[idxs])]
        hits = []
        for i in idxs:
            hits.append({
                "id": self._ids[i],
                "score": float(scores[i]),
                "metadata": self._meta[i]
            })
        return hits

class Registry:
    def __init__(self):
        self._cols: Dict[str, Collection] = {}

    def create(self, name: str, dim: int, metric: str) -> None:
        if name in self._cols:
            raise ValueError("collection exists")
        self._cols[name] = Collection(dim, metric)

    def get(self, name: str) -> Collection:
        if name not in self._cols:
            raise KeyError("collection not found")
        return self._cols[name]
