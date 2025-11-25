# sharded_hnsw.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import hnswlib

from store import BaseCollection


class ShardedHNSWCollection(BaseCollection):
    """
    Wrapper around multiple `hnswlib.Index` shards.

    - Inserts are round-robin sharded across `n_shards`.
    - Each shard is a normal HNSW index.
    - Queries:
        * Dispatch a knn_query to each shard concurrently (via a persistent
          ThreadPoolExecutor).
        * Each shard returns only a small slice of candidates.
        * All shard results are merged into a global top_k by score.

    Semantics:
      - score is "higher is better"
        * cosine: score = 1 - distance  (distance in [0, 2])
        * l2:     score = -distance     (distance is L2^2 in hnswlib)
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        n_shards: int = 4,
        max_elements_per_shard: int = 1024,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 200,
    ):
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        if n_shards <= 0:
            raise ValueError("n_shards must be >= 1")

        self.dim = dim
        self.metric = metric
        self._n_shards = n_shards

        space = "cosine" if metric == "cosine" else "l2"

        self._shards: List[hnswlib.Index] = []
        self._max_elements: List[int] = []
        self._next_label: List[int] = []
        self._label2id: List[Dict[int, str]] = []

        for _ in range(n_shards):
            index = hnswlib.Index(space=space, dim=dim)
            index.init_index(
                max_elements=max_elements_per_shard,
                ef_construction=ef_construction,
                M=M,
            )
            index.set_ef(ef)
            self._shards.append(index)
            self._max_elements.append(max_elements_per_shard)
            self._next_label.append(0)
            self._label2id.append({})

        # Global id -> (shard_idx, label)
        self._id2loc: Dict[str, Tuple[int, int]] = {}
        # Global metadata keyed by string id
        self._meta: Dict[str, Optional[dict]] = {}

        # For round-robin sharding
        self._rr_counter: int = 0

        # Persistent executor for shard queries
        self._executor = ThreadPoolExecutor(max_workers=n_shards)

    def __del__(self):
        # Best-effort shutdown; avoid hanging on interpreter exit
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ---------- internal helpers ----------

    def _pick_shard_for_new_id(self) -> int:
        """
        Round-robin shard selection for new IDs.
        """
        shard_idx = self._rr_counter % self._n_shards
        self._rr_counter += 1
        return shard_idx

    def _ensure_capacity(self, shard_idx: int, n_new: int) -> None:
        """
        Ensure a given shard can accommodate `n_new` more labels.
        """
        needed = self._next_label[shard_idx] + n_new
        if needed > self._max_elements[shard_idx]:
            new_max = max(needed, self._max_elements[shard_idx] * 2)
            self._shards[shard_idx].resize_index(new_max)
            self._max_elements[shard_idx] = new_max

    # ---------- BaseCollection API ----------

    def upsert(self, points: List[dict]) -> int:
        if not points:
            return 0

        # Group vectors per shard for batched add_items
        per_shard_vecs: List[List[np.ndarray]] = [[] for _ in range(self._n_shards)]
        per_shard_labels: List[List[int]] = [[] for _ in range(self._n_shards)]
        new_counts: List[int] = [0] * self._n_shards

        for p in points:
            vid = p["id"]
            vec = np.asarray(p["vector"], dtype=np.float32)
            if vec.shape != (self.dim,):
                raise ValueError(f"vector dim {vec.shape} != {self.dim}")
            meta = p.get("metadata")

            if vid in self._id2loc:
                # Existing ID: keep shard the same, reuse label
                shard_idx, label = self._id2loc[vid]
            else:
                # New ID: pick shard (round-robin)
                shard_idx = self._pick_shard_for_new_id()
                label = self._next_label[shard_idx]
                self._next_label[shard_idx] += 1

                self._id2loc[vid] = (shard_idx, label)
                self._label2id[shard_idx][label] = vid
                new_counts[shard_idx] += 1

            self._meta[vid] = meta
            per_shard_vecs[shard_idx].append(vec)
            per_shard_labels[shard_idx].append(label)

        # Ensure capacity and add_items per shard
        for shard_idx in range(self._n_shards):
            if not per_shard_vecs[shard_idx]:
                continue

            if new_counts[shard_idx] > 0:
                self._ensure_capacity(shard_idx, new_counts[shard_idx])

            vecs_arr = np.stack(per_shard_vecs[shard_idx], axis=0)
            labels_arr = np.asarray(per_shard_labels[shard_idx], dtype=np.int64)
            # hnswlib: if labels already exist, they are updated in-place
            self._shards[shard_idx].add_items(vecs_arr, labels_arr)

        return len(points)

    def delete(self, ids: List[str]) -> int:
        deleted = 0
        for vid in ids:
            loc = self._id2loc.pop(vid, None)
            if loc is None:
                continue
            shard_idx, label = loc
            self._shards[shard_idx].mark_deleted(label)
            self._label2id[shard_idx].pop(label, None)
            self._meta.pop(vid, None)
            deleted += 1
        return deleted

    def query(self, vector: List[float], top_k: int) -> List[dict]:
        if top_k <= 0:
            return []

        # Short-circuit if all shards are empty
        total_count = sum(s.get_current_count() for s in self._shards)
        if total_count == 0:
            return []

        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query dim {q.shape} != {self.dim}")

        # Global k must not exceed total points
        k_global = min(top_k, total_count)

        # Decide per-shard k: smaller than global to reduce extra work,
        # but with some headroom to keep recall high.
        base_per_shard = max(k_global // self._n_shards, 1)

        tasks = []
        for shard_idx, index in enumerate(self._shards):
            count = index.get_current_count()
            if count == 0:
                continue

            # Ask each shard for up to ~2 * base_per_shard (capped by count).
            k_shard = min(base_per_shard * 2, count)

            def _query_one_shard(idx=shard_idx, k=k_shard):
                labels, distances = self._shards[idx].knn_query(q[None, :], k=k)
                return idx, labels[0], distances[0]

            tasks.append(self._executor.submit(_query_one_shard))

        shard_hits: List[dict] = []
        for fut in as_completed(tasks):
            shard_idx, labels, distances = fut.result()
            l2id = self._label2id[shard_idx]
            for label, dist in zip(labels, distances):
                vid = l2id.get(int(label))
                if vid is None:
                    continue
                if self.metric == "cosine":
                    score = float(1.0 - dist)
                else:  # "l2"
                    score = float(-dist)
                shard_hits.append({
                    "id": vid,
                    "score": score,
                    "metadata": self._meta.get(vid),
                })

        if not shard_hits:
            return []

        # Global top_k by score
        shard_hits.sort(key=lambda h: h["score"], reverse=True)
        return shard_hits[:k_global]
