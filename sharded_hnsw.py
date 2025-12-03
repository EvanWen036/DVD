# sharded_hnsw.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import multiprocessing as mp

import numpy as np
import hnswlib

from store import BaseCollection

"""
Multiprocessing-powered sharded HNSW wrapper.

Key changes vs the original version:
- Uses multiprocessing.Pool to query shards in parallel (on POSIX / fork).
- Fixes k_shard so we don't over-fetch per shard:
    * We split top_k roughly evenly across non-empty shards.
    * Sum of k_shard over shards ≈ top_k (no "* 2" multiplier).
- When upsert is called after queries, we tear down the pool so the next
  query will re-fork with the updated indexes.

Implementation notes:
- This pattern relies on fork semantics: child processes inherit the
  hnswlib.Index objects via copy-on-write. We DO NOT pass indexes through
  pickling; instead, workers access them via module-level globals.
- On platforms without "fork" (e.g., Windows), we fall back to running
  queries in the main process (still correct, just not parallel).
"""

# ---------- module-level globals used by worker processes ----------

_GLOBAL_SHARDS: List[hnswlib.Index] = []
_GLOBAL_LABEL2ID: List[Dict[int, str]] = []
_GLOBAL_META: Dict[str, Optional[dict]] = {}
_GLOBAL_METRIC: str = "cosine"


def _shard_query_worker(args):
    """
    Worker function run in a separate process.

    Args:
        args: (shard_idx, q_vec, k_shard)

    Returns:
        List[hit dicts] for this shard only.
    """
    shard_idx, q, k_shard = args
    index = _GLOBAL_SHARDS[shard_idx]
    labels, distances = index.knn_query(q[None, :], k=k_shard)

    l2id = _GLOBAL_LABEL2ID[shard_idx]
    hits = []
    for label, dist in zip(labels[0], distances[0]):
        vid = l2id.get(int(label))
        if vid is None:
            continue
        if _GLOBAL_METRIC == "cosine":
            score = float(1.0 - dist)
        else:  # "l2"
            score = float(-dist)
        hits.append(
            {
                "id": vid,
                "score": score,
                "metadata": _GLOBAL_META.get(vid),
            }
        )
    return hits


class ShardedHNSWCollection(BaseCollection):
    """
    Wrapper around multiple `hnswlib.Index` shards.

    - Inserts are round-robin sharded across `n_shards`.
    - Each shard is a normal HNSW index.
    - Queries:
        * Dispatch a knn_query to each shard concurrently via a
          multiprocessing.Pool (on POSIX / fork).
        * Each shard returns a slice of candidates (k_shard).
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

        # Multiprocessing pool for shard queries (lazy-created on first query)
        self._pool: Optional[mp.pool.Pool] = None
        # Whether we were able to create a pool (if False, run in-process)
        self._can_use_pool: bool = True

    def __del__(self):
        self._close_pool()

    # ---------- internal helpers ----------

    def _close_pool(self) -> None:
        """
        Tear down the multiprocessing pool if it exists.
        Called on destruction and whenever the index is structurally changed
        (e.g., upsert).
        """
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            finally:
                self._pool = None

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

    def _ensure_pool(self) -> None:
        """
        Lazily create the multiprocessing pool and initialize module-level
        globals used by worker processes. If we fail to create a 'fork'
        context (e.g., on Windows), we disable pool usage and run queries
        in-process.
        """
        if self._pool is not None or not self._can_use_pool:
            return

        try:
            # Prefer 'fork' if available (POSIX)
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                # Fall back to default context (may be 'spawn' on Windows)
                ctx = mp.get_context()
                # On 'spawn', our pattern with globals won't work correctly.
                # So we disable pool usage entirely in that case.
                if ctx.get_start_method() != "fork":
                    self._can_use_pool = False
                    return

            # Important: set globals BEFORE creating pool, so forked
            # processes inherit the fully-built indexes.
            global _GLOBAL_SHARDS, _GLOBAL_LABEL2ID, _GLOBAL_META, _GLOBAL_METRIC
            _GLOBAL_SHARDS = self._shards
            _GLOBAL_LABEL2ID = self._label2id
            _GLOBAL_META = self._meta
            _GLOBAL_METRIC = self.metric

            self._pool = ctx.Pool(processes=self._n_shards)
        except Exception:
            # If anything goes wrong, fall back to in-process execution.
            self._pool = None
            self._can_use_pool = False

    # ---------- BaseCollection API ----------

    def upsert(self, points: List[dict]) -> int:
        if not points:
            return 0

        # Any structural change invalidates existing worker processes.
        self._close_pool()

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
        # Structural change → reset pool
        self._close_pool()

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
        counts = [s.get_current_count() for s in self._shards]
        total_count = sum(counts)
        if total_count == 0:
            return []

        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query dim {q.shape} != {self.dim}")

        # Global k must not exceed total points
        k_global = min(top_k, total_count)

        # Identify non-empty shards
        nonempty = [(idx, cnt) for idx, cnt in enumerate(counts) if cnt > 0]
        n_nonempty = len(nonempty)
        if n_nonempty == 0:
            return []

        # Split k_global across non-empty shards WITHOUT over-fetching.
        base = k_global // n_nonempty
        extra = k_global % n_nonempty

        tasks = []
        for i, (shard_idx, cnt) in enumerate(nonempty):
            k_target = base + (1 if i < extra else 0)
            if k_target <= 0:
                continue
            k_shard = min(k_target, cnt)
            if k_shard <= 0:
                continue
            tasks.append((shard_idx, q, k_shard))

        if not tasks:
            return []

        shard_hits: List[dict] = []

        # Try multiprocessing; if it's unavailable, run in-process.
        self._ensure_pool()
        if self._pool is not None and self._can_use_pool:
            # Parallel across processes
            results = self._pool.map(_shard_query_worker, tasks)
            for hits in results:
                shard_hits.extend(hits)
        else:
            # Fallback: run worker sequentially in the main process
            global _GLOBAL_SHARDS, _GLOBAL_LABEL2ID, _GLOBAL_META, _GLOBAL_METRIC
            _GLOBAL_SHARDS = self._shards
            _GLOBAL_LABEL2ID = self._label2id
            _GLOBAL_META = self._meta
            _GLOBAL_METRIC = self.metric
            for args in tasks:
                shard_hits.extend(_shard_query_worker(args))

        if not shard_hits:
            return []

        # Global top_k by score
        shard_hits.sort(key=lambda h: h["score"], reverse=True)
        return shard_hits[:k_global]
