# wal_distributed_collection.py
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import List, Optional

from store import BaseCollection, make_collection
from wal import WriteAheadLog
from replica_worker import ReplicaWorker


class WalDistributedCollection(BaseCollection):
    """
    A distributed collection with:
      - a primary replica (applies writes immediately)
      - follower replicas (apply WAL entries asynchronously)
      - read routing to the "best" up-to-date replica

    Features:
      • only route reads to replicas that are fully up-to-date
      • latency + load–aware routing
      • optional artificial per-replica network delay
      • async or sync replication modes
    """

    def __init__(
        self,
        dim: int,
        metric: str,
        replica_backend: str = "hnsw",
        n_replicas: int = 3,
        replication_factor: Optional[int] = None,
        window_size: int = 100,
        load_penalty: float = 10.0,
        replica_delays_ms: Optional[List[float]] = None,
        wait_for_replicas: bool = True,
    ):
        self.dim = dim
        self.metric = metric

        # WAL + primary collection
        self._wal = WriteAheadLog()
        self._primary: BaseCollection = make_collection(dim, metric, replica_backend)

        if replication_factor is None:
            replication_factor = n_replicas
        self._replication_factor = replication_factor

        # Replicas
        self._replicas: List[ReplicaWorker] = []
        for i in range(n_replicas):
            w = ReplicaWorker(f"replica-{i}", self._wal, dim, metric, replica_backend)
            w.start()
            self._replicas.append(w)

        # artificial delays
        if replica_delays_ms is None:
            replica_delays_ms = [0.0] * n_replicas
        else:
            replica_delays_ms = list(replica_delays_ms)
            if len(replica_delays_ms) < n_replicas:
                replica_delays_ms.extend([0.0] * (n_replicas - len(replica_delays_ms)))
            replica_delays_ms = replica_delays_ms[:n_replicas]

        self._replica_delays_ms = replica_delays_ms

        # routing stats (protected by one lock)
        self._stats_lock = threading.Lock()

        self._window_size = window_size
        self._load_penalty = load_penalty

        # moving averages + sliding window load
        self._latency_ms = [5.0] * n_replicas
        self._recent_route = deque()
        self._recent_counts = [0] * n_replicas

        # experiment tracking
        self._last_chosen_replica: Optional[int] = None
        self._last_used_primary: bool = False

        # replication mode
        self._wait_for_replicas = wait_for_replicas

    # ----------------------------------------------------------
    # Write path (append to WAL, apply to primary)
    # ----------------------------------------------------------
    def upsert(self, points: List[dict]) -> int:
        if not points:
            return 0
        lsn = self._wal.append("upsert", {"points": points})
        n = self._primary.upsert(points)

        if self._wait_for_replicas:
            self._wal.wait_replicated(lsn, self._replication_factor)

        return n

    def delete(self, ids: List[str]) -> int:
        if not ids:
            return 0
        lsn = self._wal.append("delete", {"ids": ids})
        n = self._primary.delete(ids)

        if self._wait_for_replicas:
            self._wal.wait_replicated(lsn, self._replication_factor)

        return n

    # ----------------------------------------------------------
    # Routing stats helpers
    # ----------------------------------------------------------
    def _record_route(self, idx: int) -> None:
        """
        Update load stats for replica idx.
        Caller MUST hold self._stats_lock.
        """
        if len(self._recent_route) >= self._window_size:
            old = self._recent_route.popleft()
            self._recent_counts[old] -= 1

        self._recent_route.append(idx)
        self._recent_counts[idx] += 1

    def _update_latency(self, idx: int, dt_ms: float) -> None:
        """
        Exponential moving average. Lock internally.
        """
        with self._stats_lock:
            alpha = 0.2
            prev = self._latency_ms[idx]
            self._latency_ms[idx] = (1 - alpha) * prev + alpha * dt_ms

    # ----------------------------------------------------------
    # Choose only up-to-date replicas + score them
    # ----------------------------------------------------------
    def _choose_replica(self) -> Optional[int]:
        if not self._replicas:
            return None

        latest = self._wal.latest_lsn()
        if latest < 0:
            latest = -1

        with self._stats_lock:
            total_recent = len(self._recent_route) or 1

            best_idx = None
            best_score = math.inf

            for i, rep in enumerate(self._replicas):
                repl_lsn = self._wal.replica_lsn(rep.id)
                if repl_lsn < latest:
                    continue  # stale ⇒ unsafe ⇒ skip

                latency = self._latency_ms[i]
                load_frac = self._recent_counts[i] / total_recent

                score = latency + self._load_penalty * load_frac

                if score < best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                return None

            self._record_route(best_idx)
            self._last_chosen_replica = best_idx
            return best_idx

    # ----------------------------------------------------------
    # Read path (choose best replica or fallback to primary)
    # ----------------------------------------------------------
    def query(self, vector: List[float], top_k: int) -> List[dict]:
        if top_k <= 0:
            return []

        idx = self._choose_replica()
        if idx is None:
            # no safe replica
            self._last_chosen_replica = None
            self._last_used_primary = True
            return self._primary.query(vector, top_k)

        rep = self._replicas[idx]



        start = time.perf_counter()
        # experiment: artificial per-replica delay
        delay_ms = self._replica_delays_ms[idx]
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        try:
            hits = rep.col.query(vector, top_k)
        except Exception:
            # fallback in case of replica error
            self._last_chosen_replica = None
            self._last_used_primary = True
            return self._primary.query(vector, top_k)
        dt_ms = (time.perf_counter() - start) * 1000.0

        self._last_used_primary = False
        self._update_latency(idx, dt_ms)
        return hits
