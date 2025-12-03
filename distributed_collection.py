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
    def __init__(
        self,
        dim: int,
        metric: str,
        replica_backend: str = "hnsw",
        n_replicas: int = 3,
        replication_factor: Optional[int] = None,
        window_size: int = 100,      # last K requests to track
        load_penalty: float = 10.0,  # weight on recent load vs latency
        replica_delays_ms: Optional[List[float]] = None,  # artificial per-replica delay
    ):
        self.dim = dim
        self.metric = metric

        self._wal = WriteAheadLog()
        self._primary: BaseCollection = make_collection(dim, metric, replica_backend)

        if replication_factor is None:
            replication_factor = n_replicas
        self._replication_factor = replication_factor

        # --- replicas --- #
        self._replicas: List[ReplicaWorker] = []
        for i in range(n_replicas):
            r = ReplicaWorker(f"replica-{i}", self._wal, dim, metric, replica_backend)
            r.start()
            self._replicas.append(r)

        # --- artificial per-replica delays (for experiments) --- #
        if replica_delays_ms is None:
            replica_delays_ms = [0.0] * n_replicas
        else:
            # pad / trim to length n_replicas
            if len(replica_delays_ms) < n_replicas:
                replica_delays_ms = list(replica_delays_ms) + [0.0] * (n_replicas - len(replica_delays_ms))
            elif len(replica_delays_ms) > n_replicas:
                replica_delays_ms = list(replica_delays_ms[:n_replicas])
        self._replica_delays_ms = replica_delays_ms

        # --- routing stats --- #
        self._stats_lock = threading.Lock()
        self._window_size = window_size
        self._load_penalty = load_penalty

        self._latency_ms: List[float] = [5.0] * n_replicas  # EMA of latency
        self._recent_route = deque()                        # sliding window of indices
        self._recent_counts: List[int] = [0] * n_replicas   # count per replica

        # For experiments: which replica was used last?
        self._last_chosen_replica: Optional[int] = None

    # ---------------- write path ---------------- #

    def upsert(self, points: List[dict]) -> int:
        if not points:
            return 0
        lsn = self._wal.append("upsert", {"points": points})
        n = self._primary.upsert(points)
        self._wal.wait_replicated(lsn, self._replication_factor)
        return n

    def delete(self, ids: List[str]) -> int:
        if not ids:
            return 0
        lsn = self._wal.append("delete", {"ids": ids})
        n = self._primary.delete(ids)
        self._wal.wait_replicated(lsn, self._replication_factor)
        return n

    # ---------------- routing stats helpers ---------------- #

    def _record_route(self, idx: int) -> None:
        """
        Update load stats for replica `idx`.

        Caller MUST hold self._stats_lock.
        """
        if len(self._recent_route) >= self._window_size:
            old = self._recent_route.popleft()
            self._recent_counts[old] -= 1
        self._recent_route.append(idx)
        self._recent_counts[idx] += 1

    def _update_latency(self, idx: int, dt_ms: float) -> None:
        with self._stats_lock:
            alpha = 0.2
            prev = self._latency_ms[idx]
            self._latency_ms[idx] = (1 - alpha) * prev + alpha * dt_ms

    def _choose_replica(self) -> Optional[int]:
        """
        Only consider replicas that are fully up-to-date with the WAL.
        Among those, choose the one with the lowest score:

            score = avg_latency_ms + load_penalty * recent_fraction
        """
        if not self._replicas:
            return None

        latest = self._wal.latest_lsn()
        if latest < 0:
            latest = -1

        with self._stats_lock:
            total_recent = len(self._recent_route)
            if total_recent == 0:
                total_recent = 1

            best_idx: Optional[int] = None
            best_score = math.inf

            for i, rep in enumerate(self._replicas):
                repl_lsn = self._wal.replica_lsn(rep.id)
                if repl_lsn < latest:
                    continue  # not up-to-date, skip
                latency = self._latency_ms[i]
                load_frac = self._recent_counts[i] / total_recent
                score = latency + self._load_penalty * load_frac
                #print(f"Replica: {i}, Latency: {latency}, Load_frac: {load_frac}, Score: {score}")
                if score < best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                return None

            # mark routing under the same lock
            self._record_route(best_idx)
            self._last_chosen_replica = best_idx
            return best_idx

    # ---------------- read path ---------------- #

    def query(self, vector: List[float], top_k: int) -> List[dict]:
        if top_k <= 0:
            return []

        idx = self._choose_replica()
        if idx is None:
            # No replica is fully caught up → go to primary
            return self._primary.query(vector, top_k)

        rep = self._replicas[idx]



        start = time.perf_counter()
        # simulate per-replica “network” delay for experiments
        delay_ms = self._replica_delays_ms[idx]
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        try:
            hits = rep.col.query(vector, top_k)
        except Exception:
            # If the replica errors, fall back to primary
            return self._primary.query(vector, top_k)
        dt_ms = (time.perf_counter() - start) * 1000.0

        self._update_latency(idx, dt_ms)
        return hits
