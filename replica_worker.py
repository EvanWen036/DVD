# replica_worker.py
from __future__ import annotations
import threading
from typing import List, Dict

from store import BaseCollection, make_collection
from wal import WriteAheadLog, LogRecord

class ReplicaWorker:
    def __init__(self, replica_id: str, wal: WriteAheadLog,
                 dim: int, metric: str, backend: str):
        self.id = replica_id
        self.wal = wal
        self.col: BaseCollection = make_collection(dim, metric, backend)
        self._last_lsn = -1
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = False

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join()

    def _apply(self, rec: LogRecord):
        if rec.op == "upsert":
            self.col.upsert(rec.payload["points"])
        elif rec.op == "delete":
            self.col.delete(rec.payload["ids"])
        else:
            raise ValueError(f"unknown op {rec.op}")

    def _run(self):
        while not self._stop:
            rec = self.wal.next_from(self._last_lsn)
            self._apply(rec)
            self._last_lsn = rec.lsn
            self.wal.ack_replica(self.id, self._last_lsn)
