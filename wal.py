from __future__ import annotations
from dataclasses import dataclass
from threading import Lock, Condition
from typing import List, Dict, Literal, Any

OpType = Literal["upsert", "delete"]

@dataclass
class LogRecord:
    lsn: int
    op: OpType        
    payload: Any

class WriteAheadLog:
    def __init__(self):
        self._records: List[LogRecord] = []
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self._replica_lsn: Dict[str, int] = {}  

    def append(self, op: OpType, payload: Any) -> int:
        with self._lock:
            lsn = len(self._records)
            self._records.append(LogRecord(lsn, op, payload))
            self._cond.notify_all()
            return lsn

    def next_from(self, last_lsn: int) -> LogRecord:
        want = last_lsn + 1
        with self._lock:
            while want >= len(self._records):
                self._cond.wait()
            return self._records[want]

    def ack_replica(self, replica_id: str, lsn: int) -> None:
        with self._lock:
            self._replica_lsn[replica_id] = lsn
            self._cond.notify_all()

    def wait_replicated(self, lsn: int, min_replicas: int) -> None:
        with self._lock:
            while True:
                count = sum(1 for v in self._replica_lsn.values() if v >= lsn)
                if count >= min_replicas:
                    return
                self._cond.wait()


    def latest_lsn(self) -> int:
        with self._lock:
            return len(self._records) - 1

    def replica_lsn(self, replica_id: str) -> int:
        with self._lock:
            return self._replica_lsn.get(replica_id, -1)
