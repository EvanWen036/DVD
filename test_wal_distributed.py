# test_wal_distributed.py
import time
import numpy as np

from store import BruteCollection
from distributed_collection import WalDistributedCollection


def make_points(dim: int, n: int, prefix: str = "p"):
    points = []
    for i in range(n):
        vec = np.random.randn(dim).astype("float32").tolist()
        points.append({
            "id": f"{prefix}{i}",
            "vector": vec,
            "metadata": {"i": i},
        })
    return points


def main():
    dim = 32
    n_points = 10000
    top_k = 50

    # Baseline exact collection
    baseline = BruteCollection(dim, metric="cosine")

    # WAL-distributed collection using brute replicas (for exact match)
    col = WalDistributedCollection(
        dim=dim,
        metric="cosine",
        replica_backend="hnsw",
        n_replicas=3,
        replication_factor=3,  # wait for all replicas on write
    )

    # ------------- load data ------------- #
    points = make_points(dim, n_points)
    print("Upserting points...")
    baseline.upsert(points)
    col.upsert(points)

    # give replica threads a moment just in case
    time.sleep(0.1)

    # ------------- query tests ------------- #
    print("Running query checks...")
    for q in range(50):
        vec = np.random.randn(dim).astype("float32").tolist()

        hits_base = baseline.query(vec, top_k)
        hits_dist = col.query(vec, top_k)

        ids_base = [h["id"] for h in hits_base]
        ids_dist = [h["id"] for h in hits_dist]

        assert ids_base == ids_dist, (
            f"Top-{top_k} mismatch on query {q}:\n"
            f" baseline={ids_base}\n"
            f" wal_dist={ids_dist}"
        )

    print("✅ All queries matched baseline!")
    print("Done.")


if __name__ == "__main__":
    main()
