import time
import random
from typing import List, Tuple

import numpy as np

from store import BruteCollection, HNSWCollection
from sharded_hnsw import ShardedHNSWCollection


def make_random_points(n: int, dim: int) -> List[dict]:
    vecs = np.random.randn(n, dim).astype(np.float32)
    points = []
    for i in range(n):
        points.append({
            "id": f"vec_{i}",
            "vector": vecs[i].tolist(),
            "metadata": {"idx": i},
        })
    return points


def run_pair_against_brute(
    brute,
    other,
    queries: np.ndarray,
    top_k: int,
) -> Tuple[float, float, float]:
    n_queries = queries.shape[0]
    brute_times = []
    other_times = []
    recalls = []

    for q in queries:
        q_list = q.tolist()

        t0 = time.time()
        b_hits = brute.query(q_list, top_k)
        t1 = time.time()
        brute_times.append(t1 - t0)

        t0 = time.time()
        o_hits = other.query(q_list, top_k)
        t1 = time.time()
        other_times.append(t1 - t0)

        b_ids = [h["id"] for h in b_hits]
        o_ids = [h["id"] for h in o_hits]

        if not b_ids:  
            recalls.append(1.0)
            continue

        inter = len(set(b_ids) & set(o_ids))
        recalls.append(inter / min(top_k, len(b_ids)))

    avg_brute = sum(brute_times) / n_queries
    avg_other = sum(other_times) / n_queries
    avg_recall = sum(recalls) / n_queries
    return avg_brute, avg_other, avg_recall


def main():
    random.seed(0)
    np.random.seed(0)

    dim = 128
    n_points = 100_000
    n_queries = 2000
    top_k = 100
    metric = "cosine"  

    print(f"dim={dim}, n_points={n_points}, n_queries={n_queries}, top_k={top_k}")
    print(f"metric={metric}")

    brute = BruteCollection(dim=dim, metric=metric)
    hnsw = HNSWCollection(
        dim=dim,
        metric=metric,
        max_elements=n_points,
        M=16,
        ef_construction=200,
        ef=200,
    )
    sharded = ShardedHNSWCollection(
        dim=dim,
        metric=metric,
        n_shards=4,               
        max_elements_per_shard=n_points // 4 + 1024,
        M=16,
        ef_construction=200,
        ef=200,
    )

    print("Generating random points...")
    points = make_random_points(n_points, dim)

    print("Upserting into brute...")
    t0 = time.time()
    brute.upsert(points)
    t1 = time.time()
    print(f"Brute upsert time:   {t1 - t0:.3f}s")

    print("Upserting into HNSW...")
    t0 = time.time()
    hnsw.upsert(points)
    t1 = time.time()
    print(f"HNSW upsert time:    {t1 - t0:.3f}s")

    print("Upserting into ShardedHNSW...")
    t0 = time.time()
    sharded.upsert(points)
    t1 = time.time()
    print(f"Sharded upsert time: {t1 - t0:.3f}s")

    print("Generating query vectors...")
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    print("\nRunning benchmark vs HNSW...")
    avg_b_brute, avg_hnsw, recall_hnsw = run_pair_against_brute(
        brute, hnsw, queries, top_k
    )

    print("Running benchmark vs ShardedHNSW...")
    avg_b_brute2, avg_sharded, recall_sharded = run_pair_against_brute(
        brute, sharded, queries, top_k
    )

    print("\n=== Results (HNSW) ===")
    print(f"Avg brute query time:   {avg_b_brute * 1e3:.3f} ms")
    print(f"Avg HNSW query time:    {avg_hnsw * 1e3:.3f} ms")
    print(f"Avg recall@{top_k}:       {recall_hnsw:.3f}")

    print("\n=== Results (ShardedHNSW) ===")
    print(f"Avg brute query time:   {avg_b_brute2 * 1e3:.3f} ms")
    print(f"Avg Sharded query time: {avg_sharded * 1e3:.3f} ms")
    print(f"Avg recall@{top_k}:       {recall_sharded:.3f}")


if __name__ == "__main__":
    main()
