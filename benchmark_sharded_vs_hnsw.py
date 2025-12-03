import time
import random
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from store import BruteCollection, HNSWCollection
from sharded_hnsw import ShardedHNSWCollection


def make_random_points(n: int, dim: int) -> List[dict]:
    """Generate n random points in R^dim with ids and dummy metadata."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    points = []
    for i in range(n):
        points.append(
            {
                "id": f"vec_{i}",
                "vector": vecs[i].tolist(),
                "metadata": {"idx": i},
            }
        )
    return points


def run_pair_against_brute(
    brute,
    other,
    queries: np.ndarray,
    top_k: int,
) -> Tuple[float, float, float]:
    """
    Run queries against brute & another collection and compute:
    - avg brute query time
    - avg other query time
    - avg recall@k of `other` vs brute (intersection / k)
    """
    n_queries = queries.shape[0]
    brute_times = []
    other_times = []
    recalls = []

    for q in queries:
        q_list = q.tolist()

        # Brute
        t0 = time.time()
        b_hits = brute.query(q_list, top_k)
        t1 = time.time()
        brute_times.append(t1 - t0)

        # Other collection (HNSW or ShardedHNSW)
        t0 = time.time()
        o_hits = other.query(q_list, top_k)
        t1 = time.time()
        other_times.append(t1 - t0)

        b_ids = [h["id"] for h in b_hits]
        o_ids = [h["id"] for h in o_hits]

        if not b_ids:  # empty index
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
    metric = "cosine"  # or "l2"
    top_k = 50
    n_queries = 200

    # Dataset sizes and shard counts to test
    n_points_list = [10_000, 20_000, 50_000]
    shard_counts = [1, 2, 4, 8]

    # HNSW hyperparams (fixed M = 32 as requested)
    M = 32
    ef_construction = 200
    ef = 200

    print(f"dim={dim}, metric={metric}, top_k={top_k}, n_queries={n_queries}")
    print(f"Testing N in: {n_points_list}")
    print(f"Testing shard counts: {shard_counts}")
    print(f"HNSW params: M={M}, ef_construction={ef_construction}, ef={ef}\n")

    # Results containers
    # HNSW baseline (no sharding) per N
    hnsw_time_ms_by_N: Dict[int, float] = {}
    hnsw_recall_by_N: Dict[int, float] = {}

    # Sharded: for each shard count, we store lists over N (aligned with n_points_list)
    sharded_time_ms_by_shards: Dict[int, List[float]] = {s: [] for s in shard_counts}
    sharded_recall_by_shards: Dict[int, List[float]] = {s: [] for s in shard_counts}

    # Also track brute-force time per N (same regardless of sharding)
    brute_time_ms_by_N: Dict[int, float] = {}

    for n_points in n_points_list:
        print(f"=== N = {n_points} ===")

        # Generate data & queries once per N
        points = make_random_points(n_points, dim)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        # Build brute-force index
        brute = BruteCollection(dim=dim, metric=metric)
        t0 = time.time()
        brute.upsert(points)
        t1 = time.time()
        brute_build_s = t1 - t0
        print(f"[Brute] upsert time: {brute_build_s:.3f}s")

        # Build baseline HNSW (no sharding)
        hnsw = HNSWCollection(
            dim=dim,
            metric=metric,
            max_elements=n_points,
            M=M,
            ef_construction=ef_construction,
            ef=ef,
        )
        t0 = time.time()
        hnsw.upsert(points)
        t1 = time.time()
        hnsw_build_s = t1 - t0
        print(f"[HNSW] upsert time: {hnsw_build_s:.3f}s")

        # Benchmark HNSW vs brute
        avg_b, avg_h, recall_h = run_pair_against_brute(brute, hnsw, queries, top_k)
        avg_b_ms = avg_b * 1e3
        avg_h_ms = avg_h * 1e3

        brute_time_ms_by_N[n_points] = avg_b_ms
        hnsw_time_ms_by_N[n_points] = avg_h_ms
        hnsw_recall_by_N[n_points] = recall_h

        print(f"[HNSW] Avg brute query time:  {avg_b_ms:.3f} ms")
        print(f"[HNSW] Avg HNSW  query time:  {avg_h_ms:.3f} ms")
        print(f"[HNSW] Avg recall@{top_k}:      {recall_h:.3f}\n")

        # Now benchmark ShardedHNSW for each shard count
        for shards in shard_counts:
            print(f"  -> ShardedHNSW with n_shards={shards}")

            # Reasonable per-shard capacity (growable anyway due to resize_index)
            max_elements_per_shard = n_points // shards + 1024

            sharded = ShardedHNSWCollection(
                dim=dim,
                metric=metric,
                n_shards=shards,
                max_elements_per_shard=max_elements_per_shard,
                M=16,
                ef_construction=64,
                ef=64,
            )

            t0 = time.time()
            sharded.upsert(points)
            t1 = time.time()
            sharded_build_s = t1 - t0
            print(f"     [Sharded n={shards}] upsert time: {sharded_build_s:.3f}s")

            avg_b2, avg_s, recall_s = run_pair_against_brute(brute, sharded, queries, top_k)
            avg_b2_ms = avg_b2 * 1e3
            avg_s_ms = avg_s * 1e3

            # Sanity check: brute time per N should be roughly the same
            # but we won't enforce it, just record the first measurement above.

            sharded_time_ms_by_shards[shards].append(avg_s_ms)
            sharded_recall_by_shards[shards].append(recall_s)

            print(f"     [Sharded n={shards}] Avg brute query time: {avg_b2_ms:.3f} ms")
            print(f"     [Sharded n={shards}] Avg Sharded  q time: {avg_s_ms:.3f} ms")
            print(f"     [Sharded n={shards}] Avg recall@{top_k}:    {recall_s:.3f}\n")

    # ---- Plot 1: Query time vs N (HNSW vs ShardedHNSW) ----
    Ns = n_points_list
    brute_ms = [brute_time_ms_by_N[N] for N in Ns]
    hnsw_ms = [hnsw_time_ms_by_N[N] for N in Ns]

    plt.figure()
    plt.plot(Ns, brute_ms, marker="o", label="Brute-force")
    plt.plot(Ns, hnsw_ms, marker="o", label="HNSW (no sharding)")
    for shards in shard_counts:
        plt.plot(
            Ns,
            sharded_time_ms_by_shards[shards],
            marker="o",
            label=f"ShardedHNSW (n_shards={shards})",
        )

    plt.xlabel("Number of points (N)")
    plt.ylabel("Average query time (ms)")
    plt.title("Query time vs N: Brute vs HNSW vs ShardedHNSW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sharded_vs_hnsw_time_vs_N.png")
    print("Saved plot: sharded_vs_hnsw_time_vs_N.png")

    # ---- Plot 2: Recall vs N (HNSW vs ShardedHNSW) ----
    hnsw_recall = [hnsw_recall_by_N[N] for N in Ns]

    plt.figure()
    plt.plot(Ns, hnsw_recall, marker="o", label="HNSW (no sharding)")
    for shards in shard_counts:
        plt.plot(
            Ns,
            sharded_recall_by_shards[shards],
            marker="o",
            label=f"ShardedHNSW recall (n_shards={shards})",
        )

    plt.xlabel("Number of points (N)")
    plt.ylabel(f"Average recall@{top_k}")
    plt.title(f"Recall@{top_k} vs N: HNSW vs ShardedHNSW")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sharded_vs_hnsw_recall_vs_N.png")
    print("Saved plot: sharded_vs_hnsw_recall_vs_N.png")

    # ---- Optional: For largest N, plot time/recall vs shard count ----
    N_max = Ns[-1]
    idx_max = len(Ns) - 1

    times_at_maxN = [sharded_time_ms_by_shards[s][idx_max] for s in shard_counts]
    recalls_at_maxN = [sharded_recall_by_shards[s][idx_max] for s in shard_counts]

    plt.figure()
    plt.axhline(
        y=hnsw_time_ms_by_N[N_max],
        linestyle="--",
        label="HNSW (no sharding)",
    )
    plt.plot(shard_counts, times_at_maxN, marker="o", label="ShardedHNSW")
    plt.xlabel("Number of shards")
    plt.ylabel("Average query time (ms)")
    plt.title(f"Query time vs shard count (N={N_max})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"sharded_vs_hnsw_time_vs_shards_N{N_max}.png")
    print(f"Saved plot: sharded_vs_hnsw_time_vs_shards_N{N_max}.png")

    plt.figure()
    plt.axhline(
        y=hnsw_recall_by_N[N_max],
        linestyle="--",
        label="HNSW (no sharding)",
    )
    plt.plot(shard_counts, recalls_at_maxN, marker="o", label="ShardedHNSW")
    plt.xlabel("Number of shards")
    plt.ylabel(f"Average recall@{top_k}")
    plt.title(f"Recall@{top_k} vs shard count (N={N_max})")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"sharded_vs_hnsw_recall_vs_shards_N{N_max}.png")
    print(f"Saved plot: sharded_vs_hnsw_recall_vs_shards_N{N_max}.png")


if __name__ == "__main__":
    main()
