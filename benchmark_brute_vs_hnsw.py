import time
import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from store import BruteCollection, HNSWCollection


def make_random_points(n: int, dim: int) -> List[dict]:
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
    metric = "cosine" 
    top_k = 50
    n_queries = 200

    n_points_list = [1_000, 5_000, 10_000, 20_000, 50_000]

    M_list = [8, 16, 32, 64]

    print(f"dim={dim}, n_queries={n_queries}, top_k={top_k}, metric={metric}")
    print(f"Testing n_points in: {n_points_list}")
    print(f"Testing HNSW M values: {M_list}\n")

    brute_times_by_N = {} 
    hnsw_times_by_M = {M: [] for M in M_list}
    recalls_by_M = {M: [] for M in M_list}

    for n_points in n_points_list:
        print(f"=== n_points={n_points} ===")

        points = make_random_points(n_points, dim)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        brute = BruteCollection(dim=dim, metric=metric)
        t0 = time.time()
        brute.upsert(points)
        t1 = time.time()
        brute_build = t1 - t0
        print(f"[Brute] upsert time: {brute_build:.3f}s")

        brute_avg_query_ms = None

        for M in M_list:
            print(f"  -> HNSW with M={M}")

            hnsw = HNSWCollection(
                dim=dim,
                metric=metric,
                max_elements=n_points,
                M=M,
                ef_construction=200,
                ef=200,
            )

            t0 = time.time()
            hnsw.upsert(points)
            t1 = time.time()
            hnsw_build = t1 - t0
            print(f"     [HNSW M={M}] upsert time: {hnsw_build:.3f}s")

            avg_b, avg_h, recall = run_pair_against_brute(brute, hnsw, queries, top_k)
            avg_b_ms = avg_b * 1e3
            avg_h_ms = avg_h * 1e3

            if brute_avg_query_ms is None:
                brute_avg_query_ms = avg_b_ms
                brute_times_by_N[n_points] = avg_b_ms

            hnsw_times_by_M[M].append(avg_h_ms)
            recalls_by_M[M].append(recall)

            print(f"     Avg brute query time: {avg_b_ms:.3f} ms")
            print(f"     Avg HNSW  query time: {avg_h_ms:.3f} ms")
            print(f"     Avg recall@{top_k}:     {recall:.3f}\n")

    Ns = n_points_list
    brute_ms = [brute_times_by_N[N] for N in Ns]

    plt.figure()
    plt.plot(Ns, brute_ms, marker="o", label="Brute-force")
    for M in M_list:
        plt.plot(Ns, hnsw_times_by_M[M], marker="o", label=f"HNSW (M={M})")

    plt.xlabel("Number of points in index (N)")
    plt.ylabel("Average query time (ms)")
    plt.title("Brute-force vs HNSW query time vs N")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hnsw_vs_brute_time_vs_N.png")
    print("Saved plot: hnsw_vs_brute_time_vs_N.png")

    plt.figure()
    for M in M_list:
        plt.plot(Ns, recalls_by_M[M], marker="o", label=f"HNSW recall (M={M})")

    plt.xlabel("Number of points in index (N)")
    plt.ylabel(f"Average recall@{top_k}")
    plt.title(f"HNSW recall@{top_k} vs N for different M")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hnsw_recall_vs_N.png")
    print("Saved plot: hnsw_recall_vs_N.png")


if __name__ == "__main__":
    main()
