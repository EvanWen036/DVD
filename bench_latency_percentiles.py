import time
import numpy as np
import matplotlib.pyplot as plt

from distributed_collection import WalDistributedCollection


def make_points(dim: int, n: int, prefix: str = "p"):
    points = []
    for i in range(n):
        vec = np.random.randn(dim).astype("float32").tolist()
        points.append(
            {
                "id": f"{prefix}{i}",
                "vector": vec,
                "metadata": {"i": i},
            }
        )
    return points


def run_latency_benchmark(
    load_penalty: float,
    replica_delays_ms,
    dim: int = 64,
    n_points: int = 5000,
    n_queries: int = 2000,
    top_k: int = 10,
):
    """
    Build a WalDistributedCollection with given load_penalty and per-replica
    artificial delays, then run n_queries random queries and return the
    list of per-query latencies (ms).
    """
    print(f"\n=== load_penalty={load_penalty}, delays={replica_delays_ms} ===")

    col = WalDistributedCollection(
        dim=dim,
        metric="cosine",
        replica_backend="brute",  # brute so results are exact
        n_replicas=len(replica_delays_ms),
        replication_factor=len(replica_delays_ms),  # wait for all to be up-to-date
        window_size=100,
        load_penalty=load_penalty,
        replica_delays_ms=replica_delays_ms,
    )

    # Load data once; WAL replicates to all replicas
    points = make_points(dim, n_points)
    print("Upserting points...")
    col.upsert(points)

    # Let replica workers settle
    time.sleep(0.2)

    # Warmup
    for _ in range(50):
        q = np.random.randn(dim).astype("float32").tolist()
        col.query(q, top_k)

    # Measure latencies
    latencies_ms = []
    print("Running latency benchmark...")
    for _ in range(n_queries):
        q = np.random.randn(dim).astype("float32").tolist()
        t0 = time.perf_counter()
        col.query(q, top_k)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return np.array(latencies_ms)


def main():
    # Fixed artificial per-replica delays (ms)
    replica_delays_ms = [0.0, 3.0, 7.0]

    # Different routing tradeoffs between latency vs load
    load_penalties = [0, 10, 50, 1000]

    p50s = []
    p95s = []

    for lp in load_penalties:
        lat = run_latency_benchmark(
            load_penalty=lp,
            replica_delays_ms=replica_delays_ms,
            dim=64,
            n_points=5000,
            n_queries=2000,
            top_k=10,
        )
        p50 = np.percentile(lat, 50)
        p95 = np.percentile(lat, 95)
        p50s.append(p50)
        p95s.append(p95)
        print(
            f"load_penalty={lp}: p50={p50:.3f} ms, p95={p95:.3f} ms "
            f"(min={lat.min():.3f}, max={lat.max():.3f})"
        )

    # ----- Plot p50 and p95 vs load_penalty ----- #
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    x = np.arange(len(load_penalties))

    # p50 subplot
    axes[0].plot(load_penalties, p50s, marker="o")
    axes[0].set_ylabel("p50 latency (ms)")
    axes[0].set_title(
        f"Latency vs load_penalty\nreplica_delays={replica_delays_ms}"
    )
    axes[0].grid(alpha=0.3)

    # p95 subplot
    axes[1].plot(load_penalties, p95s, marker="o")
    axes[1].set_xlabel("load_penalty")
    axes[1].set_ylabel("p95 latency (ms)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
