import time
import numpy as np
import matplotlib.pyplot as plt

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


def run_routing_benchmark(
    dim: int = 32,
    n_points: int = 3000,
    n_queries: int = 1000,
    top_k: int = 5,
    load_penalty: float = 10.0,
):
    replica_delays_ms = [0.0, 3.0, 7.0]

    col = WalDistributedCollection(
        dim=dim,
        metric="cosine",
        replica_backend="brute",
        n_replicas=3,
        replication_factor=3,   
        window_size=100,
        load_penalty=load_penalty,
        replica_delays_ms=replica_delays_ms,
    )

    points = make_points(dim, n_points)
    print(f"[load_penalty={load_penalty}] Upserting points...")
    col.upsert(points)

    time.sleep(0.2)

    chosen_indices = []
    latencies_ms = []

    print(f"[load_penalty={load_penalty}] Running routing benchmark...")
    for _ in range(n_queries):
        vec = np.random.randn(dim).astype("float32").tolist()
        t0 = time.perf_counter()
        col.query(vec, top_k)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(dt_ms)
        chosen_indices.append(col._last_chosen_replica)

    return chosen_indices, latencies_ms, replica_delays_ms


if __name__ == "__main__":
    load_penalties = [0, 10, 50, 1000]
    results = []

    for lp in load_penalties:
        chosen, latencies, delays = run_routing_benchmark(load_penalty=lp)
        results.append((lp, chosen, latencies, delays))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    bins = [-0.5, 0.5, 1.5, 2.5] 

    for ax, (lp, chosen, _, delays) in zip(axes, results):
        ax.hist(chosen, bins=bins, rwidth=0.8)
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel("Replica index")
        ax.set_ylabel("# of queries")
        ax.set_title(f"load_penalty={lp}\nreplica_delays={delays}")

    fig.suptitle("Replica selection histogram vs load_penalty", fontsize=14)
    plt.tight_layout()
    plt.show()
