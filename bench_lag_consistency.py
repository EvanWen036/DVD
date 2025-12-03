# bench_lag_consistency.py
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from store import BruteCollection
from distributed_collection import WalDistributedCollection


def make_points(dim: int, start: int, n: int, prefix: str = "p"):
    points = []
    for i in range(start, start + n):
        vec = np.random.randn(dim).astype("float32").tolist()
        points.append({
            "id": f"{prefix}{i}",
            "vector": vec,
            "metadata": {"i": i},
        })
    return points


def run_lag_experiment(
    dim: int = 16,
    initial_points: int = 1000,
    rounds: int = 300,
    top_k: int = 5,
):
    # ---------- Baseline exact collection ----------
    baseline = BruteCollection(dim, metric="cosine")

    # ---------- WAL-distributed with async replication (replicas can lag) ----------
    dist = WalDistributedCollection(
        dim=dim,
        metric="cosine",
        replica_backend="brute",       # brute so we get exact results
        n_replicas=3,
        replication_factor=1,          # minimal guarantees
        window_size=100,
        load_penalty=10.0,
        replica_delays_ms=[0.0, 3.0, 7.0],  # simulate slower replicas
        wait_for_replicas=False,       # async replication (laggy)
    )

    # ---------- Initial bulk load (sync) ----------
    print("Initial bulk load...")
    pts0 = make_points(dim, 0, initial_points)
    baseline.upsert(pts0)

    # load synchronously so all replicas start in sync
    dist._wait_for_replicas = True
    dist.upsert(pts0)
    dist._wait_for_replicas = False
    time.sleep(0.2)  # catch up

    # ---------- Interleaved async writes + queries ----------
    print("Running interleaved writes + reads under lag...")
    next_id = initial_points

    # For plotting
    latest_lsns = []
    replica_lsns = [[], [], []]     # one list per replica
    used_primary_flags = []         # True if query used primary, else False

    for r in range(rounds):
        # --- random write pattern ---
        op = random.choice(["upsert_small", "upsert_batch", "delete_some"])

        if op == "upsert_small":
            pts = make_points(dim, next_id, 5)
            next_id += 5
            baseline.upsert(pts)
            dist.upsert(pts)

        elif op == "upsert_batch":
            pts = make_points(dim, next_id, 20)
            next_id += 20
            baseline.upsert(pts)
            dist.upsert(pts)

        elif op == "delete_some":
            if next_id > 10:
                ids = [f"p{random.randint(0, next_id - 1)}" for _ in range(5)]
                baseline.delete(ids)
                dist.delete(ids)

        # tiny sleep so replica threads sometimes fall behind
        time.sleep(0.001)

        # Snapshot LSN state *before* the query
        wal = dist._wal
        reps = dist._replicas
        latest = wal.latest_lsn()
        latest_lsns.append(latest)
        for i, rep in enumerate(reps):
            replica_lsns[i].append(wal.replica_lsn(rep.id))

        # --- query and compare with baseline ---
        qvec = np.random.randn(dim).astype("float32").tolist()

        hits_base = baseline.query(qvec, top_k)
        hits_dist = dist.query(qvec, top_k)

        ids_base = [h["id"] for h in hits_base]
        ids_dist = [h["id"] for h in hits_dist]

        if ids_base != ids_dist:
            raise AssertionError(
                f"Mismatch at round {r}:\n"
                f"  baseline={ids_base}\n"
                f"  dist    ={ids_dist}"
            )

        used_primary_flags.append(dist._last_used_primary)
        # Let replicas catch up occasionally.
        if r % 10 == 0:
            # every 10th round, give them a real catchup window
            time.sleep(0.05)   # 50 ms → replicas can fully catch up
        else:
            # small delay so they *start* to fall behind
            time.sleep(0.001)

    print("✅ All queries matched baseline under async (laggy) replication.")
    return latest_lsns, replica_lsns, used_primary_flags


def plot_lag_results(latest_lsns, replica_lsns, used_primary_flags):
    rounds = len(latest_lsns)
    x = np.arange(rounds)

    # Figure 1: LSN vs round (shows lag)
    plt.figure(figsize=(10, 6))
    plt.plot(x, latest_lsns, label="latest WAL lsn", linewidth=2)
    for i, lsns in enumerate(replica_lsns):
        plt.plot(x, lsns, label=f"replica-{i} lsn", alpha=0.8)
    plt.xlabel("Round")
    plt.ylabel("LSN")
    plt.title("WAL vs replica progress over time (async replication)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Figure 2: primary vs replica usage over rounds
    plt.figure(figsize=(10, 3))
    used_primary_arr = np.array(used_primary_flags, dtype=int)
    # 1 = primary, 0 = replica
    plt.step(x, used_primary_arr, where="post")
    plt.yticks([0, 1], ["replica", "primary"])
    plt.xlabel("Round")
    plt.ylabel("Source")
    plt.title("Which node served each query (0=replica, 1=primary)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    latest_lsns, replica_lsns, used_primary = run_lag_experiment(
        dim=16,
        initial_points=1000,
        rounds=300,
        top_k=5,
    )
    plot_lag_results(latest_lsns, replica_lsns, used_primary)
