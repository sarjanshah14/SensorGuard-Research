"""Resource-cost benchmark redone (revision/AUDIT.md E5). Measures cost only; no
accuracy is computed here (the original in-container SWaT score was computed on
training rows, audit G).

Run inside the capped container (see docker/Dockerfile.edge2):
  docker run --rm --cpus=1 --memory=1g -v "$PWD":/app -v <swat_dir>:/swat \
      -e SWAT_DIR=/swat sensorguard-edge2
-> results/revised/edge.json

Per model, each in a fresh *spawned* process (a forked child's RSS also counts
pages shared with its parent, which inflated the first version's numbers):
  fit time, serialized size, load time, single-reading latency (1,000 calls,
  p50/p95/p99), batch throughput (rows/s), and memory as
    base_rss_MB   : process RSS after imports + data loading,
    model_peak_MB : peak RSS increase during fit + scoring (VmHWM reset after loading),
  plus the container-wide peak (cgroup memory.peak) and OOM-kill count. Plus the per-reading
  cost of updating the rolling features from a 30-row buffer.
"""

import io
import json
import multiprocessing as mp
import os
import platform
import resource
import time

import joblib
import numpy as np
import pandas as pd

OUT = "results/revised/edge.json"


def cpu_limit():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(p)
    except OSError:
        return None


def mem_limit_mb():
    try:
        v = open("/sys/fs/cgroup/memory.max").read().strip()
        return None if v == "max" else int(v) / 2**20
    except OSError:
        return None


def data(dataset):
    from sklearn.preprocessing import StandardScaler
    from src.features import build_features
    if dataset == "SKAB":
        from experiments.revision.skab_revised import frames
        df, X, _ = frames()
        Xtr, Xte = X[df.is_train.to_numpy()], X[df.is_test.to_numpy()]
    else:  # SWaT: 50k genuine-normal rows to fit, 20k attack-period rows to score (timing only)
        from experiments.revision.swat_revised import SENSORS
        f = os.path.join(os.environ["SWAT_DIR"], "normal.csv")
        rd = lambda skip, n: pd.read_csv(f, skipinitialspace=True, skiprows=range(1, 1 + skip), nrows=n)
        tr, te = rd(395298 + 21600, 50000), rd(0, 20000)
        for d in (tr, te):
            d.columns = d.columns.str.strip()
        Xtr, _ = build_features(tr.set_index(pd.RangeIndex(len(tr))), sensor_cols=SENSORS, include_elapsed=False)
        Xte, _ = build_features(te.set_index(pd.RangeIndex(len(te))), sensor_cols=SENSORS, include_elapsed=False)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def vm(field):
    for line in open("/proc/self/status"):
        if line.startswith(field):
            return int(line.split()[1]) / 1024  # kB -> MB
    return float("nan")


def bench(dataset, name, q):
    from threadpoolctl import threadpool_limits
    from experiments.revision.common import make_models
    with threadpool_limits(1):
        Xtr, Xte = data(dataset)
        base = vm("VmRSS:")
        try:
            open("/proc/self/clear_refs", "w").write("5")  # reset VmHWM to current RSS
        except OSError:
            pass
        m = make_models(seed=0, cap=20000)[name]
        t = time.perf_counter(); m.fit(Xtr); fit_s = time.perf_counter() - t
        buf = io.BytesIO(); joblib.dump(m, buf); size_kb = buf.tell() / 1024
        buf.seek(0); t = time.perf_counter(); m = joblib.load(buf); load_s = time.perf_counter() - t
        m.score(Xte[:1])  # warm-up
        lat = []
        for i in range(1000):
            x = Xte[i % len(Xte): i % len(Xte) + 1]
            t = time.perf_counter(); m.score(x); lat.append((time.perf_counter() - t) * 1000)
        nb = min(10000, len(Xte))
        t = time.perf_counter(); m.score(Xte[:nb]); thr = nb / (time.perf_counter() - t)
    q.put({"fit_s": round(fit_s, 3), "size_KB": round(size_kb, 1), "load_ms": round(load_s * 1000, 2),
           "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
           "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
           "latency_ms_p99": round(float(np.percentile(lat, 99)), 3),
           "batch_rows_per_s": round(thr), "base_rss_MB": round(base, 1), "model_peak_MB": round(max(0.0, vm("VmHWM:") - base), 1)})


def bench_deep(q):
    import torch
    from src.deep_baseline import LSTMAutoencoder
    torch.set_num_threads(1)
    for nf in (8, 24):
        m = LSTMAutoencoder(nf).eval()
        x = torch.randn(1, 30, nf)
        with torch.no_grad():
            m(x)
            lat = []
            for _ in range(500):
                t = time.perf_counter(); m(x); lat.append((time.perf_counter() - t) * 1000)
        q.put((nf, {"params": sum(p.numel() for p in m.parameters()),
                    "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
                    "latency_ms_p95": round(float(np.percentile(lat, 95)), 3)}))
    q.put(("rss", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)))


def feature_update_cost():
    """Cost of computing the 5-statistic features for one new reading from a
    30-row buffer with the pipeline's pandas code (naive, recomputes the buffer)."""
    from src.features import SENSOR_COLS, build_features
    buf = pd.DataFrame(np.random.rand(30, 8), columns=SENSOR_COLS, index=pd.date_range("2020", periods=30, freq="s"))
    lat = []
    for _ in range(300):
        t = time.perf_counter(); build_features(buf, include_elapsed=False); lat.append((time.perf_counter() - t) * 1000)
    return {"latency_ms_p50": round(float(np.percentile(lat, 50)), 3), "latency_ms_p95": round(float(np.percentile(lat, 95)), 3)}


def cgroup(name):
    try:
        return open(f"/sys/fs/cgroup/{name}").read().strip()
    except OSError:
        return None


def child(fn, *args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=fn, args=(*args, q)); p.start()
    items = []
    while p.is_alive() or not q.empty():
        try:
            items.append(q.get(timeout=1))
        except Exception:
            pass
    p.join()
    if p.exitcode not in (0, None) and not items:
        items.append({"error": f"child exited with code {p.exitcode} (-9 = killed, e.g. out of memory)"})
    return items


def main():
    from experiments.revision.common import make_models
    os.makedirs("results/revised", exist_ok=True)
    out = {"env": {"machine": platform.machine(), "python": platform.python_version(), "cgroup_cpus": cpu_limit(),
                   "cgroup_mem_MB": mem_limit_mb(), "note": "Docker container with CPU/RAM caps on an Apple-M4 host; not a physical edge device"},
           "feature_update_per_reading": feature_update_cost(), "models": {}}
    for ds in ["SKAB", "SWaT"]:
        for name in make_models():
            r = child(bench, ds, name)
            out["models"][f"{ds}|{name}"] = r[0] if r else {"error": "child process failed (likely OOM)"}
            print(ds, name, out["models"][f"{ds}|{name}"], flush=True)
    d = child(bench_deep)
    out["lstm_ae"] = ({("peak_rss_MB" if k == "rss" else f"{k}_features"): v for k, v in d}
                      if d and isinstance(d[0], tuple) else {"error": d})
    out["container"] = {"memory_max": cgroup("memory.max"), "memory_peak_bytes": cgroup("memory.peak"),
                        "memory_events": cgroup("memory.events")}
    print(out["lstm_ae"])
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
