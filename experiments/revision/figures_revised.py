"""Figures for the revised manuscript, read only from results/revised/*.json.

Run: python -m experiments.revision.figures_revised  -> results/revised/figures/*.png
Palette: validated categorical slots (blue, orange, aqua); every series is
also direct-labelled or in a legend, so identity never rests on colour alone.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "results/revised"
F = f"{D}/figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
J = lambda n: json.load(open(f"{D}/{n}.json"))


def auroc_ci(ax, models, title, get):
    names = sorted(models, key=lambda k: get(models[k])[0])
    v = np.array([get(models[k])[0] for k in names])
    lo = np.array([get(models[k])[1][0] for k in names]); hi = np.array([get(models[k])[1][1] for k in names])
    y = np.arange(len(names))
    ax.hlines(y, lo, hi, color=BLUE, lw=2)
    ax.plot(v, y, "o", color=BLUE, ms=6, mec="white", mew=1.5)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(0.5, -0.9, " chance", color=MUTED, fontsize=8, va="center")
    ax.set_ylim(-1.3, len(names) - 0.5)
    ax.set_yticks(y, names); ax.set_xlabel("AUROC (95% block-bootstrap CI)"); ax.set_title(title, loc="left", color=INK)


def fig_auroc():
    sk, sw = J("skab")["pooled"]["models"], J("swat")["standard"]["models"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    auroc_ci(axs[0], sk, "SKAB (pooled protocol)", lambda m: (m["auroc"], m["ci95"]["auroc"]))
    auroc_ci(axs[1], sw, "SWaT (standard protocol)", lambda m: (m["full"]["auroc"], m["full"]["ci95"]["auroc"]))
    fig.tight_layout(); fig.savefig(f"{F}/auroc_ci.png"); plt.close(fig)


def fig_protocol():
    sw = J("swat"); sk = J("skab")
    names = [n for n in sw["standard"]["models"] if n in sw["original_layout_same_code"]["models"]]
    names.sort(key=lambda n: -sw["standard"]["models"][n]["full"]["auroc"])
    a = [sw["original_layout_same_code"]["models"][n]["auroc"] for n in names]
    b = [sw["standard"]["models"][n]["full"]["auroc"] for n in names]
    sknames = [n for n in sk["skab_benchmark_protocol"]["models"]]
    sknames.sort(key=lambda n: -sk["skab_benchmark_protocol"]["models"][n]["mean_file_auroc"])
    c = [sk["pooled"]["models"][n]["auroc"] for n in sknames]
    d = [sk["skab_benchmark_protocol"]["models"][n]["mean_file_auroc"] for n in sknames]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, nm, x1, x2, l1, l2, t in [
        (axs[0], names, a, b, "re-upload layout (original paper)", "standard split", "SWaT: AUROC by split definition"),
        (axs[1], sknames, c, d, "pooled training (original paper)", "per-run, first 400 rows (SKAB benchmark)", "SKAB: AUROC by training protocol")]:
        x = np.arange(len(nm)); w = 0.38
        ax.bar(x - w / 2 - 0.01, x1, w, color=ORANGE, label=l1)
        ax.bar(x + w / 2 + 0.01, x2, w, color=BLUE, label=l2)
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.set_xticks(x, nm, rotation=40, ha="right"); ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("AUROC")
        ax.set_title(t, loc="left", color=INK); ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    fig.tight_layout(); fig.savefig(f"{F}/protocol_sensitivity.png"); plt.close(fig)


def fig_drift():
    dr = J("drift")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    traj = dr["static_if_normal_fpr_trajectory"]
    axs[0].plot(np.arange(len(traj)), np.array(traj) * 100, color=BLUE, lw=2)
    axs[0].axhline(1, color=MUTED, lw=1, ls="--"); axs[0].text(2, 4, "design rate 1%", color=MUTED, fontsize=8)
    axs[0].set_xlabel("hour of streamed normal operation"); axs[0].set_ylabel("false alarms on normal data (%)")
    axs[0].set_title("Static Isolation Forest, normal week", loc="left", color=INK)
    colors = {"oracle": BLUE, "pred_normal": ORANGE, "all": AQUA}
    labels = {"oracle": "true-normal rows (labels)", "pred_normal": "predicted-normal rows", "all": "all rows"}
    pols = ["static", "ks", "periodic", "always"]
    ylab = ["static", "KS-triggered", "periodic", "every batch"]
    for ax, model, title in [(axs[1], "IsolationForest", "Isolation Forest"), (axs[2], "EllipticEnvelope", "Elliptic Envelope")]:
        for i, pol in enumerate(pols):
            if pol == "static":
                v = dr["runs"][f"{model}|static"]
                ax.plot(v["attack_period"]["mcc"], i, "o", color=MUTED, ms=7, mec="white", mew=1.2)
                continue
            for j, (sel, c) in enumerate(colors.items()):
                v = dr["runs"][f"{model}|{pol}|{sel}"]
                ax.plot(v["attack_period"]["mcc"], i + (j - 1) * 0.18, "o", color=c, ms=7, mec="white", mew=1.2,
                        label=labels[sel] if (pol == "ks" and model == "IsolationForest") else None)
        ax.set_yticks(range(len(pols)), ylab); ax.invert_yaxis(); ax.set_xlim(-0.05, 0.8)
        ax.set_xlabel("MCC over the attack period"); ax.set_title(title, loc="left", color=INK)
    axs[2].set_yticklabels([])
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, ["retrain on " + x for x in l], loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.62, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{F}/drift_policies.png"); plt.close(fig)


def fig_edge():
    e = J("edge")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, v in e["models"].items():
        if "error" in v:
            continue
        ds, name = k.split("|")
        c = BLUE if ds == "SKAB" else ORANGE
        ax.plot(v["size_KB"], v["latency_ms_p50"], "o" if ds == "SKAB" else "s", color=c, ms=7, mec="white", mew=1.2)
        if ds == "SWaT":
            ax.annotate(name, (v["size_KB"], v["latency_ms_p50"]), textcoords="offset points", xytext=(5, 2), fontsize=7, color=MUTED)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="SKAB (40 features)"),
                       Line2D([], [], marker="s", ls="", color=ORANGE, label="SWaT (120 features)")], frameon=False, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("serialized model size (KB, log)"); ax.set_ylabel("single-reading latency p50 (ms, log)")
    ax.set_title("Cost per reading in a 1-CPU / 1 GB ARM64 container", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(f"{F}/edge_cost.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(F, exist_ok=True)
    for fn in (fig_auroc, fig_protocol, fig_drift, fig_edge):
        try:
            fn(); print("ok", fn.__name__)
        except FileNotFoundError as ex:
            print("skip", fn.__name__, ex)
