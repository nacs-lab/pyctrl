"""Analyze an interleaved-W scan: per-config real survival + per-site d'/fidelity, with SEM and a
pairwise z-test. Configs are cells of ONE scrambled axis, so drift is common-mode."""
import json, os, sys, numpy as np, h5py
REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
sys.path.insert(0, REPO)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

def analyze(data_dir, names=None):
    sid = os.path.basename(data_dir.rstrip("/\\"))
    cfg = json.load(open(os.path.join(data_dir, sid + ".json")))
    P = np.asarray(cfg["Params"]).ravel().astype(int)
    with h5py.File(os.path.join(data_dir, sid + ".h5"), "r") as f:
        sq = f["seq_ids"][:]; I1 = f["intensities_img1"][:].astype(float)
        L1 = f["logicals_img1"][:].astype(bool); L2 = f["logicals_img2"][:].astype(bool)
    n = min(len(sq), I1.shape[0]); sq, I1, L1, L2 = sq[:n], I1[:n], L1[:n], L2[:n]
    flat = P[sq - 1] - 1
    out = []
    print("scan %s  n_shots=%d  cells=%d" % (sid.replace("data_", ""), n, flat.max() + 1))
    print("%-22s %8s %9s %8s %7s %7s %7s" % ("config", "n", "survival", "+-SEM", "d'", "dist", "load"))
    for p in range(flat.max() + 1):
        r = np.where(flat == p)[0]
        if not r.size:
            out.append(None); continue
        l1, l2 = L1[r], L2[r]
        ps = np.array([(l1[k] & l2[k]).sum() / max(l1[k].sum(), 1) for k in range(len(r))])
        I, L = I1[r].ravel(), l1.ravel()
        a, b = I[L], I[~L]
        dp = (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        rec = {"p": p, "n": len(r), "surv": ps.mean(), "sem": ps.std(ddof=1) / np.sqrt(len(ps)),
               "dprime": dp, "dist": a.mean() - b.mean(), "load": l1.mean(), "shots": ps}
        out.append(rec)
        print("%-22s %8d %9.4f %8.4f %7.2f %7.2f %7.3f"
              % ((names[p] if names else "cell%d" % p), rec["n"], rec["surv"], rec["sem"],
                 rec["dprime"], rec["dist"], rec["load"]))
    print("\npairwise (Welch z on per-shot survival):")
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if not out[i] or not out[j]: continue
            d = out[j]["surv"] - out[i]["surv"]; s = np.hypot(out[i]["sem"], out[j]["sem"])
            print("  %-20s -> %-20s  %+0.4f  (%.1f sigma)%s"
                  % (names[i] if names else i, names[j] if names else j, d, abs(d) / s if s else 0,
                     "  SIGNIFICANT" if s and abs(d) / s > 2 else ""))
    return out

def figure(data_dir, recs, names, xs, xlabel, out=None):
    """Survival +- SEM and pooled d' vs the swept quantity, for the Notion entry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keep = [i for i, r in enumerate(recs) if r]
    x = np.asarray([xs[i] for i in keep], float)
    o = np.argsort(x)
    x = x[o]
    S = np.asarray([recs[keep[i]]["surv"] for i in o])
    E = np.asarray([recs[keep[i]]["sem"] for i in o])
    D = np.asarray([recs[keep[i]]["dprime"] for i in o])
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    ax.errorbar(x, S, yerr=E, fmt="o-", color="#C44E52", capsize=3)
    ax.set_xlabel(xlabel); ax.set_ylabel("survival", color="#C44E52")
    ax.tick_params(axis="y", labelcolor="#C44E52"); ax.grid(alpha=0.3)
    ax2 = ax.twinx(); ax2.plot(x, D, "s--", color="#4C72B0")
    ax2.set_ylabel("pooled d-prime", color="#4C72B0"); ax2.tick_params(axis="y", labelcolor="#4C72B0")
    sid = os.path.basename(data_dir.rstrip("/\\"))
    ax.set_title("Interleaved (drift-immune) comparison: one scrambled scan\n%s" % sid, fontsize=9)
    out = out or os.path.join(data_dir, "interleaved_comparison.png")
    fig.text(0.005, 0.005, out, fontsize=5.5, color="0.45", ha="left", va="bottom")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    dd = sys.argv[1]
    nm = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    recs = analyze(dd, nm)
    # --fig X1,X2,...  -> also plot survival/d' against that numeric axis
    if "--fig" in sys.argv:
        vals = [float(v) for v in sys.argv[sys.argv.index("--fig") + 1].split(",")]
        lab = sys.argv[sys.argv.index("--fig") + 2] if len(sys.argv) > sys.argv.index("--fig") + 2 else "config"
        figure(dd, recs, nm, vals, lab)
