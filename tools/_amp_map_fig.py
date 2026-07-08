"""Quick fidelity+survival heatmap for an imaging amp scan (from the printed tables)."""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# args: out.png  amp1_csv  amp2_csv  fid_flat_csv  surv_flat_csv  title
out, a1s, a2s, fids, survs, title = sys.argv[1:7]
a1 = [float(v) for v in a1s.split(",")]
a2 = [float(v) for v in a2s.split(",")]
F = np.array([float(v) for v in fids.split(",")]).reshape(len(a2), len(a1))
S = np.array([float(v) for v in survs.split(",")]).reshape(len(a2), len(a1))
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, M, ttl, cm, vlo in [(axs[0], F, "fidelity", "viridis", 0.93),
                             (axs[1], S, "survival (P11)", "magma", 0.0)]:
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cm, vmin=vlo, vmax=1.0,
                   extent=[min(a1)-0.015, max(a1)+0.015, min(a2)-0.015, max(a2)+0.015])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Imag399.Amp1"); ax.set_ylabel("Imag399.Amp2"); ax.set_title(ttl)
    for i, y in enumerate(a2):
        for j, x in enumerate(a1):
            ax.text(x, y, "%.3f" % M[i, j], ha="center", va="center", fontsize=6,
                    color="w" if M[i, j] < (0.985 if ttl.startswith("fid") else 0.7) else "k")
fig.suptitle(title, fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
