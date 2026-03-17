import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_score_plot(scores, gt_slice, video_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    base = str(video_name)

    T = len(scores)
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 3))
    # ax.plot(x, scores, lw=1.8, label="Anomaly score")
    ax.plot(x, scores, lw=1.8)

    # 高亮 GT 区间
    in_anom = (gt_slice > 0).astype(int)
    start = None
    for t, v in enumerate(in_anom):
        if v == 1 and start is None:
            start = t
        if (v == 0 and start is not None) or (t == T - 1 and start is not None):
            end = t if v == 0 else t + 1
            ax.axvspan(start, end, color="tab:red", alpha=0.22)
            start = None

    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Frame idx")
    ax.set_ylabel("Score")
    # ax.legend(loc="upper left", frameon=False)

    fig.suptitle(base, fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(os.path.join(out_dir, f"{base}.pdf"))
    plt.close(fig)

