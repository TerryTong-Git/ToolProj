import os
import json
import numpy as np
import matplotlib.pyplot as plt

def save_summary_plot(out_dir: str, tag: str,
                      avg_logps_code: list, avg_logps_nl: list,
                      kl_code: dict, kl_nl: dict):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(8,4.5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.boxplot([avg_logps_code, avg_logps_nl], labels=["code z", "NL z"])
    ax1.set_title("Avg log-prob of (z,x|θ,r)")
    ax1.set_ylabel("avg log p per token")
    ax2 = fig.add_subplot(1,2,2)
    x = np.arange(3)
    ax2.bar(x - 0.15, [kl_code["max_kl"], kl_code["avg_kl"], kl_code["var_kl"]], width=0.3, label="code z")
    ax2.bar(x + 0.15, [kl_nl["max_kl"], kl_nl["avg_kl"], kl_nl["var_kl"]], width=0.3, label="NL z")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["max KL", "avg KL", "var KL"])
    ax2.legend()
    fig.suptitle(f"NL vs Code intermediaries — {tag}")
    out_png = os.path.join(out_dir, f"summary_{tag}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    return out_png
