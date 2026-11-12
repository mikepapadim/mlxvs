# Copyright © 2026 mikepapadim

"""Plots bench_search.py results: queries per second against recall@10.

Reads the printed output of bench_search.py runs (one file per dataset and
dtype) and draws one panel per dataset, light and dark, as in the README:

    python bench/plot_bench.py docs/img/bench \\
        "SIFT1M: 985k x 128, L2" sift_f32.txt sift_f16.txt \\
        "ada-002: 983k x 1536, cosine" ada_f32.txt ada_f16.txt

Each panel is a title followed by its files: the first holds the float32 run
(and the hnswlib baseline, if run), the second the float16 run. Writes
<prefix>-light.png and <prefix>-dark.png. Needs matplotlib.
"""

import re
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

ROW = re.compile(r"^(mlxvs .*?|hnswlib \(CPU\))\s+(\d+)\s+([\d.]+)\s+([\d,]+)\s*$")


def parse(path):
    """{series: [(recall, qps)]} and the build seconds of mlxvs and hnswlib."""
    series, builds = {}, {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        m = ROW.match(line.rstrip())
        if m:
            name = "hnswlib" if m[1].startswith("hnswlib") else "mlxvs"
            series.setdefault(name, []).append(
                (float(m[3]), float(m[4].replace(",", "")))
            )
        elif line.startswith("graph:"):
            builds["mlxvs"] = float(line.rsplit(",", 1)[1].split()[0])
        elif line.startswith("hnswlib build"):
            builds["hnswlib"] = float(line.rsplit(":", 1)[1].split()[0])
    return series, builds


def panels(args):
    out, title = [], None
    for a in args:
        if a.endswith(".txt"):
            out[-1][1].append(a)
        else:
            title = a
            out.append((title, []))
    return out


def qps_label(v, _):
    return f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"


def draw(prefix, spec, dark):
    fg, bg, grid, sub = (
        ("#e8e8e8", "#16181c", "#33363c", "#a8abb0")
        if dark
        else ("#111111", "#fbfbf9", "#e3e3e3", "#555555")
    )
    colors = {
        "float32": "#3b8beb" if dark else "#2878d8",
        "float16": "#f07a45" if dark else "#e8673a",
        "hnswlib": "#2fbf8f" if dark else "#1aa878",
    }
    fig, axes = plt.subplots(1, len(spec), figsize=(16.5, 7.4), facecolor=bg)
    if len(spec) == 1:
        axes = [axes]
    for ax, (title, files) in zip(axes, spec):
        ax.set_facecolor(bg)
        f32, b32 = parse(files[0])
        lines = [("mlxVS float32 (GPU)", f32.get("mlxvs", []), "float32", "o")]
        if len(files) > 1:
            f16, _ = parse(files[1])
            lines.append(("mlxVS float16 (GPU)", f16.get("mlxvs", []), "float16", "s"))
        lines.append(("hnswlib (12 CPU cores)", f32.get("hnswlib", []), "hnswlib", "^"))
        for label, pts, key, marker in lines:
            if not pts:
                continue
            r, q = zip(*sorted(pts))
            ax.plot(
                r,
                q,
                marker=marker,
                color=colors[key],
                lw=2.6,
                ms=8,
                mec=bg,
                label=label,
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(qps_label))
        ax.grid(True, color=grid, lw=1)
        ax.set_axisbelow(True)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(colors=sub, labelsize=13, length=0)
        ax.set_xlabel("Recall@10", color=sub, fontsize=14)
        ax.set_title(title, loc="left", color=fg, fontsize=16, fontweight="bold")
        if "mlxvs" in b32 and "hnswlib" in b32:
            ax.text(
                0,
                -0.2,
                f"Index built in {b32['mlxvs']:.3g} s on the GPU, "
                f"{b32['hnswlib']:.3g} s with hnswlib",
                transform=ax.transAxes,
                color=sub,
                fontsize=13,
            )
    axes[0].set_ylabel("Queries per second (log scale)", color=sub, fontsize=14)
    leg = axes[0].legend(frameon=False, fontsize=13, loc="lower left")
    for t in leg.get_texts():
        t.set_color(fg)
    fig.suptitle(
        "mlxVS on a MacBook Pro M3 Pro: queries per second against recall "
        "(up and right is better)",
        x=0.012,
        y=0.975,
        ha="left",
        color=fg,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.915,
        "10,000 queries per batch, k = 10; hnswlib M=16, efC=200 on all 12 CPU cores; "
        "mlxVS CAGRA graph of degree 32",
        color=sub,
        fontsize=13,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.8, bottom=0.2, wspace=0.16)
    fig.savefig(f"{prefix}-{'dark' if dark else 'light'}.png", dpi=100, facecolor=bg)
    plt.close(fig)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    spec = panels(sys.argv[2:])
    for dark in (False, True):
        draw(sys.argv[1], spec, dark)


if __name__ == "__main__":
    main()
