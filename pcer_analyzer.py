#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Regex
# =========================
HDR_RE = re.compile(
    r"PCER values at timestamp\s+(?P<ts>\d+)\s+ps,\s*duration\s+(?P<dur>\d+)\s+cycles",
    re.IGNORECASE
)

ROW_RE = re.compile(
    r"^\s*(\d+)\s*;\s*([^;]+)\s*;.*?;\s*(-?\d+)\s*$"
)

FNAME_RE = re.compile(r"pcer_([0-9a-fA-F]+)\.log$")

# =========================
# Metrics (per-segment)
# =========================
METRICS = [
    "total_cycles",      # header duration
    "cycles_running",    # counter "Cycles"
    "instr",
    "ld",
    "st",
    "imiss",
    "raw_stall",
    "raw_ext_stall",
    "lsu_stall",
    "port_stall",
]


def parse_file(path: Path):
    """Return list of blocks. Each block: {ts_ps, duration, counters{...}}"""
    text = path.read_text(errors="ignore")
    headers = list(HDR_RE.finditer(text))
    blocks = []

    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[start:end]

        counters = {}
        for line in chunk.splitlines():
            m = ROW_RE.match(line)
            if m:
                counters[m.group(2).strip()] = int(m.group(3))

        blocks.append({
            "ts_ps": int(h.group("ts")),
            "duration": int(h.group("dur")),
            "counters": counters
        })

    return blocks


def compute_means_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Means per (segment_index, metric) across all parsed cores that have that segment.
    segment is 1-based in output.
    """
    rows = []
    for seg in sorted(df["segment"].unique()):
        sdf = df[df.segment == seg]
        cores = int(sdf["core"].nunique())
        for m in METRICS:
            rows.append({
                "segment": int(seg),
                "metric": m,
                "mean": float(sdf[m].mean()),
                "cores": cores,
            })
    return pd.DataFrame(rows)


def save_means(df: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    means_df = compute_means_df(df)

    means_df.to_csv(report_dir / "means.csv", index=False)

    lines = []
    lines.append("===== Mean values across all cores =====\n\n")
    for seg in sorted(means_df["segment"].unique()):
        seg_df = means_df[means_df.segment == seg]
        cores = int(seg_df["cores"].iloc[0]) if len(seg_df) else 0
        lines.append(f"Segment {seg} (cores={cores}):\n")
        for _, r in seg_df.iterrows():
            lines.append(f"  {r['metric']:15s}: {r['mean']:.4f}\n")
        lines.append("\n")

    (report_dir / "means.txt").write_text("".join(lines), encoding="utf-8")
    return means_df


def plot_distributions(df: pd.DataFrame, plots_dir: Path):
    for seg in sorted(df["segment"].unique()):
        sdf = df[df.segment == seg]
        for m in METRICS:
            plt.figure()
            sdf[m].plot.hist(bins=50)
            plt.title(f"{m} distribution (segment {seg})")
            plt.xlabel(m)
            plt.ylabel("core count")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{m}_seg{seg}.png", dpi=200)
            plt.close()


def total_cycles_sum_check(df: pd.DataFrame, report_dir: Path, plots_dir: Path):
    """
    For each core, sum total_cycles across all segments present in that core's file.
    Then check consistency across cores (median, max/min delta).
    """
    # 每个 core 的总周期 = 各段 total_cycles 求和
    sum_df = df.groupby("core", as_index=False)["total_cycles"].sum()
    sum_df = sum_df.rename(columns={"total_cycles": "total_cycles_sum"})

    # 画分布
    plt.figure()
    sum_df["total_cycles_sum"].plot.hist(bins=50)
    plt.xlabel("sum(total_cycles over all segments)")
    plt.ylabel("core count")
    plt.title("Total cycles sum across all segments")
    plt.tight_layout()
    plt.savefig(plots_dir / "total_cycles_sum.png", dpi=200)
    plt.close()

    ref = int(sum_df["total_cycles_sum"].median())
    delta = sum_df["total_cycles_sum"] - ref
    max_delta = int(delta.max())
    min_delta = int(delta.min())

    (report_dir / "total_cycles_sum_check.txt").write_text(
        "\n".join([
            "===== total_cycles_sum deviation check (all segments) =====",
            f"cores: {len(sum_df)}",
            f"median_total_cycles_sum: {ref}",
            f"max_delta: {max_delta}",
            f"min_delta: {min_delta}",
        ]),
        encoding="utf-8"
    )

    return {
        "cores": int(len(sum_df)),
        "median": ref,
        "max_delta": max_delta,
        "min_delta": min_delta,
    }


def write_summary_md(
    report_dir: Path,
    log_dir: Path,
    parsed_files: int,
    parsed_cores: int,
    segments_global_max: int,
    means_df: pd.DataFrame,
    cycle_sum_stats: dict,
):
    plots_dir = report_dir / "plots"

    def means_table(seg: int) -> str:
        seg_df = means_df[means_df.segment == seg][["metric", "mean", "cores"]].copy()
        if seg_df.empty:
            return "_No data_\n"
        seg_df["mean"] = seg_df["mean"].map(lambda x: f"{x:.4f}")
        return seg_df.to_markdown(index=False)

    md = []
    md.append("# PCER Report Summary\n")
    md.append(f"- Input directory: `{log_dir}`\n")
    md.append(f"- Output directory: `{report_dir}`\n")
    md.append(f"- Parsed files: **{parsed_files}**\n")
    md.append(f"- Parsed cores: **{parsed_cores}**\n")
    md.append(f"- Detected segments (max across files): **{segments_global_max}**\n")
    md.append(
        "\n> Note: Different cores may have different number of segments. "
        "Per-segment means are computed over cores that contain that segment.\n"
    )

    md.append("\n## Segment means (across cores that have the segment)\n")
    for seg in sorted(means_df["segment"].unique()):
        md.append(f"\n### Segment {seg}\n")
        md.append(means_table(int(seg)) + "\n")

    md.append("\n## Total cycles sum consistency check\n")
    md.append(
        f"- `total_cycles_sum = sum(total_cycles over all segments in that core)`\n"
        f"- Median: **{cycle_sum_stats['median']}**\n"
        f"- Max delta (vs median): **{cycle_sum_stats['max_delta']}**\n"
        f"- Min delta (vs median): **{cycle_sum_stats['min_delta']}**\n"
        f"- Interpretation: deltas should be small if cores cover the same time window; "
        f"large tails suggest segment coverage mismatch across cores.\n"
    )

    md.append("\n## Artifacts\n")
    md.append("- `pcer_all_cores.csv`: per-core, per-segment extracted counters\n")
    md.append("- `means.csv` / `means.txt`: per-segment mean values\n")
    md.append("- `total_cycles_sum_check.txt`: cycle sum deviation summary\n")
    md.append("- `plots/`: histograms for each metric × each detected segment, plus `total_cycles_sum.png`\n")

    # 给出关键图的相对路径（存在则列出）
    key = ["total_cycles_sum.png"]
    for seg in sorted(means_df["segment"].unique()):
        for m in ["total_cycles", "cycles_running", "instr", "imiss", "raw_stall", "raw_ext_stall", "lsu_stall", "port_stall", "ld", "st"]:
            key.append(f"{m}_seg{int(seg)}.png")

    existing = [p for p in key if (plots_dir / p).exists()]
    if existing:
        md.append("\n### Quick links to plots\n")
        for p in existing:
            md.append(f"- `plots/{p}`\n")

    (report_dir / "summary.md").write_text("".join(md), encoding="utf-8")


def main(log_dir: str):
    log_dir = Path(log_dir)

    report_dir = log_dir / "report"
    plots_dir = report_dir / "plots"
    report_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    rows = []
    parsed_cores = set()
    parsed_files = 0
    segments_global_max = 0

    # 自动扫描 pcer_*.log
    for path in sorted(log_dir.glob("pcer_*.log")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue

        core_id = int(m.group(1), 16)

        blocks = parse_file(path)
        if len(blocks) < 1:
            continue

        parsed_files += 1
        parsed_cores.add(core_id)
        segments_global_max = max(segments_global_max, len(blocks))

        for seg_idx, b in enumerate(blocks, start=1):  # 1-based segment index
            c = b["counters"]

            rows.append({
                "core": core_id,
                "segment": seg_idx,
                "ts_ps": b["ts_ps"],

                "total_cycles": b["duration"],
                "cycles_running": c.get("Cycles", 0),
                "instr": c.get("instr", 0),
                "ld": c.get("ld", 0),
                "st": c.get("st", 0),
                "imiss": c.get("imiss", 0),
                "raw_stall": c.get("raw_stall", 0),
                "raw_ext_stall": c.get("raw_ext_stall", 0),
                "lsu_stall": c.get("lsu_stall", 0),
                "port_stall": c.get("port_stall", 0),
            })

    if not rows:
        raise SystemExit(f"[ERROR] No valid pcer_*.log files found in {log_dir}")

    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "pcer_all_cores.csv", index=False)

    # means
    means_df = save_means(df, report_dir)

    # plots
    plot_distributions(df, plots_dir)

    # cycle sum check across ALL segments
    cycle_sum_stats = total_cycles_sum_check(df, report_dir, plots_dir)

    # summary.md
    write_summary_md(
        report_dir=report_dir,
        log_dir=log_dir,
        parsed_files=parsed_files,
        parsed_cores=len(parsed_cores),
        segments_global_max=segments_global_max,
        means_df=means_df,
        cycle_sum_stats=cycle_sum_stats,
    )

    print(f"[OK] Report written to: {report_dir}")
    print(f"[INFO] Parsed files: {parsed_files}, parsed cores: {len(parsed_cores)}, max segments: {segments_global_max}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze PCER logs (variable core count & variable segment count) and write results into <log_dir>/report/"
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=".",
        help="Directory containing pcer_*.log files (default: current directory)"
    )
    args = parser.parse_args()
    main(args.log_dir)
