"""Build consolidated result tables and figures from committed summary JSON files.

RACA-Collective.txt Section 232 (`analysis/`) calls this directory "result analysis scripts."
This script is the first one: it reads the small, human-readable summary JSON files that back
the numbers reported in docs/PHASE_*.md (tracked in git as of the .gitignore fix described in
docs/RESEARCH_LOG.md; previously experiments/**/results/ was entirely gitignored) and produces:

  (a) analysis/output/consolidated_summary.csv and .md -- one row per experimental condition
      (B0-B2/Oracle/B3 for phases 1-3, C0-C3 for memory, A0-A7 for the ablation matrix, H0/H1 for
      heterogeneity, plus the scaling and perturbation conditions) with its headline
      utility/regret number and the source file it came from.
  (b) analysis/output/flip_rate_vs_injection_depth.png (or .md table if matplotlib is
      unavailable) -- the flip-rate-vs-injection-depth figure proposed in
      paper/MANUSCRIPT_NOTES.md Section 7, from experiments/phase_o/results/injection_depth_summary.json.
  (c) analysis/output/gini_vs_peer_capacity.png (or .md table) -- the Gini-vs-peer_capacity
      figure proposed in the same section, from
      experiments/phase_o/results/peer_capacity_ablation_summary.json.

Matplotlib is used only here (analysis/), never in src/, and only if available -- this script
checks with a guarded import and falls back to a clearly-labeled markdown table (with axes
named in the table itself) if matplotlib is not installed, per this project's rule against
adding a new hard dependency just for plotting.

PNG reproducibility: by default matplotlib embeds a "Software" tEXt chunk (its own version
string) and, depending on backend/version, other metadata that can vary between otherwise
identical runs. This script passes an explicit, fixed `metadata` dict to every `savefig(...)`
call (empty `Software`/`Creator`/etc. strings) so that two consecutive runs against the same
input JSON produce byte-identical PNG files. This was verified directly: running this script
twice in a row and comparing `hashlib.sha256(open(path, "rb").read()).hexdigest()` for both
`analysis/output/*.png` files gives matching hashes on the second run.

Usage: python3 -m analysis.build_summary_tables
"""
from __future__ import annotations

import csv
import json
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Fixed PNG metadata so that two consecutive runs on identical input data produce byte-identical
# PNG files. Without this, matplotlib's Agg/PNG writer embeds its own version string (and, on some
# backends, a timestamp) in a "Software"/"Creator" tEXt chunk, which changes across matplotlib
# versions/environments even when every pixel is identical. Empty strings suppress those chunks.
_DETERMINISTIC_PNG_METADATA = {
    "Software": "",
    "Creator": "",
    "Author": "",
    "Description": "",
}


def _load(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _row(phase, condition, metric, value, ci_lo, ci_hi, source):
    return {
        "phase": phase,
        "condition": condition,
        "metric": metric,
        "value": value,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "source_file": source,
    }


def _metric_from_dict(d, keys):
    """Pull a {mean, ci_lo, ci_hi} or {mean_utility, ci_lo, ci_hi} style block out of d,
    trying each candidate key in order. Returns (metric_name, value, ci_lo, ci_hi) or None."""
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, dict):
                mean = v.get("mean", v.get("mean_utility"))
                return k, mean, v.get("ci_lo"), v.get("ci_hi")
            else:
                return k, v, d.get("ci_lo"), d.get("ci_hi")
    return None


def rows_phase_a():
    d = _load("experiments/phase_a/summary.json")
    if d is None:
        return []
    src = "experiments/phase_a/summary.json"
    out = []
    for condition, stats in d.items():
        m = _metric_from_dict(stats, ["mean_regret"])
        if m:
            _, val, lo, hi = m
            out.append(_row("Phase 1 (baselines)", condition, "mean_regret", val, lo, hi, src))
    return out


def rows_phase_c():
    d = _load("experiments/phase_c/summary.json")
    if d is None:
        return []
    src = "experiments/phase_c/summary.json"
    out = []
    for config_name in ("small_dense", "large_sparse"):
        cfg = d.get(config_name, {})
        held_out = cfg.get("held_out_test", cfg.get("test", {}))
        for label, key in (("B2_heuristic", "b2_regret_vs_oracle"), ("B3_learned", "b3_regret_vs_oracle")):
            if key in held_out:
                out.append(_row("Phase 3 (learned router)", f"{config_name}__{label}",
                                 "regret_vs_oracle", held_out[key], None, None, src))
    return out


def rows_phase_e():
    d = _load("experiments/phase_e/summary.json")
    if d is None:
        return []
    src = "experiments/phase_e/summary.json"
    out = []
    for condition, stats in d.items():
        m = _metric_from_dict(stats, ["mean_utility"])
        if m:
            _, val, lo, hi = m
            out.append(_row("Phase 5 (memory C0-C3)", condition, "mean_utility", val, lo, hi, src))
    return out


def rows_phase_k():
    d = _load("experiments/phase_k/results/ablation_summary.json")
    if d is None:
        return []
    src = "experiments/phase_k/results/ablation_summary.json"
    out = []
    for condition, stats in d.items():
        if not isinstance(stats, dict) or "mean_utility" not in stats:
            continue
        out.append(_row("Phase 11 (ablation matrix)", condition, "mean_utility",
                         stats["mean_utility"], stats.get("ci_lo"), stats.get("ci_hi"), src))
    return out


def rows_phase_l():
    d = _load("experiments/phase_l/results/ablation_full_summary.json")
    if d is None:
        return []
    src = "experiments/phase_l/results/ablation_full_summary.json"
    out = []
    for condition, stats in d.items():
        if not isinstance(stats, dict) or "mean_utility" not in stats:
            continue
        out.append(_row("Phase 12 (A2/A7 ablation)", condition, "mean_utility",
                         stats["mean_utility"], stats.get("ci_lo"), stats.get("ci_hi"), src))
    return out


def rows_phase_j():
    d = _load("experiments/phase_j/results/heterogeneity_summary.json")
    if d is None:
        return []
    src = "experiments/phase_j/results/heterogeneity_summary.json"
    out = []
    for condition, stats in d.items():
        if not isinstance(stats, dict):
            continue
        out.append(_row("Phase 10 (heterogeneity H0/H1)", condition, "mean_utility",
                         stats.get("mean_utility"), None, None, src))
    return out


def rows_phase_i():
    out = []
    d = _load("experiments/phase_i/results/scaling_summary.json")
    if d is not None:
        src = "experiments/phase_i/results/scaling_summary.json"
        for size, stats in d.items():
            out.append(_row("Phase 9 (scaling)", f"N={size}", "mean_utility_mean",
                             stats.get("mean_utility_mean"), None, None, src))
    d100 = _load("experiments/phase_i/results/scaling_n100_summary.json")
    if d100 is not None:
        src = "experiments/phase_i/results/scaling_n100_summary.json"
        out.append(_row("Phase 9 revised (N=100)", "N=100", "mean_utility_mean",
                         d100.get("mean_utility_mean"), None, None, src))
    return out


def rows_phase_h():
    out = []
    d = _load("experiments/phase_h/results/perturbation_summary.json")
    if d is not None:
        src = "experiments/phase_h/results/perturbation_summary.json"
        out.append(_row("Phase 8 (perturbation, n=6)", "hub_removal", "mean_utility_drop",
                         d.get("mean_utility_drop"), None, None, src))
    d30 = _load("experiments/phase_h/results/perturbation_summary_30seeds.json")
    if d30 is not None:
        src = "experiments/phase_h/results/perturbation_summary_30seeds.json"
        ci = d30.get("utility_drop_95ci_bootstrap", [None, None])
        out.append(_row("Phase 8 revised (perturbation, n=30)", "hub_removal", "mean_utility_drop",
                         d30.get("mean_utility_drop"), ci[0], ci[1], src))
    return out


def build_consolidated_table():
    rows = []
    for fn in (rows_phase_a, rows_phase_c, rows_phase_e, rows_phase_k, rows_phase_l,
               rows_phase_j, rows_phase_i, rows_phase_h):
        rows.extend(fn())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "consolidated_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "condition", "metric", "value", "ci_lo", "ci_hi", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(OUTPUT_DIR, "consolidated_summary.md")
    with open(md_path, "w") as f:
        f.write("# Consolidated summary table\n\n")
        f.write("Generated by `analysis/build_summary_tables.py` from the committed summary JSON "
                "files listed in the `source_file` column. One row per experimental condition.\n\n")
        f.write("| Phase | Condition | Metric | Value | 95% CI lo | 95% CI hi | Source |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            def fmt(x):
                return f"{x:.4f}" if isinstance(x, (int, float)) else ("" if x is None else str(x))
            f.write(f"| {r['phase']} | {r['condition']} | {r['metric']} | {fmt(r['value'])} | "
                     f"{fmt(r['ci_lo'])} | {fmt(r['ci_hi'])} | `{r['source_file']}` |\n")

    print(f"Wrote {len(rows)} rows to {csv_path} and {md_path}")
    return rows


def build_flip_rate_figure():
    d = _load("experiments/phase_o/results/injection_depth_summary.json")
    if d is None:
        print("Skipping flip-rate figure: experiments/phase_o/results/injection_depth_summary.json not found")
        return
    # Schema: {"by_depth": {"0.0": {"target_round": 0, "flip_rate": ..., "n_valid_perturbation_points": ...}, ...}}
    entries = d.get("by_depth", {})
    depths, flip_rates, n_valid = [], [], []
    items = sorted(entries.items(), key=lambda kv: float(kv[0]))
    for k, v in items:
        depths.append(v.get("target_round", k))
        flip_rates.append(v.get("flip_rate"))
        n_valid.append(v.get("n_valid_perturbation_points"))

    if not flip_rates:
        print("Skipping flip-rate figure: could not find flip_rate entries in injection_depth_summary.json "
              "(schema differs from expected; see raw JSON for exact keys)")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(len(depths)), flip_rates, marker="o")
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([str(x) for x in depths])
        ax.set_xlabel("Injection round (perturbation depth)")
        ax.set_ylabel("Hub-identity flip rate")
        ax.set_title("Flip rate vs. injection depth (Phase O, seeds 40000-40029)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "flip_rate_vs_injection_depth.png")
        fig.savefig(out_path, dpi=150, metadata=_DETERMINISTIC_PNG_METADATA)
        plt.close(fig)
        print(f"Wrote {out_path}")
    else:
        out_path = os.path.join(OUTPUT_DIR, "flip_rate_vs_injection_depth.md")
        with open(out_path, "w") as f:
            f.write("# Flip rate vs. injection depth (Phase O, seeds 40000-40029)\n\n")
            f.write("matplotlib not available; rendered as a table. X axis: injection round "
                    "(perturbation depth). Y axis: hub-identity flip rate (fraction of seeds "
                    "where the final delegate-in hub identity differs after perturbation).\n\n")
            f.write("| Injection round | Flip rate | n valid seeds |\n|---|---|---|\n")
            for dep, fr, n in zip(depths, flip_rates, n_valid):
                f.write(f"| {dep} | {fr} | {n} |\n")
        print(f"matplotlib unavailable, wrote table instead: {out_path}")


def build_gini_vs_capacity_figure():
    d = _load("experiments/phase_o/results/peer_capacity_ablation_summary.json")
    if d is None:
        print("Skipping Gini-vs-capacity figure: experiments/phase_o/results/peer_capacity_ablation_summary.json not found")
        return
    # Schema: {"by_capacity": {"1": {"gini_ci": {"mean":..., "ci_lo":..., "ci_hi":...}, ...}, ...}}
    entries = d.get("by_capacity", {})
    caps, ginis, ci_lo, ci_hi = [], [], [], []
    items = sorted(entries.items(), key=lambda kv: float(kv[0]))
    for k, v in items:
        gini_ci = v.get("gini_ci", {})
        caps.append(k)
        ginis.append(gini_ci.get("mean"))
        ci_lo.append(gini_ci.get("ci_lo"))
        ci_hi.append(gini_ci.get("ci_hi"))

    if not ginis:
        print("Skipping Gini-vs-capacity figure: could not find gini entries in peer_capacity_ablation_summary.json "
              "(schema differs from expected; see raw JSON for exact keys)")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(caps))
        ax.plot(x, ginis, marker="o")
        if all(v is not None for v in ci_lo) and all(v is not None for v in ci_hi):
            err_lo = [g - lo for g, lo in zip(ginis, ci_lo)]
            err_hi = [hi - g for g, hi in zip(ginis, ci_hi)]
            ax.errorbar(x, ginis, yerr=[err_lo, err_hi], fmt="none", capsize=4, alpha=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(c) for c in caps])
        ax.set_xlabel("peer_capacity")
        ax.set_ylabel("Delegate-in Gini coefficient")
        ax.set_title("Gini vs. peer_capacity (Phase O part 2, seeds 42000-42019)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "gini_vs_peer_capacity.png")
        fig.savefig(out_path, dpi=150, metadata=_DETERMINISTIC_PNG_METADATA)
        plt.close(fig)
        print(f"Wrote {out_path}")
    else:
        out_path = os.path.join(OUTPUT_DIR, "gini_vs_peer_capacity.md")
        with open(out_path, "w") as f:
            f.write("# Gini vs. peer_capacity (Phase O part 2, seeds 42000-42019)\n\n")
            f.write("matplotlib not available; rendered as a table. X axis: peer_capacity. "
                    "Y axis: delegate-in Gini coefficient (mean over 20 seeds, with 95% "
                    "bootstrap CI where available).\n\n")
            f.write("| peer_capacity | Gini mean | 95% CI lo | 95% CI hi |\n|---|---|---|---|\n")
            for c, g, lo, hi in zip(caps, ginis, ci_lo, ci_hi):
                f.write(f"| {c} | {g} | {lo} | {hi} |\n")
        print(f"matplotlib unavailable, wrote table instead: {out_path}")


def run():
    build_consolidated_table()
    build_flip_rate_figure()
    build_gini_vs_capacity_figure()


if __name__ == "__main__":
    run()
