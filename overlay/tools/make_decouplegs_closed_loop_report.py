#!/usr/bin/env python3
"""Materialize the final DecoupleGS 4x50 closed-loop metric report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.closed_loop_report import (
    DIFFICULTIES,
    PLANNERS,
    build_final_metrics,
)

REQUIRED_CONTRACTS = {
    "H-E2E-03",
    "H-PHYSICS-01",
    "H-PHYSICS-02",
    "H-PHYSICS-04",
    "H-BEHAVIOR-03",
    "H-RC-01",
    "H-PLAN-01",
    "H-NAV-01",
    "H-CONTROL-01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT
        / "results/decouplegs/closed-loop-8scene/closed-loop-suite/summary.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "configs/benchmark/nuscenes/stress_8scene_strict/manifest.json",
    )
    parser.add_argument(
        "--paper-targets",
        type=Path,
        default=ROOT / "configs/benchmark/decouplegs_paper_targets.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=260801761)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu": properties.name,
                "gpu_memory_bytes": int(properties.total_memory),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return result


def audit_contracts(summary: dict[str, Any]) -> dict[str, Any]:
    common: set[str] | None = None
    checked = 0
    mtimes: list[float] = []
    missing_reports: list[str] = []
    for planner in PLANNERS:
        for difficulty in DIFFICULTIES:
            for episode in summary["episodes"][planner][difficulty]:
                source = Path(episode["source"])
                data_path = source / "data.pkl"
                report_path = source / "paper-metrics.json"
                if not data_path.is_file() or not report_path.is_file():
                    missing_reports.append(str(source))
                    continue
                payload = read_json(report_path)
                contracts = set(payload["protocol"]["implementation_contracts"])
                absent = REQUIRED_CONTRACTS - contracts
                if absent:
                    raise ValueError(f"{source} lacks contracts {sorted(absent)}")
                common = contracts if common is None else common & contracts
                checked += 1
                mtimes.append(data_path.stat().st_mtime)
    if missing_reports:
        raise FileNotFoundError(f"missing data/report artifacts: {missing_reports[:3]}")
    if checked != 400:
        raise ValueError(f"contract audit checked {checked} artifacts, expected 400")
    local_tz = dt.datetime.now().astimezone().tzinfo
    return {
        "artifacts_checked": checked,
        "required_contracts": sorted(REQUIRED_CONTRACTS),
        "common_contracts": sorted(common or set()),
        "data_created_first": dt.datetime.fromtimestamp(min(mtimes), tz=local_tz).isoformat(),
        "data_created_last": dt.datetime.fromtimestamp(max(mtimes), tz=local_tz).isoformat(),
    }


def format_stat(metric: dict[str, Any], digits: int = 3) -> str:
    return f"{metric['mean']:.{digits}f} ± {metric['std']:.{digits}f}"


def write_main_csv(payload: dict[str, Any], path: Path) -> None:
    fields = [
        "planner",
        "difficulty",
        "episodes",
        "driving_score",
        "driving_score_status",
        "success_rate_proxy_mean",
        "success_rate_proxy_std",
        "success_rate_proxy_ci95_low",
        "success_rate_proxy_ci95_high",
        "strict_success_rate_mean",
        "route_completion_mean",
        "route_completion_std",
        "route_completion_ci95_low",
        "route_completion_ci95_high",
        "min_ttc_literal_mean_s",
        "min_ttc_literal_std_s",
        "min_ttc_literal_ci95_low_s",
        "min_ttc_literal_ci95_high_s",
        "six_camera_observations_per_second",
        "sensor_images_per_second",
        "planner_and_ipc_updates_per_second",
        "paper_driving_score_mean",
        "paper_success_rate_mean",
        "paper_route_completion_mean",
        "paper_min_ttc_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for planner in PLANNERS:
            for difficulty in DIFFICULTIES:
                cell = payload["cells"][planner][difficulty]
                sr = cell["success_rate_proxy_safety_only"]
                strict = cell["success_rate_strict_h_metric_01"]
                rc = cell["route_completion"]
                ttc = cell["min_ttc_paper_literal_center_seconds"]
                runtime = cell["runtime"]
                paper = cell["paper_reference"]
                writer.writerow(
                    {
                        "planner": planner,
                        "difficulty": difficulty,
                        "episodes": cell["episodes"],
                        "driving_score": "",
                        "driving_score_status": cell["driving_score_status"],
                        "success_rate_proxy_mean": sr["mean"],
                        "success_rate_proxy_std": sr["std"],
                        "success_rate_proxy_ci95_low": sr["ci95_low"],
                        "success_rate_proxy_ci95_high": sr["ci95_high"],
                        "strict_success_rate_mean": strict["mean"],
                        "route_completion_mean": rc["mean"],
                        "route_completion_std": rc["std"],
                        "route_completion_ci95_low": rc["ci95_low"],
                        "route_completion_ci95_high": rc["ci95_high"],
                        "min_ttc_literal_mean_s": ttc["mean"],
                        "min_ttc_literal_std_s": ttc["std"],
                        "min_ttc_literal_ci95_low_s": ttc["ci95_low"],
                        "min_ttc_literal_ci95_high_s": ttc["ci95_high"],
                        "six_camera_observations_per_second": runtime[
                            "six_camera_observations_per_second"
                        ]["mean"],
                        "sensor_images_per_second": runtime["sensor_images_per_second"][
                            "mean"
                        ],
                        "planner_and_ipc_updates_per_second": runtime[
                            "planner_and_ipc_updates_per_second"
                        ]["mean"],
                        "paper_driving_score_mean": paper["driving_score"]["mean"],
                        "paper_success_rate_mean": paper["success_rate"]["mean"],
                        "paper_route_completion_mean": paper["route_completion"]["mean"],
                        "paper_min_ttc_mean": paper["min_ttc"]["mean"],
                    }
                )


def write_paired_csv(payload: dict[str, Any], path: Path) -> None:
    fields = [
        "difficulty",
        "metric",
        "pairs",
        "uniad_minus_vad_mean",
        "std",
        "ci95_low",
        "ci95_high",
        "uniad_win_rate",
        "tie_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for difficulty in DIFFICULTIES:
            block = payload["paired_uniad_minus_vad"][difficulty]
            for metric, value in block["metrics"].items():
                writer.writerow(
                    {
                        "difficulty": difficulty,
                        "metric": metric,
                        "pairs": block["pairs"],
                        "uniad_minus_vad_mean": value["mean"],
                        "std": value["std"],
                        "ci95_low": value["ci95_low"],
                        "ci95_high": value["ci95_high"],
                        "uniad_win_rate": value["uniad_win_rate"],
                        "tie_rate": value["tie_rate"],
                    }
                )


def make_plot(payload: dict[str, Any], path: Path) -> None:
    x = list(range(len(DIFFICULTIES)))
    labels = [name.title() for name in DIFFICULTIES]
    colors = {"uniad": "#1764ab", "vad": "#d1495b"}
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=False)
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.075,
        top=0.835,
        hspace=0.34,
        wspace=0.18,
    )

    panels = [
        (axes[0, 0], "route_completion", "Route Completion", "RC"),
        (
            axes[0, 1],
            "success_rate_proxy_safety_only",
            "Collision/off-route-free proxy (SR*)",
            "rate",
        ),
        (
            axes[1, 0],
            "min_ttc_paper_literal_center_seconds",
            "Literal minTTC",
            "seconds",
        ),
    ]
    paper_key = {
        "route_completion": "route_completion",
        "success_rate_proxy_safety_only": "success_rate",
        "min_ttc_paper_literal_center_seconds": "min_ttc",
    }
    for axis, metric, title, ylabel in panels:
        for planner in PLANNERS:
            local = [payload["cells"][planner][d][metric]["mean"] for d in DIFFICULTIES]
            low = [payload["cells"][planner][d][metric]["ci95_low"] for d in DIFFICULTIES]
            high = [payload["cells"][planner][d][metric]["ci95_high"] for d in DIFFICULTIES]
            paper = [
                payload["cells"][planner][d]["paper_reference"][paper_key[metric]]["mean"]
                for d in DIFFICULTIES
            ]
            axis.plot(x, local, marker="o", linewidth=2.3, color=colors[planner], label=f"local {planner.upper()}")
            axis.fill_between(x, low, high, color=colors[planner], alpha=0.14)
            axis.plot(x, paper, marker="x", linestyle="--", color=colors[planner], alpha=0.65, label=f"paper {planner.upper()}")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.25)
    axes[0, 1].set_ylim(-0.04, 1.04)

    axis = axes[1, 1]
    for planner in PLANNERS:
        batch = [
            payload["cells"][planner][d]["runtime"]["six_camera_observations_per_second"]["mean"]
            for d in DIFFICULTIES
        ]
        images = [
            payload["cells"][planner][d]["runtime"]["sensor_images_per_second"]["mean"]
            for d in DIFFICULTIES
        ]
        axis.plot(x, batch, marker="o", linewidth=2.3, color=colors[planner], label=f"{planner.upper()} six-camera batches/s")
        axis.plot(x, images, marker="s", linestyle=":", linewidth=2.0, color=colors[planner], label=f"{planner.upper()} images/s")
    axis.axhline(45.0, color="#555555", linestyle="--", linewidth=1.5, label="paper bare FPS = 45")
    axis.set_title("Renderer throughput (paper FPS unit is undisclosed)")
    axis.set_ylabel("per second")
    axis.set_xticks(x, labels)
    axis.grid(alpha=0.25)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=4,
        frameon=False,
    )
    handles, legend_labels = axis.get_legend_handles_labels()
    axis.legend(handles, legend_labels, fontsize=8, frameon=False)
    figure.suptitle(
        "DecoupleGS public replacement: complete 4×50×2 closed-loop suite",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DecoupleGS closed-loop final metrics — public replacement protocol",
        "",
        "Status: **complete (400/400 valid episodes)**. This is the full strict UniAD/VAD matrix, not the earlier one-episode smoke table.",
        "",
        "![Final metric curves](final-metrics.png)",
        "",
        "## Paper-facing 4×50 table",
        "",
        "`SR*` is the collision/off-route/planner-failure-free proxy. The paper names SR but does not publish its terminal rule. `DS` is intentionally null because the paper does not publish the penalty factors.",
        "",
        "| Difficulty | Planner | N | DS | SR* | Strict SR | RC | Literal minTTC | 6-camera obs/s | Images/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for difficulty in DIFFICULTIES:
        for planner in PLANNERS:
            cell = payload["cells"][planner][difficulty]
            runtime = cell["runtime"]
            lines.append(
                f"| {difficulty.title()} | {planner.upper()} | {cell['episodes']} | —† | "
                f"{format_stat(cell['success_rate_proxy_safety_only'])} | "
                f"{format_stat(cell['success_rate_strict_h_metric_01'])} | "
                f"{format_stat(cell['route_completion'])} | "
                f"{format_stat(cell['min_ttc_paper_literal_center_seconds'])} s | "
                f"{runtime['six_camera_observations_per_second']['mean']:.2f} | "
                f"{runtime['sensor_images_per_second']['mean']:.2f} |"
            )
    lines.extend(
        [
            "",
            "† `DS = RC × ∏ p_i^{C_i}` is implemented, but the author values of `p_i` are absent; no factor was guessed or fitted.",
            "",
            "## Local minus paper Table 3",
            "",
            "These deltas are descriptive only: the author clip IDs, seeds, SR rule, and minTTC/FPS aggregation details are not public.",
            "",
            "| Difficulty | Planner | ΔSR* | ΔRC | ΔminTTC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for difficulty in DIFFICULTIES:
        for planner in PLANNERS:
            delta = payload["cells"][planner][difficulty]["delta_vs_paper"]
            lines.append(
                f"| {difficulty.title()} | {planner.upper()} | {delta['success_rate_proxy']:+.3f} | "
                f"{delta['route_completion']:+.3f} | {delta['min_ttc']:+.3f} s |"
            )
    lines.extend(
        [
            "",
            "## Paired UniAD − VAD differences",
            "",
            "Each difference uses the same 50 scenario seeds, densities, and scenes. Intervals are deterministic 20,000-resample episode bootstrap CIs.",
            "",
            "| Difficulty | ΔRC [95% CI] | ΔSR* [95% CI] | ΔminTTC [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for difficulty in DIFFICULTIES:
        metrics = payload["paired_uniad_minus_vad"][difficulty]["metrics"]
        rc = metrics["route_completion"]
        sr = metrics["success_rate_proxy_safety_only"]
        ttc = metrics["min_ttc_paper_literal_center_seconds"]
        lines.append(
            f"| {difficulty.title()} | {rc['mean']:+.3f} [{rc['ci95_low']:+.3f}, {rc['ci95_high']:+.3f}] | "
            f"{sr['mean']:+.3f} [{sr['ci95_low']:+.3f}, {sr['ci95_high']:+.3f}] | "
            f"{ttc['mean']:+.3f} [{ttc['ci95_low']:+.3f}, {ttc['ci95_high']:+.3f}] s |"
        )
    lines.extend(
        [
            "",
            "## Termination audit",
            "",
            "Counts below are terminal flags; a composite reason can contribute to two flag columns. `Composite` reports how many such episodes exist.",
            "",
            "| Difficulty | Planner | Time limit | Route complete | Agent collision | Background collision | Off route | Planner failure | Composite |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    termination_names = (
        "time_limit",
        "route_complete",
        "agent_collision",
        "background_collision",
        "off_route",
        "planner_failure",
    )
    for difficulty in DIFFICULTIES:
        for planner in PLANNERS:
            cell = payload["cells"][planner][difficulty]
            counts = cell["termination_token_counts"]
            values = [int(counts.get(name, 0)) for name in termination_names]
            lines.append(
                f"| {difficulty.title()} | {planner.upper()} | "
                + " | ".join(str(value) for value in values)
                + f" | {cell['composite_terminations']} |"
            )
    coverage = payload["protocol_coverage"]
    contracts = payload["artifact_audit"]
    hw = payload["hardware"]
    lines.extend(
        [
            "",
            "## Evidence and protocol boundary",
            "",
            f"- Coverage: {coverage['planner_episode_count']} planner episodes; 50 per planner/difficulty; {len(coverage['scene_counts'])} scenes; {coverage['asset_pool_size']} canonical assets; {coverage['unique_scenario_seeds']} unique paired scenario seeds.",
            f"- Timing: {coverage['episode_seconds']} s maximum, {coverage['control_frequency_hz']} Hz policy loop, {coverage['maximum_steps']} maximum steps.",
            f"- Artifact audit: {contracts['artifacts_checked']}/400 `data.pkl` + `paper-metrics.json` pairs contain all {len(contracts['required_contracts'])} required current implementation contracts.",
            f"- Local hardware: {hw.get('gpu', 'CPU')} ({hw.get('gpu_memory_bytes', 0) / 2**30:.1f} GiB), PyTorch {hw['torch']} / CUDA {hw['torch_cuda']}; the paper uses an RTX 4090, so FPS is not hardware-equivalent.",
            "- Exact disclosed equations: metric-polyline `RC = D_traveled / D_planned`; literal center-distance minTTC with relative-speed norm.",
            "- Non-identifiable: DS penalty factors, SR rule, author scenes/seeds, and FPS timing/count unit. The paper's `SR=0.850` over exactly 50 binary episodes is itself not an integer success count, confirming an unpublished aggregation or rounding layer.",
            "- Strict SR is zero in every cell because no strict-policy run both reached `RC ≥ 0.99` and avoided severe failure within 20 s. This is reported rather than relabeled as paper SR.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "conda run -n decouplegs python tools/run_decouplegs_closed_loop_suite.py \\",
            "  --manifest configs/benchmark/nuscenes/stress_8scene_strict/manifest.json \\",
            "  --base-config configs/benchmark/local_nuscenes_8scene.yaml --aggregate-only",
            "conda run -n decouplegs python tools/make_decouplegs_closed_loop_report.py",
            "```",
            "",
            "Machine-readable files: `final-metrics.json`, `final-table3.csv`, and `paired-differences.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = read_json(args.summary)
    manifest = read_json(args.manifest)
    paper_targets = yaml.safe_load(args.paper_targets.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_final_metrics(
        summary,
        paper_targets,
        manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload.update(
        {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_files": {
                "summary": str(args.summary.resolve()),
                "summary_sha256": sha256(args.summary),
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256(args.manifest),
                "paper_targets": str(args.paper_targets.resolve()),
                "paper_targets_sha256": sha256(args.paper_targets),
            },
            "artifact_audit": audit_contracts(summary),
            "hardware": hardware(),
        }
    )
    (output_dir / "final-metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_main_csv(payload, output_dir / "final-table3.csv")
    write_paired_csv(payload, output_dir / "paired-differences.csv")
    make_plot(payload, output_dir / "final-metrics.png")
    (output_dir / "REPORT.md").write_text(markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "episodes": payload["source_suite"]["completed"],
                "report": str((output_dir / "REPORT.md").resolve()),
                "metrics": str((output_dir / "final-metrics.json").resolve()),
                "figure": str((output_dir / "final-metrics.png").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
