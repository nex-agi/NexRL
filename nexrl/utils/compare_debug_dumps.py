# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comparison script for debug dumps between self_hosted and remote_api modes.

This script loads and compares training data dumps from both modes to help
debug training accuracy issues.

Usage:
    python compare_debug_dumps.py \
        --self_hosted_dir /path/to/self_hosted_dump \
        --remote_api_dir /path/to/remote_api_dump \
        --weaver_dir /path/to/weaver_trainer_dump \
        --step 1

Or as a module:
    from nexrl.utils.compare_debug_dumps import compare_dumps
    report = compare_dumps(self_hosted_dir, remote_api_dir, weaver_dir, step=1)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_pt_file(path: Path) -> dict[str, Any]:
    """Load a .pt file."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return {}
    return torch.load(path)


def compare_tensors(
    name: str,
    t1: torch.Tensor | None,
    t2: torch.Tensor | None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare two tensors and return comparison metrics."""
    result: dict[str, Any] = {"name": name}

    if t1 is None and t2 is None:
        result["status"] = "both_none"
        return result

    if t1 is None:
        result["status"] = "first_none"
        result["t2_shape"] = list(t2.shape) if hasattr(t2, "shape") else "N/A"  # type: ignore[union-attr]
        return result

    if t2 is None:
        result["status"] = "second_none"
        result["t1_shape"] = list(t1.shape) if hasattr(t1, "shape") else "N/A"
        return result

    # Convert to tensors if needed
    if not isinstance(t1, torch.Tensor):
        t1 = torch.tensor(t1)
    if not isinstance(t2, torch.Tensor):
        t2 = torch.tensor(t2)

    result["t1_shape"] = list(t1.shape)
    result["t2_shape"] = list(t2.shape)

    if t1.shape != t2.shape:
        result["status"] = "shape_mismatch"
        return result

    # Compute comparison metrics
    t1_flat = t1.flatten().float()
    t2_flat = t2.flatten().float()

    diff = (t1_flat - t2_flat).abs()

    result["status"] = "compared"
    result["max_diff"] = float(diff.max().item())
    result["mean_diff"] = float(diff.mean().item())
    result["t1_mean"] = float(t1_flat.mean().item())
    result["t2_mean"] = float(t2_flat.mean().item())
    result["t1_std"] = float(t1_flat.std().item())
    result["t2_std"] = float(t2_flat.std().item())
    result["t1_min"] = float(t1_flat.min().item())
    result["t1_max"] = float(t1_flat.max().item())
    result["t2_min"] = float(t2_flat.min().item())
    result["t2_max"] = float(t2_flat.max().item())

    # Check if close
    is_close = torch.allclose(t1_flat, t2_flat, atol=atol, rtol=rtol)
    result["is_close"] = bool(is_close)

    # Compute correlation
    if t1_flat.std() > 1e-8 and t2_flat.std() > 1e-8:
        corr = torch.corrcoef(torch.stack([t1_flat, t2_flat]))[0, 1]
        result["correlation"] = float(corr.item())

    return result


def compare_scalars(
    name: str,
    s1: float | None,
    s2: float | None,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Compare two scalar values."""
    result: dict[str, Any] = {"name": name}

    if s1 is None and s2 is None:
        result["status"] = "both_none"
        return result

    if s1 is None:
        result["status"] = "first_none"
        result["s2"] = s2
        return result

    if s2 is None:
        result["status"] = "second_none"
        result["s1"] = s1
        return result

    result["status"] = "compared"
    result["s1"] = float(s1)
    result["s2"] = float(s2)
    result["diff"] = abs(float(s1) - float(s2))
    result["is_close"] = abs(float(s1) - float(s2)) < atol

    return result


def compare_trajectories(
    sh_traj_file: Path,
    ra_traj_file: Path,
) -> dict[str, Any]:
    """Compare trajectory dumps."""
    report: dict[str, Any] = {"section": "trajectories"}

    sh_data = load_pt_file(sh_traj_file)
    ra_data = load_pt_file(ra_traj_file)

    if not sh_data:
        report["status"] = "self_hosted_not_found"
        return report

    if not ra_data:
        report["status"] = "remote_api_not_found"
        return report

    sh_trajs = sh_data.get("trajectories", [])
    ra_trajs = ra_data.get("trajectories", [])

    report["sh_num_trajectories"] = len(sh_trajs)
    report["ra_num_trajectories"] = len(ra_trajs)

    if len(sh_trajs) != len(ra_trajs):
        report["status"] = "count_mismatch"
        return report

    # Compare first few trajectories
    comparisons: list[dict[str, Any]] = []
    for i, (sh_t, ra_t) in enumerate(zip(sh_trajs[:5], ra_trajs[:5])):
        traj_cmp: dict[str, Any] = {"index": i}

        # Compare tokens
        sh_tokens = sh_t.get("tokens", [])
        ra_tokens = ra_t.get("tokens", [])
        traj_cmp["tokens_match"] = sh_tokens == ra_tokens
        traj_cmp["sh_tokens_len"] = len(sh_tokens)
        traj_cmp["ra_tokens_len"] = len(ra_tokens)

        # Compare loss_mask
        sh_mask = sh_t.get("loss_mask", [])
        ra_mask = ra_t.get("loss_mask", [])
        traj_cmp["loss_mask_match"] = sh_mask == ra_mask

        # Compare reward
        traj_cmp["sh_reward"] = sh_t.get("reward")
        traj_cmp["ra_reward"] = ra_t.get("reward")

        comparisons.append(traj_cmp)

    report["trajectory_comparisons"] = comparisons
    report["status"] = "compared"
    return report


def compare_forward_data(
    sh_forward_file: Path,
    weaver_forward_file: Path,
) -> dict[str, Any]:
    """Compare forward pass data (logprobs, entropy).

    Both files use the same format from DataDumper:
    - log_probs (self_hosted) / log_probs (weaver)
    - entropy
    - response_mask
    """
    report: dict[str, Any] = {"section": "forward_data"}

    sh_data = load_pt_file(sh_forward_file)
    weaver_data = load_pt_file(weaver_forward_file)

    if not sh_data:
        report["status"] = "self_hosted_not_found"
        return report

    if not weaver_data:
        report["status"] = "weaver_not_found"
        return report

    report["status"] = "compared"
    comparisons: list[dict[str, Any]] = []

    # Compare logprobs (both use "log_probs" key now)
    sh_logprobs = sh_data.get("log_probs")
    weaver_logprobs = weaver_data.get("log_probs")
    comparisons.append(compare_tensors("log_probs", sh_logprobs, weaver_logprobs))

    # Compare entropy
    sh_entropy = sh_data.get("entropy")
    weaver_entropy = weaver_data.get("entropy")
    comparisons.append(compare_tensors("entropy", sh_entropy, weaver_entropy))

    # Compare response_mask
    sh_mask = sh_data.get("response_mask")
    weaver_mask = weaver_data.get("response_mask")
    comparisons.append(compare_tensors("response_mask", sh_mask, weaver_mask))

    report["comparisons"] = comparisons
    return report


def compare_loss_data(
    sh_loss_file: Path,
    weaver_loss_file: Path,
) -> dict[str, Any]:
    """Compare loss computation data.

    Both DataDumpers use the same format:
    - loss: Total loss value
    - pg_loss: Policy gradient loss value
    - entropy_loss: Entropy loss value
    """
    report: dict[str, Any] = {"section": "loss_data"}

    sh_data = load_pt_file(sh_loss_file)
    weaver_data = load_pt_file(weaver_loss_file)

    if not sh_data:
        report["status"] = "self_hosted_not_found"
        return report

    if not weaver_data:
        report["status"] = "weaver_not_found"
        return report

    report["status"] = "compared"
    comparisons: list[dict[str, Any]] = []

    # Compare loss values (both use "loss" key)
    sh_loss = sh_data.get("loss")
    weaver_loss = weaver_data.get("loss")
    comparisons.append(compare_scalars("loss", sh_loss, weaver_loss))

    # Compare pg_loss (policy gradient loss)
    sh_pg_loss = sh_data.get("pg_loss")
    weaver_pg_loss = weaver_data.get("pg_loss")
    comparisons.append(compare_scalars("pg_loss", sh_pg_loss, weaver_pg_loss))

    # Compare entropy loss
    sh_entropy_loss = sh_data.get("entropy_loss")
    weaver_entropy_loss = weaver_data.get("entropy_loss")
    comparisons.append(compare_scalars("entropy_loss", sh_entropy_loss, weaver_entropy_loss))

    report["comparisons"] = comparisons
    return report


def compare_is_loss_debug(
    sh_loss_file: Path,
    weaver_is_debug_file: Path,
    sh_old_log_probs_file: Path | None = None,
) -> dict[str, Any]:
    """Compare detailed importance sampling loss computation.

    Both DataDumpers now use consistent keys:
    - log_probs: new logprobs from current model forward
    - old_log_probs: old logprobs from rollout (weaver) / recomputed (self_hosted)
    - advantages, prob_ratio, elementwise_loss, effective_valid, loss
    """
    report: dict[str, Any] = {"section": "is_loss_debug"}

    sh_data = load_pt_file(sh_loss_file)
    weaver_data = load_pt_file(weaver_is_debug_file)

    # Load self_hosted old_log_probs from separate file if provided
    sh_old_log_probs_data = None
    if sh_old_log_probs_file:
        sh_old_log_probs_data = load_pt_file(sh_old_log_probs_file)

    if not sh_data:
        report["status"] = "self_hosted_not_found"
        return report

    if not weaver_data:
        report["status"] = "weaver_debug_not_found"
        return report

    report["status"] = "compared"
    comparisons: list[dict[str, Any]] = []

    # Compare logprobs (both use "log_probs" key)
    comparisons.append(
        compare_tensors(
            "log_probs",
            sh_data.get("log_probs"),
            weaver_data.get("log_probs"),
        )
    )

    # Compare old_log_probs
    # self_hosted stores in separate file, weaver stores in same file
    sh_old_log_probs = None
    if sh_old_log_probs_data:
        sh_old_log_probs = sh_old_log_probs_data.get("old_log_probs")
    comparisons.append(
        compare_tensors(
            "old_log_probs",
            sh_old_log_probs,
            weaver_data.get("old_log_probs"),
        )
    )

    # Compare advantages
    comparisons.append(
        compare_tensors(
            "advantages",
            sh_data.get("advantages"),
            weaver_data.get("advantages"),
        )
    )

    # Compare prob_ratio
    comparisons.append(
        compare_tensors(
            "prob_ratio",
            sh_data.get("prob_ratio"),
            weaver_data.get("prob_ratio"),
        )
    )

    # Compare elementwise_loss
    comparisons.append(
        compare_tensors(
            "elementwise_loss",
            sh_data.get("elementwise_loss"),
            weaver_data.get("elementwise_loss"),
        )
    )

    # Compare effective_valid / loss_mask
    comparisons.append(
        compare_tensors(
            "effective_valid",
            sh_data.get("effective_valid"),
            weaver_data.get("effective_valid"),
        )
    )

    # Compare final loss
    sh_loss = sh_data.get("loss")
    wv_loss = weaver_data.get("loss")
    if isinstance(wv_loss, torch.Tensor):
        wv_loss = wv_loss.item()
    comparisons.append(compare_scalars("loss", sh_loss, wv_loss))

    report["comparisons"] = comparisons
    return report


def compare_datums(
    ra_datums_file: Path,
) -> dict[str, Any]:
    """Analyze datum structure from remote_api mode."""
    report: dict[str, Any] = {"section": "datums_analysis"}

    data = load_pt_file(ra_datums_file)

    if not data:
        report["status"] = "not_found"
        return report

    datums = data.get("datums", [])
    report["num_datums"] = len(datums)

    # Analyze first datum structure
    if datums:
        first_datum = datums[0]
        report["first_datum_keys"] = list(first_datum.keys())

        if "loss_fn_inputs" in first_datum:
            report["loss_fn_inputs_keys"] = list(first_datum["loss_fn_inputs"].keys())

        if "input_tokens" in first_datum:
            report["input_tokens_len"] = len(first_datum["input_tokens"])

    report["status"] = "analyzed"
    return report


def compare_dumps(
    self_hosted_dir: str | Path,
    remote_api_dir: str | Path | None = None,
    weaver_dir: str | Path | None = None,
    step: int = 1,
    num_ranks: int = 8,
    num_micros: int = 2,
) -> dict[str, Any]:
    """
    Compare debug dumps from self_hosted and remote_api modes.

    Args:
        self_hosted_dir: Directory containing self_hosted debug dumps
        remote_api_dir: Directory containing remote_api (NexRL side) debug dumps
        weaver_dir: Directory containing weaver-trainer debug dumps
        step: Training step to compare
        num_ranks: Number of ranks to compare (default: 8)
        num_micros: Number of microbatches to compare (default: 2)

    Returns:
        Comparison report dictionary
    """
    sh_dir = Path(self_hosted_dir)
    ra_dir = Path(remote_api_dir) if remote_api_dir else None
    wv_dir = Path(weaver_dir) if weaver_dir else None

    sections_list: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "step": step,
        "self_hosted_dir": str(sh_dir),
        "remote_api_dir": str(ra_dir) if ra_dir else None,
        "weaver_dir": str(wv_dir) if wv_dir else None,
        "num_ranks": num_ranks,
        "num_micros": num_micros,
        "sections": sections_list,
    }

    step_str = f"step_{step:06d}"

    # 1. Compare trajectories
    sh_traj = sh_dir / "trajectory" / f"{step_str}.pt"
    ra_traj = (
        (ra_dir / "prepared_trajectories" / f"{step_str}.pt") if ra_dir else Path("nonexistent")
    )
    sections_list.append(compare_trajectories(sh_traj, ra_traj))

    # 2. Compare forward data for all ranks and microbatches
    forward_comparisons: list[dict[str, Any]] = []
    for rank in range(num_ranks):
        for micro in range(num_micros):
            sh_forward = sh_dir / "forward_data" / f"{step_str}_rank{rank}_micro{micro}.pt"
            wv_forward = (
                (wv_dir / "forward_data" / f"{step_str}_rank{rank}_micro{micro}.pt")
                if wv_dir
                else Path("nonexistent")
            )
            cmp = compare_forward_data(sh_forward, wv_forward)
            cmp["rank"] = rank
            cmp["micro"] = micro
            forward_comparisons.append(cmp)

    sections_list.append(
        {
            "section": "forward_data_all",
            "comparisons": forward_comparisons,
        }
    )

    # 3. Compare loss data for all ranks and microbatches
    loss_comparisons: list[dict[str, Any]] = []
    for rank in range(num_ranks):
        for micro in range(num_micros):
            sh_loss = sh_dir / "loss" / f"{step_str}_rank{rank}_micro{micro}.pt"
            wv_loss = (
                (wv_dir / "loss" / f"{step_str}_rank{rank}_micro{micro}.pt")
                if wv_dir
                else Path("nonexistent")
            )
            cmp = compare_loss_data(sh_loss, wv_loss)
            cmp["rank"] = rank
            cmp["micro"] = micro
            loss_comparisons.append(cmp)

    sections_list.append(
        {
            "section": "loss_data_all",
            "comparisons": loss_comparisons,
        }
    )

    # 4. Compare IS loss debug data for all ranks and microbatches
    is_debug_comparisons: list[dict[str, Any]] = []
    sh_old_log_probs = sh_dir / "old_log_probs" / f"{step_str}.pt"
    for rank in range(num_ranks):
        for micro in range(num_micros):
            sh_loss = sh_dir / "loss" / f"{step_str}_rank{rank}_micro{micro}.pt"
            wv_is_debug = (
                (wv_dir / "is_loss_debug" / f"{step_str}_rank{rank}_micro{micro}.pt")
                if wv_dir
                else Path("nonexistent")
            )
            cmp = compare_is_loss_debug(sh_loss, wv_is_debug, sh_old_log_probs)
            cmp["rank"] = rank
            cmp["micro"] = micro
            is_debug_comparisons.append(cmp)

    sections_list.append(
        {
            "section": "is_loss_debug_all",
            "comparisons": is_debug_comparisons,
        }
    )

    # 5. Analyze datums
    ra_datums = (ra_dir / "datums" / f"{step_str}.pt") if ra_dir else Path("nonexistent")
    sections_list.append(compare_datums(ra_datums))

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print comparison report in human-readable format."""
    print("\n" + "=" * 80)
    print(f"Debug Dump Comparison Report - Step {report['step']}")
    print("=" * 80)

    print(f"\nDirectories:")
    print(f"  Self-Hosted: {report['self_hosted_dir']}")
    print(f"  Remote API:  {report['remote_api_dir']}")
    print(f"  Weaver:      {report['weaver_dir']}")
    print(
        f"\nComparing: {report.get('num_ranks', 1)} ranks × {report.get('num_micros', 1)} microbatches"
    )

    for section in report.get("sections", []):
        section_name = section["section"].upper()
        print(f"\n--- {section_name} ---")

        # Handle sections with multiple rank/micro comparisons
        if section_name in ["FORWARD_DATA_ALL", "LOSS_DATA_ALL", "IS_LOSS_DEBUG_ALL"]:
            comparisons = section.get("comparisons", [])
            if not comparisons:
                print("Status: no data")
                continue

            # Count statuses across all ranks/micros
            status_counts: dict[str, int] = {}
            all_diffs: list[tuple[int, int, str, float, bool]] = []

            for cmp in comparisons:
                rank = cmp.get("rank", 0)
                micro = cmp.get("micro", 0)
                status = cmp.get("status", "N/A")

                # For sections with nested comparisons
                if "comparisons" in cmp:
                    for item_cmp in cmp["comparisons"]:
                        item_name = item_cmp.get("name", "unknown")
                        item_status = item_cmp.get("status", "N/A")
                        key = f"{item_name}_{item_status}"
                        status_counts[key] = status_counts.get(key, 0) + 1

                        if item_status == "compared":
                            if "diff" in item_cmp:
                                all_diffs.append(
                                    (
                                        rank,
                                        micro,
                                        item_name,
                                        item_cmp["diff"],
                                        item_cmp.get("is_close", False),
                                    )
                                )

            # Print summary
            print(f"Total comparisons: {len(comparisons)}")

            if all_diffs:
                print("\nLoss differences across ranks/micros:")
                for rank, micro, name, diff, is_close in sorted(
                    all_diffs, key=lambda x: x[3], reverse=True
                )[:10]:
                    status_symbol = "✓" if is_close else "✗"
                    print(f"  rank{rank}_micro{micro} {name}: {status_symbol} diff={diff:.6e}")

                # Summary statistics
                diffs_only = [d[3] for d in all_diffs]
                print(f"\nDiff statistics:")
                print(f"  max: {max(diffs_only):.6e}")
                print(f"  min: {min(diffs_only):.6e}")
                print(f"  mean: {sum(diffs_only)/len(diffs_only):.6e}")

        else:
            # Original handling for single comparisons
            print(f"Status: {section.get('status', 'N/A')}")

            if "comparisons" in section:
                for cmp in section["comparisons"]:
                    name = cmp.get("name", "unknown")
                    status = cmp.get("status", "N/A")

                    if status == "compared":
                        is_close_symbol = "✓" if cmp.get("is_close", False) else "✗"
                        print(f"\n  {name}: {is_close_symbol}")

                        if "max_diff" in cmp:
                            print(f"    max_diff: {cmp['max_diff']:.6e}")
                            print(f"    mean_diff: {cmp['mean_diff']:.6e}")
                            print(
                                f"    t1_mean: {cmp['t1_mean']:.6f}, t2_mean: {cmp['t2_mean']:.6f}"
                            )
                            if "correlation" in cmp:
                                print(f"    correlation: {cmp['correlation']:.6f}")
                        elif "s1" in cmp and "s2" in cmp:
                            print(f"    s1: {cmp['s1']:.6f}, s2: {cmp['s2']:.6f}")
                            print(f"    diff: {cmp['diff']:.6e}")
                    else:
                        print(f"\n  {name}: {status}")

            if "trajectory_comparisons" in section:
                print("\n  Trajectory comparisons (first 5):")
                for tc in section["trajectory_comparisons"]:
                    idx = tc["index"]
                    tokens_match = "✓" if tc.get("tokens_match") else "✗"
                    mask_match = "✓" if tc.get("loss_mask_match") else "✗"
                    print(
                        f"    [{idx}] tokens: {tokens_match}, mask: {mask_match}, "
                        f"sh_len: {tc.get('sh_tokens_len')}, ra_len: {tc.get('ra_tokens_len')}"
                    )

    print("\n" + "=" * 80)


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Compare debug dumps between self_hosted and remote_api modes"
    )
    parser.add_argument(
        "--self_hosted_dir",
        type=str,
        required=True,
        help="Directory containing self_hosted debug dumps",
    )
    parser.add_argument(
        "--remote_api_dir",
        type=str,
        default=None,
        help="Directory containing remote_api (NexRL side) debug dumps",
    )
    parser.add_argument(
        "--weaver_dir",
        type=str,
        default=None,
        help="Directory containing weaver-trainer debug dumps (EASY_DUMP_DIR)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Training step to compare (default: 1)",
    )
    parser.add_argument(
        "--num_ranks",
        type=int,
        default=8,
        help="Number of ranks to compare (default: 8)",
    )
    parser.add_argument(
        "--num_micros",
        type=int,
        default=2,
        help="Number of microbatches to compare (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON report (optional)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report = compare_dumps(
        self_hosted_dir=args.self_hosted_dir,
        remote_api_dir=args.remote_api_dir,
        weaver_dir=args.weaver_dir,
        step=args.step,
        num_ranks=args.num_ranks,
        num_micros=args.num_micros,
    )

    print_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
