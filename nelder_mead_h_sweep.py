"""Sweep h/J while optimizing averaged-channel parameters with Nelder-Mead."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


PARAMETER_NAMES = ("alpha", "omega_max", "sigma", "T", "tau")
STANDARD_INITIAL = (0.75, 8.0, 2.0, 25.0, 0.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--h_min", type=float, default=0.1)
    parser.add_argument("--h_max", type=float, default=2.0)
    parser.add_argument("--h_points", type=int, default=20)
    parser.add_argument("--omega-points", type=int, default=10)
    parser.add_argument("--eps-fit", type=float, default=0.05)
    parser.add_argument("--op-set", default="XZ")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trace-distance-tol", type=float, default=0.05)
    parser.add_argument("--maxfev", type=int, default=100)
    parser.add_argument("--xatol", type=float, default=1e-3)
    parser.add_argument("--fatol", type=float, default=0.5)
    parser.add_argument("--initial", type=float, nargs=5, metavar=PARAMETER_NAMES, default=STANDARD_INITIAL)
    parser.add_argument(
        "--warm-start",
        help="Initialize each h/J optimization from the previous valid optimum.",
        action="store_true",
    )
    parser.add_argument(
        "--save-classical-populations",
        help="Store final-optimum transition and population matrices.",
        action="store_true",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/nelder_mead"))
    parser.add_argument("--save-as-nr", type=int, default=-1)
    return parser.parse_args()


def next_running_number(folder, N):
    pattern = re.compile(rf"nelder_mead_N{N}_h_sweep_(\d+)\.npz")
    numbers = [
        int(match.group(1))
        for path in folder.glob(f"nelder_mead_N{N}_h_sweep_*.npz")
        if (match := pattern.fullmatch(path.name))
    ]
    return max(numbers, default=0) + 1


def empty_result(h_over_J, status):
    return {
        "h_over_J": h_over_J,
        "parameters": {name: np.nan for name in PARAMETER_NAMES},
        "num_iterations": np.nan,
        "trace_distance": np.nan,
        "objective": np.nan,
        "function_evaluations": 0,
        "optimizer_success": False,
        "has_valid_candidate": False,
        "status": status,
    }


def read_result(summary_path, h_over_J, save_classical_populations):
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError) as error:
        return empty_result(h_over_J, f"unreadable_summary: {error}")

    best = summary.get("best_valid_evaluation")
    if best is None:
        result = empty_result(h_over_J, summary.get("message", "no_valid_candidate"))
        result["function_evaluations"] = int(summary.get("function_evaluations", 0))
        result["optimizer_success"] = bool(summary.get("success", False))
    else:
        try:
            parameters = {name: float(best["parameters"][name]) for name in PARAMETER_NAMES}
            result = {
                "h_over_J": h_over_J,
                "parameters": parameters,
                "num_iterations": float(best["num_iterations"]),
                "trace_distance": float(best["trace_distance"]),
                "objective": float(best["objective"]),
                "function_evaluations": int(summary["function_evaluations"]),
                "optimizer_success": bool(summary["success"]),
                "has_valid_candidate": True,
                "status": summary["message"],
            }
        except (KeyError, TypeError, ValueError) as error:
            result = empty_result(h_over_J, f"invalid_summary: {error}")
    if save_classical_populations:
        with np.load(summary_path.with_name("classical_matrices_0.npz"), allow_pickle=False) as archive:
            result["transition_generator"] = archive["transition_generator"]
            result["classical_populations"] = archive["classical_populations"]
    return result


def save_sweep(results, snapshot_path, args):
    num_iterations = np.array([result["num_iterations"] for result in results])
    alpha_opt = np.array([result["parameters"]["alpha"] for result in results])
    T_opt = np.array([result["parameters"]["T"] for result in results])
    transition_generators = [result["transition_generator"] for result in results] if args.save_classical_populations else []
    classical_populations = [result["classical_populations"] for result in results] if args.save_classical_populations else []
    np.savez_compressed(
        snapshot_path,
        h_over_J=np.array([result["h_over_J"] for result in results]),
        beta=np.full(len(results), args.beta),
        alpha_opt=alpha_opt,
        omega_max_opt=np.array([result["parameters"]["omega_max"] for result in results]),
        sigma_opt=np.array([result["parameters"]["sigma"] for result in results]),
        T_opt=T_opt,
        tau_opt=np.array([result["parameters"]["tau"] for result in results]),
        num_iterations=num_iterations,
        effective_iteration_time=num_iterations * 2 * T_opt * alpha_opt**2,
        trace_distance=np.array([result["trace_distance"] for result in results]),
        objective=np.array([result["objective"] for result in results]),
        function_evaluations=np.array([result["function_evaluations"] for result in results]),
        optimizer_success=np.array([result["optimizer_success"] for result in results]),
        has_valid_candidate=np.array([result["has_valid_candidate"] for result in results]),
        status=np.array([result["status"] for result in results]),
        transition_generator=np.stack(transition_generators) if args.save_classical_populations else np.array([]),
        classical_populations=np.stack(classical_populations) if args.save_classical_populations else np.array([]),
        N=args.N,
        omega_points=args.omega_points,
        eps_fit=args.eps_fit,
        op_set=args.op_set,
        normalize_Jh=True,
        dense_spectrum=True,
        workers=args.workers,
        trace_distance_tol=args.trace_distance_tol,
        maxfev=args.maxfev,
        xatol=args.xatol,
        fatol=args.fatol,
        warm_start=args.warm_start,
        save_classical_populations=args.save_classical_populations,
        standard_initial=np.asarray(args.initial, dtype=float),
    )


def main():
    args = parse_args()
    if args.N < 1 or args.beta <= 0 or args.workers < 1 or args.h_points < 1:
        raise ValueError("N, beta, workers, and h_points must be positive.")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    number = next_running_number(args.data_dir, args.N) if args.save_as_nr == -1 else args.save_as_nr
    snapshot_path = args.data_dir / f"nelder_mead_N{args.N}_h_sweep_{number}.npz"
    run_dir = args.data_dir / f"nelder_mead_N{args.N}_h_sweep_{number}_runs"
    run_dir.mkdir()

    h_values = np.linspace(args.h_min, args.h_max, args.h_points)
    initial = np.asarray(args.initial, dtype=float)
    results = []
    optimizer_script = Path(__file__).with_name("nelder_mead_average.py")
    print("Sweep over h/J:", h_values)

    for index, h_over_J in enumerate(h_values, start=1):
        point_dir = run_dir / f"point_{index:03d}"
        command = [
            sys.executable, str(optimizer_script), "--h-over-J", str(h_over_J), "--beta", str(args.beta),
            "--N", str(args.N), "--omega-points", str(args.omega_points), "--eps-fit", str(args.eps_fit),
            "--op-set", args.op_set, "--workers", str(args.workers),
            "--trace-distance-tol", str(args.trace_distance_tol), "--maxfev", str(args.maxfev),
            "--xatol", str(args.xatol), "--fatol", str(args.fatol), "--output-dir", str(point_dir),
            "--initial", *(str(value) for value in initial),
        ]
        if args.save_classical_populations:
            command.append("--save-classical-populations")
        print(f"\nh/J = {h_over_J:.6g} ({index}/{len(h_values)})", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            result = empty_result(h_over_J, f"subprocess_failed: {completed.returncode}")
        else:
            result = read_result(point_dir / "summary_0.json", h_over_J, args.save_classical_populations)
        if args.save_classical_populations and "transition_generator" not in result:
            result["transition_generator"] = np.full((2 ** args.N, 2 ** args.N), np.nan)
            result["classical_populations"] = np.full((2 ** args.N, 2 ** args.N), np.nan)
        results.append(result)

        if args.warm_start and result["has_valid_candidate"]:
            initial = np.array([result["parameters"][name] for name in PARAMETER_NAMES])

    save_sweep(results, snapshot_path, args)
    print(f"Saved h/J optimization sweep to {snapshot_path}")


if __name__ == "__main__":
    main()
