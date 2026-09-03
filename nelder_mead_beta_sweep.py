"""Optimize averaged-channel parameters at each beta value with Nelder-Mead."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


PARAMETER_NAMES = ("alpha", "omega_max", "sigma", "T", "tau")
ORDINARY_BETA_INITIAL = (0.5, 2.5, 2.0, 25.0, 0.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.2)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=10.0)
    parser.add_argument("--beta-points", type=int, default=25)
    parser.add_argument(
        "--beta-values", type=float, nargs="+", default=None,
        help="Explicit beta values. Overrides --beta-min, --beta-max, and --beta-points.",
    )
    parser.add_argument("--omega-points", type=int, default=10)
    parser.add_argument("--eps-fit", type=float, default=0.05)
    parser.add_argument("--op-set", default="XZ")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trace-distance-tol", type=float, default=0.05)
    parser.add_argument("--maxfev", type=int, default=100)
    parser.add_argument("--xatol", type=float, default=1e-3)
    parser.add_argument("--fatol", type=float, default=0.5)
    parser.add_argument(
        "--initial", type=float, nargs=5, metavar=PARAMETER_NAMES,
        default=ORDINARY_BETA_INITIAL,
        help="Initial alpha, omega_max, sigma, T, and tau for the first beta point.",
    )
    parser.add_argument(
        "--warm-start", action="store_true",
        help="Initialize each beta optimization from the preceding valid optimum.",
    )
    parser.add_argument(
        "--save-classical-populations",
        help="Store final-optimum transition and population matrices.",
        action="store_true",
    )
    parser.add_argument(
        "--normalize-Jh", "--normalize_Jh", dest="normalize_Jh", action="store_true",
        help="Normalize J and h by N sqrt(J^2 + h^2). Disabled by default to match the ordinary beta sweep.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/nelder_mead"))
    parser.add_argument("--save-as-nr", type=int, default=-1)
    return parser.parse_args()


def next_running_number(folder, N):
    pattern = re.compile(rf"nelder_mead_N{N}_beta_sweep_(\d+)\.npz")
    numbers = [
        int(match.group(1))
        for path in folder.glob(f"nelder_mead_N{N}_beta_sweep_*.npz")
        if (match := pattern.fullmatch(path.name))
    ]
    return max(numbers, default=0) + 1


def empty_result(beta, status):
    return {
        "beta": beta,
        "parameters": {name: np.nan for name in PARAMETER_NAMES},
        "num_iterations": np.nan,
        "trace_distance": np.nan,
        "objective": np.nan,
        "function_evaluations": 0,
        "optimizer_success": False,
        "has_valid_candidate": False,
        "status": status,
    }


def read_result(summary_path, beta, save_classical_populations):
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError) as error:
        return empty_result(beta, f"unreadable_summary: {error}")

    best = summary.get("best_valid_evaluation")
    if best is None:
        result = empty_result(beta, summary.get("message", "no_valid_candidate"))
        result["function_evaluations"] = int(summary.get("function_evaluations", 0))
        result["optimizer_success"] = bool(summary.get("success", False))
    else:
        try:
            result = {
                "beta": beta,
                "parameters": {name: float(best["parameters"][name]) for name in PARAMETER_NAMES},
                "num_iterations": float(best["num_iterations"]),
                "trace_distance": float(best["trace_distance"]),
                "objective": float(best["objective"]),
                "function_evaluations": int(summary["function_evaluations"]),
                "optimizer_success": bool(summary["success"]),
                "has_valid_candidate": True,
                "status": summary["message"],
            }
        except (KeyError, TypeError, ValueError) as error:
            result = empty_result(beta, f"invalid_summary: {error}")
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
        beta=np.array([result["beta"] for result in results]),
        h=np.full(len(results), args.h),
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
        J=args.J,
        h_over_J=args.h / args.J,
        omega_points=args.omega_points,
        eps_fit=args.eps_fit,
        op_set=args.op_set,
        normalize_Jh=args.normalize_Jh,
        workers=args.workers,
        trace_distance_tol=args.trace_distance_tol,
        maxfev=args.maxfev,
        xatol=args.xatol,
        fatol=args.fatol,
        warm_start=args.warm_start,
        save_classical_populations=args.save_classical_populations,
        ordinary_beta_initial=np.asarray(args.initial, dtype=float),
    )


def main():
    args = parse_args()
    if min(args.N, args.workers, args.beta_points) < 1 or args.J <= 0 or args.h <= 0:
        raise ValueError("N, J, h, workers, and beta_points must be positive.")
    if args.beta_min <= 0 or args.beta_max < args.beta_min:
        raise ValueError("beta_min must be positive and beta_max must be at least beta_min.")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    number = next_running_number(args.data_dir, args.N) if args.save_as_nr == -1 else args.save_as_nr
    snapshot_path = args.data_dir / f"nelder_mead_N{args.N}_beta_sweep_{number}.npz"
    run_dir = args.data_dir / f"nelder_mead_N{args.N}_beta_sweep_{number}_runs"
    run_dir.mkdir()

    beta_values = np.asarray(args.beta_values, dtype=float) if args.beta_values is not None else np.geomspace(args.beta_min, args.beta_max, args.beta_points)
    initial = np.asarray(args.initial, dtype=float)
    h_over_J = args.h / args.J
    optimizer_script = Path(__file__).with_name("nelder_mead_average.py")
    results = []
    print("Sweep over beta:", beta_values)

    for index, beta in enumerate(beta_values, start=1):
        point_dir = run_dir / f"point_{index:03d}"
        command = [
            sys.executable, str(optimizer_script), "--h-over-J", str(h_over_J), "--J", str(args.J),
            "--beta", str(beta), "--N", str(args.N), "--omega-points", str(args.omega_points),
            "--eps-fit", str(args.eps_fit), "--op-set", args.op_set, "--workers", str(args.workers),
            "--trace-distance-tol", str(args.trace_distance_tol), "--maxfev", str(args.maxfev),
            "--xatol", str(args.xatol), "--fatol", str(args.fatol), "--output-dir", str(point_dir),
            "--initial", *(str(value) for value in initial),
        ]
        if args.save_classical_populations:
            command.append("--save-classical-populations")
        if not args.normalize_Jh:
            command.append("--no-normalize-Jh")
        print(f"\nbeta = {beta:.6g} ({index}/{len(beta_values)})", flush=True)
        completed = subprocess.run(command, check=False)
        result = (
            read_result(point_dir / "summary_0.json", beta, args.save_classical_populations)
            if completed.returncode == 0
            else empty_result(beta, f"subprocess_failed: {completed.returncode}")
        )
        if args.save_classical_populations and "transition_generator" not in result:
            result["transition_generator"] = np.full((2 ** args.N, 2 ** args.N), np.nan)
            result["classical_populations"] = np.full((2 ** args.N, 2 ** args.N), np.nan)
        results.append(result)

        if args.warm_start and result["has_valid_candidate"]:
            initial = np.array([result["parameters"][name] for name in PARAMETER_NAMES])

    save_sweep(results, snapshot_path, args)
    print(f"Saved beta optimization sweep to {snapshot_path}")


if __name__ == "__main__":
    main()
