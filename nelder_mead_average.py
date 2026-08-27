"""Optimize averaged-channel parameters by repeatedly calling parallel_average.py."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


NAMES = ("alpha", "omega_max", "sigma", "T", "tau")
BOUNDS = np.array(((0.01, 1.0), (1.0, 10.0), (0.5, 5.0), (10.0, 100.0), (0.01, 1.0)))
INITIAL = (0.75, 8.0, 2.0, 25.0, 0.1)
SIMPLEX_STEPS = (0.1, 1.0, 0.5, 10.0, 0.1)
INVALID_OBJECTIVE = 1e6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-over-J", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--omega-points", type=int, default=10)
    parser.add_argument("--eps-fit", type=float, default=0.05)
    parser.add_argument("--op-set", default="XZ")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trace-distance-tol", type=float, default=0.05)
    parser.add_argument("--maxfev", type=int, default=100)
    parser.add_argument("--xatol", type=float, default=1e-3)
    parser.add_argument("--fatol", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/nelder_mead"))
    parser.add_argument("--initial", type=float, nargs=5, metavar=NAMES, default=INITIAL)
    return parser.parse_args()


def main():
    args = parse_args()
    initial = np.asarray(args.initial, dtype=float)
    lower, upper = BOUNDS.T
    if args.N < 1 or args.beta <= 0 or args.workers < 1:
        raise ValueError("N, beta, and workers must be positive.")
    if args.maxfev < len(NAMES) + 1:
        raise ValueError("maxfev must allow one full Nelder-Mead simplex.")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError(f"initial values must lie within {dict(zip(NAMES, BOUNDS.tolist()))}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_file = args.output_dir / f"superoperator_N{args.N}_single_0.npz"
    history = []
    previous_values = None

    def objective(values):
        nonlocal previous_values
        values = np.asarray(values, dtype=float)
        parameters = dict(zip(NAMES, map(float, values)))
        violation = np.maximum(lower - values, 0) + np.maximum(values - upper, 0)
        record = {"evaluation": len(history) + 1, "parameters": parameters}
        if previous_values is None:
            update_text = "initial " + ", ".join(
                f"{name}={value:.6g}" for name, value in parameters.items()
            )
        else:
            deltas = values - previous_values
            update_text = "delta " + ", ".join(
                f"{name}={delta:+.6g}"
                for name, delta in zip(NAMES, deltas)
                if not np.isclose(delta, 0.0)
            )
        previous_values = values.copy()
        print(f"Evaluation {record['evaluation']}/{args.maxfev}: {update_text}", flush=True)
        if np.any(violation):
            record.update(status="outside_bounds", objective=float(INVALID_OBJECTIVE + np.sum(violation)))
            history.append(record)
            return record["objective"]

        command = [
            sys.executable, str(Path(__file__).with_name("parallel_average.py")),
            "--N", str(args.N), "--T", str(parameters["T"]),
            "--alpha", str(parameters["alpha"]), "--sigma", str(parameters["sigma"]),
            "--omega_max", str(parameters["omega_max"]), "--omega-points", str(args.omega_points),
            "--tau", str(parameters["tau"]), "--J", "1.0", "--h", str(args.h_over_J),
            "--beta", str(args.beta), "--eps_fit", str(args.eps_fit), "--op-set", args.op_set,
            "--normalize_Jh", "--workers", str(args.workers), "--data-dir", str(args.output_dir),
            "--save-as-nr", "0", "--skip-normality-residual", "--skip-filename-info",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0 or not evaluation_file.exists():
            record.update(status="subprocess_failed", objective=INVALID_OBJECTIVE)
        else:
            try:
                with np.load(evaluation_file, allow_pickle=False) as archive:
                    iterations = float(archive["num_iterations"])
                    trace_distance = float(archive["trace_distance"])
                record.update(num_iterations=iterations, trace_distance=trace_distance)
                if not np.isfinite(iterations) or not np.isfinite(trace_distance):
                    record.update(status="non_finite_result", objective=INVALID_OBJECTIVE)
                elif trace_distance >= args.trace_distance_tol:
                    record.update(status="gibbs_constraint_failed", objective=INVALID_OBJECTIVE + trace_distance)
                else:
                    record.update(status="valid", objective=iterations)
            except (KeyError, OSError, ValueError) as error:
                record.update(status=f"unreadable_result: {error}", objective=INVALID_OBJECTIVE)

        history.append(record)
        iterations = record.get("num_iterations", float("nan"))
        trace_distance = record.get("trace_distance", float("nan"))
        print(
            f"  iterations={iterations:.6g} | trace_distance={trace_distance:.6g} | objective={record['objective']:.6g} "
            f"({record['status']})",
            flush=True,
        )
        return record["objective"]

    simplex = np.vstack((initial, initial + np.eye(len(NAMES)) * SIMPLEX_STEPS))
    result = minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "maxfev": args.maxfev, "xatol": args.xatol, "fatol": args.fatol},
    )
    valid = [record for record in history if record["status"] == "valid"]
    summary = {
        "success": bool(result.success),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "optimizer_candidate": dict(zip(NAMES, map(float, result.x))),
        "best_valid_evaluation": min(valid, key=lambda record: record["objective"], default=None),
        "fixed_parameters": {
            "N": args.N, "J": 1.0, "h_over_J": args.h_over_J, "beta": args.beta,
            "omega_points": args.omega_points, "eps_fit": args.eps_fit, "op_set": args.op_set,
            "normalize_Jh": True, "workers": args.workers, "trace_distance_tol": args.trace_distance_tol,
        },
        "history": history,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Optimization finished: {summary_path}")


if __name__ == "__main__":
    main()
