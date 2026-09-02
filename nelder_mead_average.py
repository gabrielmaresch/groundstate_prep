"""Optimize averaged-channel parameters by repeatedly calling parallel_average.py."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import get_averaged_channel, get_transition_generator_and_classical_populations

PARAMETER_NAMES = ("alpha", "omega_max", "sigma", "T", "tau")
INITIAL = (0.75, 8.0, 2.0, 25.0, 0.1)
BOUNDS = ((0.01, 1.0), (1.0, 10.0), (0.5, 5.0), (10.0, 100.0), (0.01, 1.0))
INVALID_OBJECTIVE = 1e6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-over-J", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
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
    parser.add_argument("--initial", type=float, nargs=5, metavar=PARAMETER_NAMES, default=INITIAL)
    parser.add_argument(
        "--normalize-Jh", "--normalize_Jh", dest="normalize_Jh", action="store_true", default=True,
        help="Normalize J and h by N sqrt(J^2 + h^2) (the default).",
    )
    parser.add_argument(
        "--no-normalize-Jh", dest="normalize_Jh", action="store_false",
        help="Use the supplied physical J and h without normalization.",
    )
    parser.add_argument(
        "--save-classical-populations",
        help="Store final-optimum transition and population matrices.",
        action="store_true",
    )
    return parser.parse_args()


def run_parallel_average(args, parameters, output_file):
    command = [
        sys.executable, str(Path(__file__).with_name("parallel_average.py")),
        "--N", str(args.N), "--T", str(parameters["T"]),
        "--alpha", str(parameters["alpha"]), "--sigma", str(parameters["sigma"]),
        "--omega_max", str(parameters["omega_max"]), "--omega-points", str(args.omega_points),
        "--tau", str(parameters["tau"]), "--J", str(args.J), "--h", str(args.h_over_J * args.J),
        "--beta", str(args.beta), "--eps_fit", str(args.eps_fit), "--op-set", args.op_set,
        "--workers", str(args.workers), "--data-dir", str(args.output_dir),
        "--save-as-nr", "0", "--skip-normality-residual", "--skip-filename-info",
        "--dense-spectrum",
    ]
    if args.normalize_Jh:
        command.append("--normalize_Jh")
    subprocess.run(command, check=True)
    with np.load(output_file, allow_pickle=False) as archive:
        return float(archive["num_iterations"]), float(archive["trace_distance"])


def save_final_classical_matrices(args, best_valid):
    dimension = 2 ** args.N
    transition_generator = np.full((dimension, dimension), np.nan)
    classical_populations = np.full((dimension, dimension), np.nan)
    if best_valid is not None:
        parameters = best_valid["parameters"]
        J, h = args.J, args.h_over_J * args.J
        if args.normalize_Jh:
            scale = args.N * np.sqrt(J**2 + h**2)
            J, h = J / scale, h / scale
        operators = construct_opset(args.N, type=args.op_set)
        H_sys = transverse_ising_hamiltonian(J, h, args.N)
        channel, _ = get_averaged_channel(
            args.N,
            parameters["tau"],
            parameters["T"],
            parameters["sigma"],
            operators,
            parameters["omega_max"],
            H_sys,
            parameters["alpha"],
            args.beta,
        )
        transition_generator, classical_populations = get_transition_generator_and_classical_populations(
            channel,
            H_sys,
            operators,
            args.beta,
            parameters["omega_max"],
            parameters["sigma"],
        )
    np.savez_compressed(
        args.output_dir / "classical_matrices_0.npz",
        transition_generator=transition_generator,
        classical_populations=classical_populations,
    )


def main():
    args = parse_args()
    initial = np.asarray(args.initial, dtype=float)
    if args.N < 1 or args.J <= 0 or args.beta <= 0 or args.workers < 1:
        raise ValueError("N, J, beta, and workers must be positive.")
    if args.maxfev < len(PARAMETER_NAMES) + 1:
        raise ValueError("maxfev must allow one full Nelder-Mead simplex.")
    if any(value < lower or value > upper for value, (lower, upper) in zip(initial, BOUNDS)):
        raise ValueError(f"initial values must lie within {dict(zip(PARAMETER_NAMES, BOUNDS))}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"superoperator_N{args.N}_single_0.npz"
    history = []
    previous_values = None
    previous_iterations = None
    stagnation_count = 0
    stagnation_limit = max(1, int(np.ceil(0.05 * args.maxfev)))

    def objective(values):
        nonlocal previous_values, previous_iterations, stagnation_count
        alpha, omega_max, sigma, T, tau = map(float, values)
        parameters = {
            "alpha": alpha,
            "omega_max": omega_max,
            "sigma": sigma,
            "T": T,
            "tau": tau,
        }
        evaluation = len(history) + 1

        if previous_values is None:
            update = "initial " + ", ".join(f"{name}={value:.6g}" for name, value in parameters.items())
        else:
            deltas = values - previous_values
            update = "delta " + ", ".join(
                f"{name}={delta:+.6g}"
                for name, delta in zip(PARAMETER_NAMES, deltas)
                if not np.isclose(delta, 0.0)
            )
        previous_values = values.copy()
        print(f"Evaluation {evaluation}/{args.maxfev}: {update}", flush=True)

        iterations, trace_distance = run_parallel_average(args, parameters, output_file)
        if not np.isfinite(iterations) or not np.isfinite(trace_distance):
            status, objective_value = "non_finite_result", INVALID_OBJECTIVE
        elif trace_distance >= args.trace_distance_tol:
            status, objective_value = "gibbs_constraint_failed", INVALID_OBJECTIVE
        else:
            status, objective_value = "valid", iterations

        history.append({
            "evaluation": evaluation,
            "parameters": parameters,
            "num_iterations": iterations,
            "trace_distance": trace_distance,
            "status": status,
            "objective": objective_value,
        })
        print(
            f"  iterations={iterations:.6g} | trace_distance={trace_distance:.6g} | "
            f"objective={objective_value:.6g} ({status})",
            flush=True,
        )

        if status == "valid":
            stagnation_count = stagnation_count + 1 if iterations == previous_iterations else 1
            previous_iterations = iterations
            if stagnation_count >= stagnation_limit:
                raise StopIteration
        else:
            stagnation_count = 0
            previous_iterations = None
        return objective_value

    try:
        result = minimize(
            objective,
            initial,
            method="Nelder-Mead",
            bounds=BOUNDS,
            options={"maxfev": args.maxfev, "xatol": args.xatol, "fatol": args.fatol},
        )
    except StopIteration:
        result = None

    valid = [record for record in history if record["status"] == "valid"]
    best_valid = min(valid, key=lambda record: record["objective"], default=None)
    if result is None:
        success = True
        message = f"Stopped after {stagnation_limit} consecutive equal valid iteration counts."
        function_evaluations = len(history)
    else:
        success = bool(result.success)
        message = str(result.message)
        function_evaluations = int(result.nfev)

    summary = {
        "success": success,
        "message": message,
        "function_evaluations": function_evaluations,
        "best_valid_evaluation": best_valid,
        "fixed_parameters": {
            "N": args.N, "J": args.J, "h_over_J": args.h_over_J, "beta": args.beta,
            "omega_points": args.omega_points, "eps_fit": args.eps_fit, "op_set": args.op_set,
            "normalize_Jh": args.normalize_Jh, "workers": args.workers, "trace_distance_tol": args.trace_distance_tol,
            "dense_spectrum": True,
        },
        "history": history,
    }
    summary_path = args.output_dir / "summary_0.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_classical_populations:
        save_final_classical_matrices(args, best_valid)
    print(f"Optimization finished: {summary_path}")


if __name__ == "__main__":
    main()
