"""Collect a parallel 4D grid over h, alpha, sigma, and omega_max with live progress."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import product
from pathlib import Path
import argparse
import sys
import time

import numpy as np

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import (
    check_if_TFIM_gibbs,
    get_averaged_channel_matrix,
    get_transition_generator_and_classical_populations,
    get_normality_residual,
    num_iterations,
    get_superoperator_spectral_data,
    next_running_number,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--T", type=float, default=25.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--eps_fit", type=float, default=0.05)
    parser.add_argument(
        "--method",
        type=str,
        default="superoperator",
        choices=("superoperator", "kraus"),
        help="Channel construction method.",
    )
    parser.add_argument(
        "--normalize_Jh",
        help="Whether to normalize the Hamiltonian.",
        action="store_true",
    )
    parser.add_argument(
        "--save-channel",
        help="Whether to store the full channel matrices in the output file.",
        action="store_true",
    )
    parser.add_argument(
        "--save-classical-populations",
        help="Store the transition generator and classical population map.",
        action="store_true",
    )
    parser.add_argument(
        "--dense-spectrum",
        help="Diagonalize the full dense channel spectrum instead of using ARPACK.",
        action="store_true",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("data/grid"))
    parser.add_argument("--save-as-nr", type=int, default=-1)

    parser.add_argument("--h_min", type=float, default=0.2)
    parser.add_argument("--h_max", type=float, default=2.0)
    parser.add_argument("--h_points", type=int, default=10)
    parser.add_argument("--alpha_min", type=float, default=0.25)
    parser.add_argument("--alpha_max", type=float, default=1.5)
    parser.add_argument("--alpha_points", type=int, default=6)
    parser.add_argument("--sigma_min", type=float, default=0.5)
    parser.add_argument("--sigma_max", type=float, default=2.0)
    parser.add_argument("--sigma_points", type=int, default=5)
    parser.add_argument("--omega_min", type=float, default=4.0)
    parser.add_argument("--omega_max", type=float, default=8.0)
    parser.add_argument("--omega_points", type=int, default=2)
    return parser.parse_args()


def validate_dense_size(N, dense_requested, *, max_dense_dim=4096):
    d_so = 2 ** (2 * N)
    if dense_requested and d_so > max_dense_dim:
        raise ValueError(
            f"Dense channel use would require a {d_so}x{d_so} matrix. "
            f"Refusing because max_dense_dim={max_dense_dim}."
        )


def compute_single_point(
    *,
    N,
    T,
    alpha,
    sigma,
    omega_max,
    tau,
    J,
    h,
    beta,
    eps_fit,
    method,
    normalize_Jh,
    save_channel,
    save_classical_populations,
    dense_spectrum,
    verbose=False,
):
    if verbose:
        print(f"Computing h={h:.4g}, alpha={alpha:.4g}, sigma={sigma:.4g}, omega_max={omega_max:.4g}", flush=True)

    J_hot, h_hot = J, h
    if normalize_Jh:
        H_norm = N * np.sqrt(J_hot**2 + h_hot**2)
        J_hot = J_hot / H_norm
        h_hot = h_hot / H_norm

    op_set = construct_opset(N, type="XZ")
    h_sys = transverse_ising_hamiltonian(J_hot, h_hot, N)
    channel, channel_params = get_averaged_channel_matrix(
        N,
        tau,
        T,
        sigma,
        op_set,
        omega_max,
        h_sys,
        alpha,
        beta,
        method=method,
    )
    eigvals, fixedpoint, num_closer, Delta_sep, Delta_gap, Delta_th = get_superoperator_spectral_data(
        channel,
        beta,
        [N, J_hot, h_hot],
        full_spectrum=dense_spectrum,
    )
    _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J_hot, h_hot])
    normality_residual = get_normality_residual(channel)
    iteration_count = num_iterations(channel, fixedpoint, eps=eps_fit)

    result = {
        "N": N,
        "T": T,
        "alpha": alpha,
        "sigma": sigma,
        "omega_max": omega_max,
        "tau": tau,
        "J": J,
        "h": h,
        "h_over_J": h / J,
        "beta": beta,
        "eps_fit": eps_fit,
        "method": method,
        "normalize_Jh": normalize_Jh,
        "channel_params": channel_params,
        "spectrum_data": {
            "eigvals": np.asarray(eigvals),
            "Delta_sep": float(Delta_sep),
            "Delta_gap": float(Delta_gap),
            "Delta_th": float(Delta_th),
            "num_closer": int(num_closer),
            "trace_distance": float(trace_distance),
            "normality_residual": float(normality_residual),
            "num_iterations": np.nan if iteration_count is None else float(iteration_count),
        },
    }
    if save_channel:
        result["channel"] = channel
    if save_classical_populations:
        result["transition_generator"], result["classical_populations"] = get_transition_generator_and_classical_populations(
            channel, h_sys, op_set, beta, omega_max, sigma
        )

    return result


def flatten_grid(args):
    h_values = np.linspace(args.h_min, args.h_max, args.h_points)
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    sigma_values = np.linspace(args.sigma_min, args.sigma_max, args.sigma_points)
    omega_values = np.linspace(args.omega_min, args.omega_max, args.omega_points)
    return h_values, alpha_values, sigma_values, omega_values


def worker(point, *, fixed):
    h, alpha, sigma, omega_max = point
    return compute_single_point(
        N=fixed["N"],
        T=fixed["T"],
        alpha=alpha,
        sigma=sigma,
        omega_max=omega_max,
        tau=fixed["tau"],
        J=fixed["J"],
        h=h,
        beta=fixed["beta"],
        eps_fit=fixed["eps_fit"],
        method=fixed["method"],
        normalize_Jh=fixed["normalize_Jh"],
        save_channel=fixed["save_channel"],
        save_classical_populations=fixed["save_classical_populations"],
        dense_spectrum=fixed["dense_spectrum"],
        verbose=False,
    )


def format_point(point):
    h, alpha, sigma, omega_max = point
    return f"h={h:.4g}, alpha={alpha:.4g}, sigma={sigma:.4g}, omega={omega_max:.4g}"


def render_status(completed, total, active_points, elapsed):
    lines = [f"Progress: {completed}/{total} | elapsed {elapsed:.1f}s", "Active workers:"]
    if active_points:
        for index, point in enumerate(active_points, start=1):
            lines.append(f"  {index}: {format_point(point)}")
    else:
        lines.append("  none")
    return lines


def save_grid(rows, snapshot_path, save_channel, save_classical_populations, dense_spectrum):
    channel_entries = [row["channel"] for row in rows] if save_channel else []
    transition_generators = [row["transition_generator"] for row in rows] if save_classical_populations else []
    classical_populations = [row["classical_populations"] for row in rows] if save_classical_populations else []
    np.savez(
        snapshot_path,
        h=np.array([row["h"] for row in rows]),
        alpha=np.array([row["alpha"] for row in rows]),
        sigma=np.array([row["sigma"] for row in rows]),
        omega_max=np.array([row["omega_max"] for row in rows]),
        beta=np.array([row["beta"] for row in rows]),
        h_over_J=np.array([row["h_over_J"] for row in rows]),
        eigvals=np.stack([row["spectrum_data"]["eigvals"] for row in rows]),
        Delta_sep=np.array([row["spectrum_data"]["Delta_sep"] for row in rows]),
        Delta_gap=np.array([row["spectrum_data"]["Delta_gap"] for row in rows]),
        Delta_th=np.array([row["spectrum_data"]["Delta_th"] for row in rows]),
        num_closer=np.array([row["spectrum_data"]["num_closer"] for row in rows]),
        trace_distance=np.array([row["spectrum_data"]["trace_distance"] for row in rows]),
        normality_residual=np.array([row["spectrum_data"]["normality_residual"] for row in rows]),
        num_iterations=np.array([row["spectrum_data"]["num_iterations"] for row in rows]),
        channels=np.stack(channel_entries) if save_channel else np.array([]),
        transition_generator=np.stack(transition_generators) if save_classical_populations else np.array([]),
        classical_populations=np.stack(classical_populations) if save_classical_populations else np.array([]),
        dense_spectrum=dense_spectrum,
        save_classical_populations=save_classical_populations,
    )


def main():
    args = parse_args()
    validate_dense_size(args.N, args.save_channel or args.dense_spectrum)

    h_values, alpha_values, sigma_values, omega_values = flatten_grid(args)
    points = list(product(h_values, alpha_values, sigma_values, omega_values))

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.save_as_nr == -1:
        snapshot_number = next_running_number(data_dir, "npz")
    else:
        snapshot_number = args.save_as_nr

    snapshot_path = data_dir / f"superoperator_N{args.N}_grid_{snapshot_number}.npz"
    print("File will be saved as", snapshot_path)
    print(f"Grid size: {len(points)} points")

    fixed = {
        "N": args.N,
        "T": args.T,
        "J": args.J,
        "beta": args.beta,
        "tau": args.tau,
        "eps_fit": args.eps_fit,
        "method": args.method,
        "normalize_Jh": args.normalize_Jh,
        "save_channel": args.save_channel,
        "save_classical_populations": args.save_classical_populations,
        "dense_spectrum": args.dense_spectrum,
    }

    rows = []
    completed = 0
    total = len(points)
    started_at = time.monotonic()
    previous_render_lines = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_point = {executor.submit(worker, point, fixed=fixed): point for point in points}
        pending = set(future_to_point)
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                rows.append(future.result())
                completed += 1

            elapsed = time.monotonic() - started_at
            active_points = [future_to_point[future] for future in future_to_point if future.running()]
            status_lines = render_status(completed, total, active_points[: args.workers], elapsed)

            if previous_render_lines:
                sys.stdout.write(f"\x1b[{previous_render_lines}A")
            sys.stdout.write("\x1b[J")
            sys.stdout.write("\n".join(status_lines) + "\n")
            sys.stdout.flush()
            previous_render_lines = len(status_lines)

    if previous_render_lines:
        sys.stdout.write(f"\x1b[{previous_render_lines}A")
        sys.stdout.write("\x1b[J")
    sys.stdout.write("\n")
    sys.stdout.flush()

    save_grid(
        rows, snapshot_path, args.save_channel, args.save_classical_populations, args.dense_spectrum
    )
    print(f"Saved grid to {snapshot_path}")


if __name__ == "__main__":
    main()
