"""Evaluate channel non-normality at saved Nelder-Mead h-sweep optima."""

import argparse
from pathlib import Path

import numpy as np

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from parallel_average import compute_average
from superoperator import get_normality_residual


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nm_archive", type=Path, help="Nelder-Mead h-sweep archive with optimal parameters.")
    parser.add_argument("analysis_archive", type=Path, help="Grid-analysis archive to update.")
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def scalar(archive, key):
    value = np.asarray(archive[key])
    if value.ndim != 0:
        raise ValueError(f"Expected scalar metadata field {key!r}")
    return value.item()


def main():
    args = parse_args()
    if args.J <= 0:
        raise ValueError("J must be positive.")

    with np.load(args.nm_archive, allow_pickle=False) as archive:
        nm = {key: archive[key].copy() for key in archive.files}
    with np.load(args.analysis_archive, allow_pickle=False) as archive:
        analysis = {key: archive[key].copy() for key in archive.files}

    h_over_J = np.asarray(nm["h_over_J"], dtype=float)
    if not np.allclose(h_over_J, analysis["h_over_J"]):
        raise ValueError("The NM and grid-analysis h/J points do not agree.")

    N = int(scalar(nm, "N"))
    omega_points = int(scalar(nm, "omega_points"))
    op_set_type = str(scalar(nm, "op_set"))
    normalize_Jh = bool(scalar(nm, "normalize_Jh"))
    workers = args.workers if args.workers is not None else int(scalar(nm, "workers"))
    operators = construct_opset(N, type=op_set_type)
    residuals = []

    for index, h_ratio in enumerate(h_over_J):
        J_effective = args.J
        h_effective = h_ratio * args.J
        if normalize_Jh:
            scale = N * np.sqrt(J_effective**2 + h_effective**2)
            J_effective /= scale
            h_effective /= scale

        omega_max = float(nm["omega_max_opt"][index])
        omegas = (np.arange(omega_points) + 0.5) * omega_max / omega_points
        contributions = [(operator, omega) for operator in operators for omega in omegas]
        hamiltonian = transverse_ising_hamiltonian(J_effective, h_effective, N)
        channel = compute_average(
            N=N,
            T=float(nm["T_opt"][index]),
            alpha=float(nm["alpha_opt"][index]),
            sigma=float(nm["sigma_opt"][index]),
            tau=float(nm["tau_opt"][index]),
            beta=float(nm["beta"][index]),
            contributions=contributions,
            workers=workers,
            H_sys=hamiltonian,
        )
        residual = get_normality_residual(channel)
        residuals.append(residual)
        print(f"{index + 1}/{len(h_over_J)}: h/J={h_ratio:g}, residual={residual:.8g}", flush=True)

    analysis["normality_residual"] = np.asarray(residuals)
    temporary_path = args.analysis_archive.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **analysis)
    temporary_path.replace(args.analysis_archive)
    print(f"Updated {args.analysis_archive}")


if __name__ == "__main__":
    main()
