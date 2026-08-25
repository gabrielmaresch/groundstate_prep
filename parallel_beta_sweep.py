from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import (
    check_if_TFIM_gibbs,
    get_averaged_channel,
    get_normality_residual,
    num_iterations,
    get_superoperator_spectral_data,
    next_running_number,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--T", type=float, default=25.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--omega_max", type=float, default=20)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--eps_fit", type=float, default=0.05)
    parser.add_argument("--beta_min", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=2.0)
    parser.add_argument("--beta_points", type=int, default=20)
    parser.add_argument(
        "--normalize_Jh",
        help="Whether to normalize the Hamiltonian.",
        action="store_true",
    )
    parser.add_argument(
        "--save-channel",
        help="Whether to store the full channel matrices in the .npz file.",
        action="store_true",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--save-as-nr", type=int, default=-1)
    return parser.parse_args()


def validate_save_channel_size(N, save_channel, *, max_dense_dim=4096):
    d_so = 2 ** (2 * N)
    if save_channel and d_so > max_dense_dim:
        raise ValueError(
            f"--save-channel would require a dense {d_so}x{d_so} channel. "
            f"Refusing because max_dense_dim={max_dense_dim}."
        )


def save_sweep(sweep_data, snapshot_path, save_channel):
    channel_entries = [entry["channel"] for entry in sweep_data] if save_channel else []
    np.savez_compressed(
        snapshot_path,
        h=np.array([entry["h"] for entry in sweep_data]),
        beta=np.array([entry["beta"] for entry in sweep_data]),
        channels=np.stack(channel_entries) if save_channel else np.array([]),
        eigvals=np.stack([entry["spectrum_data"]["eigvals"] for entry in sweep_data]),
        Delta_sep=np.array([entry["spectrum_data"]["Delta_sep"] for entry in sweep_data]),
        Delta_gap=np.array([entry["spectrum_data"]["Delta_gap"] for entry in sweep_data]),
        Delta_th=np.array([entry["spectrum_data"]["Delta_th"] for entry in sweep_data]),
        num_closer=np.array([entry["spectrum_data"]["num_closer"] for entry in sweep_data]),
        trace_distance=np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data]),
        normality_residual=np.array([entry["spectrum_data"]["normality_residual"] for entry in sweep_data]),
        num_iterations=np.array([entry["spectrum_data"]["num_iterations"] for entry in sweep_data]),
    )


def compute_single_beta(
    beta,
    *,
    N,
    T,
    alpha,
    sigma,
    omega_max,
    tau,
    J,
    h,
    eps_fit,
    normalize_Jh,
    save_channel,
):
    print(f"Computing beta={beta:.4g}", flush=True)

    J_hot, h_hot = J, h
    if normalize_Jh:
        H_norm = N * np.sqrt(J_hot**2 + h_hot**2)
        J_hot = J_hot / H_norm
        h_hot = h_hot / H_norm

    op_set = construct_opset(N, type="XZ")
    h_sys = transverse_ising_hamiltonian(J_hot, h_hot, N)
    channel, channel_params = get_averaged_channel(
        N,
        tau,
        T,
        sigma,
        op_set,
        omega_max,
        h_sys,
        alpha,
        beta,
    )
    eigvals, fixedpoint, num_closer, Delta_sep, Delta_gap, Delta_th = get_superoperator_spectral_data(
        channel,
        beta,
        [N, J_hot, h_hot],
    )
    _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J_hot, h_hot])
    normality_residual = get_normality_residual(channel)
    iteration_count = num_iterations(channel, fixedpoint, eps=eps_fit)
    print(f"iterations for eps={eps_fit:.4g}: {iteration_count}", flush=True)

    result = {
        "h": h,
        "beta": beta,
        "channel_params": channel_params,
        "spectrum_data": {
            "eigvals": eigvals,
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

    return result


def compute_sweep(
    *,
    N,
    T,
    alpha,
    sigma,
    omega_max,
    tau,
    J,
    h,
    eps_fit,
    normalize_Jh,
    beta_values,
    save_channel,
    workers,
):
    worker = partial(
        compute_single_beta,
        N=N,
        T=T,
        alpha=alpha,
        sigma=sigma,
        omega_max=omega_max,
        tau=tau,
        J=J,
        h=h,
        eps_fit=eps_fit,
        normalize_Jh=normalize_Jh,
        save_channel=save_channel,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, beta_values))


def main():
    args = parse_args()

    N = args.N
    T = args.T
    alpha = args.alpha
    sigma = args.sigma
    omega_max = args.omega_max
    tau = args.tau
    J = args.J
    h = args.h
    eps_fit = args.eps_fit
    normalize_Jh = args.normalize_Jh
    save_channel = args.save_channel
    validate_save_channel_size(N, save_channel)
    workers = args.workers
    save_as_nr = args.save_as_nr

    beta_values = np.geomspace(args.beta_min, args.beta_max, args.beta_points)

    print("Sweep over beta:", beta_values)

    data_dir = args.data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    if save_as_nr == -1:
        snapshot_number = next_running_number(data_dir, "npz")
    else:
        snapshot_number = save_as_nr

    snapshot_path = data_dir / f"superoperator_N{N}_beta_sweep_{snapshot_number}.npz"
    print("File will be saved as", snapshot_path)

    sweep_data = compute_sweep(
        N=N,
        T=T,
        alpha=alpha,
        sigma=sigma,
        omega_max=omega_max,
        tau=tau,
        J=J,
        h=h,
        eps_fit=eps_fit,
        normalize_Jh=normalize_Jh,
        beta_values=beta_values,
        save_channel=save_channel,
        workers=workers,
    )
    save_sweep(sweep_data, snapshot_path, save_channel)


if __name__ == "__main__":
    main()
