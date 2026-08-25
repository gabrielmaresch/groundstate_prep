from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy.sparse.linalg import LinearOperator

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import (
    check_if_TFIM_gibbs,
    get_U_matrix,
    get_kraus_blocks,
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
    parser.add_argument("--omega-points", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps_fit", type=float, default=0.05)
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--op-set", type=str, default="XZ")
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


def save_instance(result, snapshot_path, save_channel):
    np.savez_compressed(
        snapshot_path,
        h=result["h"],
        h_over_J=result["h_over_J"],
        beta=result["beta"],
        channel=result["channel"] if save_channel else np.array([]),
        eigvals=result["spectrum_data"]["eigvals"],
        Delta_sep=result["spectrum_data"]["Delta_sep"],
        Delta_gap=result["spectrum_data"]["Delta_gap"],
        Delta_th=result["spectrum_data"]["Delta_th"],
        num_closer=result["spectrum_data"]["num_closer"],
        trace_distance=result["spectrum_data"]["trace_distance"],
        normality_residual=result["spectrum_data"]["normality_residual"],
        num_iterations=result["spectrum_data"]["num_iterations"],
    )


def linear_operator_to_dense(channel):
    d_so = channel.shape[0]
    basis = np.eye(d_so, dtype=np.complex64)
    return np.column_stack([channel @ basis[:, j] for j in range(d_so)])


def averaged_blocks_as_linop(num_system_qubits, computed_instances, beta):
    d_sys = 2 ** num_system_qubits
    d_so = d_sys * d_sys
    weights = []
    for _, omega in computed_instances:
        # Z = np.exp(omega * beta / 2) + np.exp(-omega * beta / 2)
        # weights.append(
        #     np.array(
        #         [np.exp(omega * beta / 2) / Z, np.exp(-omega * beta / 2) / Z],
        #         dtype=np.complex64,
        #     )
        # )
        p0 = 1 / (1 + np.exp(-omega * beta))
        p1 = 1 - p0
        weights.append(
            np.array([p0, p1], dtype=np.complex64)
        )
    averages = len(computed_instances)

    def matvec(vec):
        rho = np.asarray(vec, dtype=np.complex64).reshape((d_sys, d_sys))
        out = np.zeros((d_sys, d_sys), dtype=np.complex64)

        for (U_blocks, _), p in zip(computed_instances, weights):
            for b in range(2):
                for a in range(2):
                    A = U_blocks[a][b]
                    out += p[b] * (A @ rho @ A.conj().T)

        return (out / averages).reshape(-1)

    return LinearOperator((d_so, d_so), matvec=matvec, dtype=np.complex64)


def compute_instance(
    contribution,
    *,
    N,
    tau,
    T,
    sigma,
    H_sys,
    alpha,
):
    op, omega = contribution
    U = get_U_matrix(N, tau, T, sigma, op, omega, H_sys, alpha)
    U_blocks = get_kraus_blocks(N, U)
    U_blocks = [
        [np.ascontiguousarray(block) for block in row]
        for row in U_blocks
    ]
    return U_blocks, omega


def compute_average(
    *,
    N,
    T,
    alpha,
    sigma,
    tau,
    beta,
    contributions,
    workers,
    H_sys,
):
    worker = partial(
        compute_instance,
        N=N,
        T=T,
        sigma=sigma,
        tau=tau,
        H_sys=H_sys,
        alpha=alpha,
    )

    computed_instances = []
    total = len(contributions)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for n, instance in enumerate(executor.map(worker, contributions), start=1):
            print(f"finished {n} of {total}", flush=True)
            computed_instances.append(instance)

    return averaged_blocks_as_linop(N, computed_instances, beta)


def main():
    args = parse_args()

    N = args.N
    T = args.T
    alpha = args.alpha
    sigma = args.sigma
    omega_max = args.omega_max
    omega_points = args.omega_points
    tau = args.tau
    J = args.J
    beta = args.beta
    eps_fit = args.eps_fit
    normalize_Jh = args.normalize_Jh
    save_channel = args.save_channel
    validate_save_channel_size(N, save_channel)
    workers = args.workers
    save_as_nr = args.save_as_nr
    op_set_type = args.op_set

    h = args.h
    if normalize_Jh:
        H_norm = N * np.sqrt(J**2 + h**2)
        J = J / H_norm
        h = h / H_norm
    H_sys = transverse_ising_hamiltonian(J, h, N)
    op_set = construct_opset(N, type=op_set_type)
    delta_omega = omega_max / omega_points
    omegas = [(k + 0.5) * delta_omega for k in range(omega_points)]
    averages = len(op_set) * omega_points
    channel_params = (N, tau, T, sigma, op_set, omega_max, H_sys, alpha, beta, averages)
    contributions = [(op, omega) for op in op_set for omega in omegas]

    print(f"Computing one channel instance: N={N}, J={J:.4g}, h={h:.4g}")

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if save_as_nr == -1:
        snapshot_number = next_running_number(data_dir, "npz")
    else:
        snapshot_number = save_as_nr

    snapshot_path = data_dir / f"superoperator_N{N}_single_{snapshot_number}.npz"
    print("File will be saved as", snapshot_path)

    channel = compute_average(
        N=N,
        T=T,
        alpha=alpha,
        sigma=sigma,
        tau=tau,
        beta=beta,
        contributions=contributions,
        workers=workers,
        H_sys=H_sys,
    )
    eigvals, fixedpoint, num_closer, Delta_sep, Delta_gap, Delta_th = get_superoperator_spectral_data(
        channel,
        beta,
        [N, J, h],
    )
    _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J, h])
    normality_residual = get_normality_residual(channel)
    iteration_count = num_iterations(channel, fixedpoint, eps=eps_fit)

    result = {
        "h": h,
        "h_over_J": h / J,
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
        result["channel"] = linear_operator_to_dense(channel)

    save_instance(result, snapshot_path, save_channel)


if __name__ == "__main__":
    main()
