from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib

if __name__ == "__main__" and "--load" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--omega_max", type=float, default=8)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps_fit", type=float, default=0.05)
    parser.add_argument("--skip-iterations", action="store_true", help="Skip fixed-point iteration-count calculations.")
    parser.add_argument("--h_min", type=float, default=0.1)
    parser.add_argument("--h_max", type=float, default=2.0)
    parser.add_argument("--h_points", type=int, default=5)
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
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--npz-number", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def save_sweep_snapshot(sweep_data, snapshot_path, save_channel, save_classical_populations, dense_spectrum):
    temp_path = snapshot_path.with_name(snapshot_path.stem + ".tmp" + snapshot_path.suffix)
    channel_entries = [entry["channel"] for entry in sweep_data] if save_channel else []
    transition_generators = [entry["transition_generator"] for entry in sweep_data] if save_classical_populations else []
    classical_populations = [entry["classical_populations"] for entry in sweep_data] if save_classical_populations else []
    np.savez_compressed(
        temp_path,
        h_over_J=np.array([entry["h_over_J"] for entry in sweep_data]),
        beta=np.array([entry["beta"] for entry in sweep_data]),
        channels=np.stack(channel_entries) if save_channel else np.array([]),
        transition_generator=np.stack(transition_generators) if save_classical_populations else np.array([]),
        classical_populations=np.stack(classical_populations) if save_classical_populations else np.array([]),
        eigvals=np.stack([entry["spectrum_data"]["eigvals"] for entry in sweep_data]),
        Delta_sep=np.array([entry["spectrum_data"]["Delta_sep"] for entry in sweep_data]),
        Delta_gap=np.array([entry["spectrum_data"]["Delta_gap"] for entry in sweep_data]),
        Delta_th=np.array([entry["spectrum_data"]["Delta_th"] for entry in sweep_data]),
        num_closer=np.array([entry["spectrum_data"]["num_closer"] for entry in sweep_data]),
        trace_distance=np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data]),
        normality_residual=np.array([entry["spectrum_data"]["normality_residual"] for entry in sweep_data]),
        num_iterations=np.array([entry["spectrum_data"]["num_iterations"] for entry in sweep_data]),
        dense_spectrum=dense_spectrum,
        save_classical_populations=save_classical_populations,
    )
    temp_path.replace(snapshot_path)


def compute_sweep(
    op_set,
    *,
    N,
    T,
    alpha,
    sigma,
    omega_max,
    tau,
    J,
    beta,
    eps_fit,
    skip_iterations,
    h_values,
    normalize_Jh,
    save_channel,
    save_classical_populations,
    dense_spectrum,
    snapshot_path,
):
    sweep_data = []

    print("Sweep over h:", h_values)

    for h in h_values:
        print(f"Computing h={h:.4g}", sep="\t")
        J_hot, h_hot = J, h
        if normalize_Jh:
            H_norm = N * np.sqrt(J_hot**2 + h_hot**2)
            J_hot = J_hot / H_norm
            h_hot = h_hot / H_norm
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
            method="superoperator",
        )
        eigvals, fixedpoint, num_closer, Delta_sep, Delta_gap, Delta_th = get_superoperator_spectral_data(
            channel, beta, [N, J_hot, h_hot], full_spectrum=dense_spectrum
        )
        _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J_hot, h_hot])
        normality_residual = get_normality_residual(channel)
        iteration_count = None if skip_iterations else num_iterations(channel, fixedpoint, eps=eps_fit)
        if not skip_iterations:
            print(f"iterations for eps={eps_fit:.4g}: {iteration_count}")
        
        
        entry = {
                "h": h,
                "h_over_J": h / J,
                "beta": beta,
                "channel": channel,
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
        if save_classical_populations:
            entry["transition_generator"], entry["classical_populations"] = get_transition_generator_and_classical_populations(
                channel, h_sys, op_set, beta, omega_max, sigma
            )
        sweep_data.append(entry)
        save_sweep_snapshot(
            sweep_data, snapshot_path, save_channel, save_classical_populations, dense_spectrum
        )

    return sweep_data

def show_plots(
    *,
    N,
    T,
    beta,
    tau,
    alpha,
    sigma,
    omega_max,
    J,
    h_grid,
    Delta_sep,
    trace_distance,
    iteration_counts,
    plot_dir,
    show_window=False,
):
    separation = np.maximum(Delta_sep, 1e-16)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax_sep = axes[0, 0]
    ax_dist = axes[0, 1]
    ax_iterations = axes[1, 0]
    ax_info = axes[1, 1]

    ax_sep.semilogy(h_grid, separation, marker="o")
    ax_sep.set_xlabel(r"$h/J$")
    ax_sep.set_ylabel(r"$\Delta_{\mathrm{sep}}$")
    ax_sep.set_title(r"$\Delta_{\mathrm{sep}}$ vs $h/J$")
    ax_sep.grid(True, which="both", linestyle=":")

    ax_dist.plot(h_grid, trace_distance, marker="o")
    ax_dist.set_xlabel(r"$h/J$")
    ax_dist.set_ylabel(r"$\|\rho_{\rm fix} - \rho_\beta\|_1$")
    ax_dist.set_title("fixed point distance")
    ax_dist.grid(True, linestyle=":")

    if iteration_counts is not None:
        ax_iterations.plot(h_grid, iteration_counts, marker="o")
        ax_iterations.set_xlabel(r"$h/J$")
        ax_iterations.set_ylabel("number of iterations")
        ax_iterations.set_title("iterations to fixed point")
        ax_iterations.grid(True, linestyle=":")
    else:
        ax_iterations.axis("off")

    info = (
        f"N = {N}\n"
        f"T = {T}\n"
        f"beta = {beta}\n"
        f"tau = {tau}\n"
        f"alpha = {alpha}\n"
        f"sigma = {sigma}\n"
        f"omega_max = {omega_max}\n"
        f"J = {J}"
    )
    ax_info.axis("off")
    ax_info.text(
        0.02,
        0.98,
        info,
        ha="left",
        va="top",
        transform=ax_info.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    fig.tight_layout()

    plot_number = next_running_number(plot_dir, "png")
    fig.savefig(plot_dir / f"superoperator_N{N}_h_sweep_{plot_number}.png", dpi=200, bbox_inches="tight")
    if show_window:
        plt.show()
    plt.close(fig)

def main():
    args = parse_args()

    N = args.N
    T = args.T
    alpha = args.alpha
    sigma = args.sigma
    omega_max = args.omega_max
    tau = args.tau
    J = args.J
    beta = args.beta
    eps_fit = args.eps_fit
    normalize_Jh = args.normalize_Jh
    save_channel = args.save_channel
    save_classical_populations = args.save_classical_populations
    dense_spectrum = args.dense_spectrum

    h_values = np.linspace(args.h_min, args.h_max, args.h_points)

    data_dir = args.data_dir
    plot_dir = args.plot_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    op_set = construct_opset(N, type="XZ")
    snapshot_number = next_running_number(data_dir, "npz")
    snapshot_path = data_dir / f"superoperator_N{N}_h_sweep_{snapshot_number}.npz"

    if args.load:
        fallback = snapshot_number - 1
        load_npz_number = args.npz_number if args.npz_number is not None else fallback
        matches = list(data_dir.glob(f"superoperator_N*_h_sweep_{load_npz_number}.npz"))
        if not matches:
            matches = list(data_dir.glob(f"superoperator_N*_h_sweep_{fallback}.npz"))
        saved = np.load(matches[0])
        h_grid = saved["h_over_J"] if "h_over_J" in saved.files else saved["h"]
        # channels = saved["channels"] if "channels" in saved.files else np.array([])
        Delta_sep = saved["Delta_sep"]
        trace_distance = saved["trace_distance"]
        iteration_counts = saved["num_iterations"] if "num_iterations" in saved.files else None
    else:
        sweep_data = compute_sweep(
            op_set,
            N=N,
            T=T,
            alpha=alpha,
            sigma=sigma,
            omega_max=omega_max,
            tau=tau,
            J=J,
            beta=beta,
            eps_fit=eps_fit,
            skip_iterations=args.skip_iterations,
            h_values=h_values,
            normalize_Jh=normalize_Jh,
            save_channel=save_channel,
            save_classical_populations=save_classical_populations,
            dense_spectrum=dense_spectrum,
            snapshot_path=snapshot_path,
        )
        h_grid = np.array([entry["h_over_J"] for entry in sweep_data])
        Delta_sep = np.array([entry["spectrum_data"]["Delta_sep"] for entry in sweep_data])
        trace_distance = np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data])
        iteration_counts = np.array([entry["spectrum_data"]["num_iterations"] for entry in sweep_data])
    
    show_plots(
        N=N,
        T=T,
        beta=beta,
        tau=tau,
        alpha=alpha,
        sigma=sigma,
        omega_max=omega_max,
        J=J,
        h_grid=h_grid,
        Delta_sep=Delta_sep,
        trace_distance=trace_distance,
        iteration_counts=iteration_counts,
        plot_dir=plot_dir,
        show_window=args.load,
    )



if __name__ == "__main__":
    main()
