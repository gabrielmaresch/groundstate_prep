from pathlib import Path
import argparse
import sys

import matplotlib
import numpy as np
from matplotlib.widgets import Slider

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import (
    check_if_TFIM_gibbs,
    get_averaged_channel,
    get_mixing_time,
    get_superoperator_spectral_data,
    next_running_number,
)

if "--load" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--T", type=float, default=25.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--omega_max", type=float, default=8.0)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.2)
    parser.add_argument(
        "--normalize_Jh",
        help="Whether to normalize the Hamiltonian.",
        action="store_true",
    )
    parser.add_argument("--beta-min", dest="beta_min", type=float, default=0.5)
    parser.add_argument("--beta-max", dest="beta_max", type=float, default=20.0)
    parser.add_argument("--beta-points", dest="beta_points", type=int, default=10)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--npz-number", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--save-channel",
        help="Whether to store the full channel matrices in the .npz file.",
        action="store_true",
    )
    return parser.parse_args()


def save_sweep_snapshot(sweep_data, snapshot_path, save_channel):
    temp_path = snapshot_path.with_name(snapshot_path.stem + ".tmp" + snapshot_path.suffix)
    channel_entries = [entry["channel"] for entry in sweep_data] if save_channel else []
    np.savez_compressed(
        temp_path,
        beta=np.array([entry["beta"] for entry in sweep_data]),
        channels=np.stack(channel_entries) if save_channel else np.array([]),
        eigvals=np.stack([entry["spectrum_data"]["eigvals"] for entry in sweep_data]),
        Delta2=np.array([entry["spectrum_data"]["Delta2"] for entry in sweep_data]),
        Delta_th=np.array([entry["spectrum_data"]["Delta_th"] for entry in sweep_data]),
        num_closer=np.array([entry["spectrum_data"]["num_closer"] for entry in sweep_data]),
        trace_distance=np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data]),
        get_mixingtime=np.array([entry["spectrum_data"]["get_mixingtime"] for entry in sweep_data]),
    )
    temp_path.replace(snapshot_path)


def precompute_sweep(
    op_set,
    *,
    N,
    T,
    alpha,
    sigma,
    omega_max,
    tau,
    J,
    h,
    normalize_Jh,
    beta_values,
    snapshot_path,
    save_channel,
):
    sweep_data = []
    J_hot, h_hot = J, h
    if normalize_Jh:
        H_norm = N * np.sqrt(J_hot**2 + h_hot**2)
        J_hot = J_hot / H_norm
        h_hot = h_hot / H_norm

    h_sys = transverse_ising_hamiltonian(J_hot, h_hot, N)

    for beta in beta_values:
        print(f"Computing beta={beta:.4g}")
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
            method="choi",
        )
        eigvals, fixedpoint, num_closer, Delta2, Delta_th = get_superoperator_spectral_data(
            channel, beta, [N, J_hot, h_hot]
        )
        _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J_hot, h_hot])
        mixing_time = get_mixing_time(channel, fixedpoint)

        sweep_data.append(
            {
                "beta": beta,
                "channel": channel,
                "channel_params": channel_params,
                "spectrum_data": {
                    "eigvals": eigvals,
                    "Delta2": float(Delta2),
                    "Delta_th": float(Delta_th),
                    "num_closer": int(num_closer),
                    "trace_distance": float(trace_distance),
                    "get_mixingtime": np.nan if mixing_time is None else float(mixing_time),
                },
            }
        )
        save_sweep_snapshot(sweep_data, snapshot_path, save_channel)

    return sweep_data


def draw_entry(ax, bar_ax, entry, extent, *, N, T, alpha, sigma, omega_max, tau, J, h):
    eigvals = entry["spectrum_data"]["eigvals"]
    Delta2 = entry["spectrum_data"]["Delta2"]
    Delta_th = entry["spectrum_data"]["Delta_th"]
    num_closer = entry["spectrum_data"]["num_closer"]
    trace_distance = entry["spectrum_data"]["trace_distance"]

    ax.clear()
    ax.scatter(eigvals.real, eigvals.imag, s=12)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re($\\lambda$)")
    ax.set_ylabel("Im($\\lambda$)")

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1)

    info_h = f"N = {N}\nJ = {J}\nh = {h}"
    info_ch = (
        f"$\\beta$ = {entry['beta']:.2f}\n"
        f"$T$ = {T}\n"
        f"$\\tau$ = {tau}\n"
        f"$\\omega_{{\\max}}$ = {omega_max}\n"
        f"$\\alpha$ = {alpha}\n"
        f"$\\sigma$ = {sigma}\n"
    )
    info_so = (
        f"num_closer = {num_closer}\n"
        f"$\\Delta_2$ = {Delta2:.4f}\n"
        f"$\\Delta_{{\\mathrm{{th}}}}$ = {Delta_th:.4f}\n"
    )

    ax.text(
        0.03,
        0.97,
        info_h,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )
    ax.text(
        0.97,
        0.03,
        info_ch,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )
    ax.text(
        0.97,
        0.97,
        info_so,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    bar_ax.clear()
    bar_ax.bar([0], [trace_distance], width=0.6, color="tab:orange")
    bar_ax.set_xlim(-0.75, 0.75)
    bar_ax.set_ylim(0, 1)
    bar_ax.set_xticks([])
    bar_ax.set_ylabel("trace distance")
    bar_ax.set_title(r"$\|\rho_{\rm fix} - \rho_\beta\|_1$")
    bar_ax.text(0, trace_distance, f"{trace_distance:.4f}", ha="center", va="bottom")


def build_figure(sweep_data, beta_grid, *, N, T, alpha, sigma, omega_max, tau, J, h):
    fig = plt.figure(figsize=(9.5, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    bar_ax = fig.add_subplot(gs[0, 1])
    plt.subplots_adjust(bottom=0.18)

    all_eigvals = np.concatenate([entry["spectrum_data"]["eigvals"] for entry in sweep_data])
    extent = max(1.05, np.max(np.abs(np.concatenate([all_eigvals.real, all_eigvals.imag]))))

    draw_entry(
        ax,
        bar_ax,
        sweep_data[0],
        extent,
        N=N,
        T=T,
        alpha=alpha,
        sigma=sigma,
        omega_max=omega_max,
        tau=tau,
        J=J,
        h=h,
    )

    slider_ax = fig.add_axes([0.15, 0.07, 0.7, 0.03])
    slider = Slider(
        ax=slider_ax,
        label=r"$\beta$",
        valmin=beta_grid[0],
        valmax=beta_grid[-1],
        valinit=beta_grid[0],
        valstep=beta_grid,
        valfmt="%.2f",
    )

    fig._beta_slider = slider

    def update(beta):
        idx = np.argmin(np.abs(beta_grid - beta))
        draw_entry(
            ax,
            bar_ax,
            sweep_data[idx],
            extent,
            N=N,
            T=T,
            alpha=alpha,
            sigma=sigma,
            omega_max=omega_max,
            tau=tau,
            J=J,
            h=h,
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)

    return fig


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
    normalize_Jh = args.normalize_Jh
    beta_values = np.linspace(args.beta_min, args.beta_max, args.beta_points)

    data_dir = args.data_dir
    plot_dir = args.plot_dir
    save_channel = args.save_channel

    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_number = next_running_number(data_dir, "npz")
    snapshot_path = data_dir / f"superoperator_N{N}_sweep_{snapshot_number}.npz"

    if args.load:
        fallback = snapshot_number - 1
        load_npz_number = args.npz_number if args.npz_number is not None else fallback
        matches = list(data_dir.glob(f"superoperator_N*_sweep_{load_npz_number}.npz"))
        if not matches:
            matches = list(data_dir.glob(f"superoperator_N*_sweep_{fallback}.npz"))
        saved = np.load(matches[0])
        beta_grid = saved["beta"]
        channels = saved["channels"] if "channels" in saved.files else np.array([])
        mixing_times = saved["get_mixingtime"] if "get_mixingtime" in saved.files else None
        sweep_data = [
            {
                "beta": beta,
                "channel": channel if channels.size else None,
                "spectrum_data": {
                    "eigvals": eigvals,
                    "Delta2": float(Delta2),
                    "Delta_th": float(Delta_th),
                    "num_closer": int(num_closer),
                    "trace_distance": float(trace_distance),
                    "get_mixingtime": np.nan if mix_time is None else float(mix_time),
                },
            }
            for beta, channel, eigvals, Delta2, Delta_th, num_closer, trace_distance, mix_time in zip(
                saved["beta"],
                channels if channels.size else [None] * len(saved["beta"]),
                saved["eigvals"],
                saved["Delta2"],
                saved["Delta_th"],
                saved["num_closer"],
                saved["trace_distance"],
                mixing_times if mixing_times is not None else [np.nan] * len(saved["beta"]),
            )
        ]
    else:
        op_set = construct_opset(N, type="XZ")
        sweep_data = precompute_sweep(
            op_set,
            N=N,
            T=T,
            alpha=alpha,
            sigma=sigma,
            omega_max=omega_max,
            tau=tau,
            J=J,
            h=h,
            normalize_Jh=normalize_Jh,
            beta_values=beta_values,
            snapshot_path=snapshot_path,
            save_channel=save_channel,
        )
        beta_grid = beta_values

    figure = build_figure(
        sweep_data,
        beta_grid,
        N=N,
        T=T,
        alpha=alpha,
        sigma=sigma,
        omega_max=omega_max,
        tau=tau,
        J=J,
        h=h,
    )

    running_png = next_running_number(plot_dir, "png")
    output_path = plot_dir / f"superoperator_beta_sweep_overview_{running_png}.png"

    figure.savefig(output_path, dpi=200)
    if args.load:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
