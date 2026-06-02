from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import (
    check_if_TFIM_gibbs,
    get_averaged_channel,
    get_superoperator_spectral_data,
)

######### the slider was implemented with the help of CODEX

N = 4
T = 5
alpha = 1
sigma = 1
omega_max = 5
tau = 0.25
J = 1
h = 2
averages = 25
k_max = 150
beta_values = np.geomspace(0.1, 50.0, 100)

path = Path(__file__).resolve().parent
file_name = "superoperator_N"+str(N)+"_sweep.npz"


def precompute_sweep(op_set, h_sys):
    sweep_data = []

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
            averages=averages,
        )
        eigvals, fixedpoint, degeneracy, lambda2 = get_superoperator_spectral_data(channel, k_max=k_max)
        _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J, h])
        sweep_data.append(
            {
                "beta": beta,
                "channel": channel,
                "channel_params": channel_params,
                "spectrum_data": {
                    "eigvals": eigvals,
                    "lambda2": float(lambda2),
                    "degeneracy": int(degeneracy),
                    "trace_distance": float(trace_distance),
                },
            }
        )
    np.savez_compressed(
        path / file_name,
        beta=np.array([entry["beta"] for entry in sweep_data]),
        channels=np.stack([entry["channel"] for entry in sweep_data]),
        eigvals=np.stack([entry["spectrum_data"]["eigvals"] for entry in sweep_data]),
        lambda2=np.array([entry["spectrum_data"]["lambda2"] for entry in sweep_data]),
        degeneracy=np.array([entry["spectrum_data"]["degeneracy"] for entry in sweep_data]),
        trace_distance=np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data]),
    )
    return sweep_data


def draw_entry(ax, bar_ax, entry, extent):
    eigvals = entry["spectrum_data"]["eigvals"]
    lambda2 = entry["spectrum_data"]["lambda2"]
    degeneracy = entry["spectrum_data"]["degeneracy"]
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
        f"$\\beta$ = {entry['beta']:.4g}\n"
        f"$T$ = {T}\n"
        f"$\\tau$ = {tau}\n"
        f"$\\omega_{{\\max}}$ = {omega_max}\n"
        f"$\\alpha$ = {alpha}\n"
        f"$\\sigma$ = {sigma}\n"
    )
    info_so = (
        f"degeneracy = {degeneracy}\n"
        f"$|\\lambda_2|$ = {lambda2:.4f}\n"
        f"averages = {averages}"
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
    bar_ax.set_title(f"$\\|\\rho^* - \\rho_\\beta\\|_1$")
    bar_ax.text(0, trace_distance, f"{trace_distance:.4f}", ha="center", va="bottom")


def build_figure(sweep_data):
    fig = plt.figure(figsize=(9.5, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    bar_ax = fig.add_subplot(gs[0, 1])
    plt.subplots_adjust(bottom=0.18)

    all_eigvals = np.concatenate([entry["spectrum_data"]["eigvals"] for entry in sweep_data])
    extent = max(1.05, np.max(np.abs(np.concatenate([all_eigvals.real, all_eigvals.imag]))))

    draw_entry(ax, bar_ax, sweep_data[0], extent)

    slider_ax = fig.add_axes([0.15, 0.07, 0.7, 0.03])
    slider = Slider(
        ax=slider_ax,
        label=r"$\beta$",
        valmin=beta_values[0],
        valmax=beta_values[-1],
        valinit=beta_values[0],
        valstep=beta_values,
        valfmt="%.2f",
    )

    fig._beta_slider = slider

    def update(beta):
        idx = np.argmin(np.abs(beta_values - beta))
        draw_entry(ax, bar_ax, sweep_data[idx], extent)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    return fig


def main():
    op_set = construct_opset(N, type="XZ")
    h_sys = transverse_ising_hamiltonian(J, h, N)
    print("\n\n\n")
    ans = input("Load saved sweep-data? [y/n] ")
    if ans not in {'y', 'yes', 'Y', 'Yes'}:
        sweep_data = precompute_sweep(op_set, h_sys)
    else:
        saved = np.load(path / file_name)
        sweep_data = [
            {
                "beta": beta,
                "channel": channel,
                "spectrum_data": {
                    "eigvals": eigvals,
                    "lambda2": float(lambda2),
                    "degeneracy": int(degeneracy),
                    "trace_distance": float(trace_distance),
                },
            }
            for beta, channel, eigvals, lambda2, degeneracy, trace_distance in zip(
                saved["beta"],
                saved["channels"],
                saved["eigvals"],
                saved["lambda2"],
                saved["degeneracy"],
                saved["trace_distance"],
            )
        ]
    
    figure = build_figure(sweep_data)

    output_path = Path(__file__).resolve().parent / "plots" / "superoperator_beta_sweep_overview.png"
    figure.savefig(output_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()