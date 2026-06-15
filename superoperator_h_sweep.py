from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import (
    check_if_TFIM_gibbs,
    get_averaged_channel,
    get_superoperator_spectral_data,
    next_running_number,
)


N = 4
T = 25.0
alpha = .25
sigma = 1.
omega_max = 20
tau = 0.1
J = 1.0
beta = 1.0

h_values = np.linspace(0.25, 4.0, 16)

path = Path(__file__).resolve().parent / "data"
running_npz = next_running_number(path, "npz")
file_name = "superoperator_N" + str(N) + "_h_sweep_" + str(running_npz) + ".npz"


def save_sweep_snapshot(sweep_data):
    snapshot_path = path / file_name
    temp_path = snapshot_path.with_name(snapshot_path.stem + ".tmp" + snapshot_path.suffix)
    np.savez_compressed(
        temp_path,
        h=np.array([entry["h"] for entry in sweep_data]),
        beta=np.array([entry["beta"] for entry in sweep_data]),
        channels=np.stack([entry["channel"] for entry in sweep_data]),
        eigvals=np.stack([entry["spectrum_data"]["eigvals"] for entry in sweep_data]),
        Delta2=np.array([entry["spectrum_data"]["Delta2"] for entry in sweep_data]),
        Delta_th=np.array([entry["spectrum_data"]["Delta_th"] for entry in sweep_data]),
        num_closer=np.array([entry["spectrum_data"]["num_closer"] for entry in sweep_data]),
        trace_distance=np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data]),
    )
    temp_path.replace(snapshot_path)


def precompute_sweep(op_set):
    sweep_data = []

    for h in h_values:
        print(f"Computing h={h:.4g}")
        h_sys = transverse_ising_hamiltonian(J, h, N)
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
            method="krausz",
        )
        eigvals, fixedpoint, num_closer, Delta2, Delta_th = get_superoperator_spectral_data(channel, beta, [N, J, h])
        _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J, h])
        sweep_data.append(
            {
                "h": h,
                "beta": beta,
                "channel": channel,
                "channel_params": channel_params,
                "spectrum_data": {
                    "eigvals": eigvals,
                    "Delta2": float(Delta2),
                    "Delta_th": float(Delta_th),
                    "num_closer": int(num_closer),
                    "trace_distance": float(trace_distance),
                },
            }
        )
        save_sweep_snapshot(sweep_data)

    return sweep_data


def main():
    op_set = construct_opset(N, type="XZ")
    print("\n\n\n")
    ans = input("Load saved sweep-data? [y/n] ")

    if ans not in {"y", "yes", "Y", "Yes"}:
        sweep_data = precompute_sweep(op_set)
        h_grid = np.array([entry["h"] for entry in sweep_data])
        Delta2 = np.array([entry["spectrum_data"]["Delta2"] for entry in sweep_data])
        trace_distance = np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data])
    else:
        fallback = next_running_number(path, "npz") - 1
        ans = input(f"running number of .npz? [{fallback}] ")
        if ans == "":
            ans = fallback
        load_npz_number = int(ans)
        matches = list(path.glob(f"superoperator_N*_h_sweep_{load_npz_number}.npz"))
        if not matches:
            matches = list(path.glob(f"superoperator_N*_h_sweep_{fallback}.npz"))
        saved = np.load(matches[0])
        h_grid = saved["h"]
        Delta2 = saved["Delta2"]
        trace_distance = saved["trace_distance"]

    gap = np.maximum(1.0 - np.abs(Delta2), 1e-16)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].semilogy(h_grid, gap, marker="o")
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("spectral gap")
    axes[0].set_title("gap vs h")
    axes[0].grid(True, which="both", linestyle=":")

    axes[1].plot(h_grid, trace_distance, marker="o")
    axes[1].set_xlabel("h")
    axes[1].set_ylabel(r"$\|\rho_{\rm fix} - \rho_\beta\|_1$")
    axes[1].set_title("fixed point distance")
    axes[1].grid(True, linestyle=":")

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
    fig.text(
        0.98,
        0.98,
        info,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    fig.tight_layout()

    output_folder = Path(__file__).resolve().parent / "plots"
    output_folder.mkdir(exist_ok=True)
    file_name = f"superoperator_N{N}_h_sweep_{next_running_number(output_folder)}.png"
    fig.savefig(output_folder / file_name, dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()