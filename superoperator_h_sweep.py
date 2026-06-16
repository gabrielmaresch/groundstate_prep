from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import (
    check_if_TFIM_gibbs,
    get_averaged_channel,
    get_mixing_time,
    get_superoperator_spectral_data,
    next_running_number,
)


N = 4
T = 25.0
alpha = 1.
#alpha = .75
#alpha = .5
#alpha = .25

sigma = 2.
omega_max = 20
tau = 0.1
J = 1.0
beta = 1.0
eps_fit = 0.05

#h_values = np.linspace(0.25, 1.0, 4)
h_values = np.linspace(0.1, 2.0, 20)

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
        get_mixingtime=np.array([entry["spectrum_data"]["get_mixingtime"] for entry in sweep_data]),
    )
    temp_path.replace(snapshot_path)


def precompute_sweep(op_set):
    sweep_data = []

    for h in h_values:
        print(f"Computing h={h:.4g}", sep='\t')
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
            method="choi",
        )
        eigvals, fixedpoint, num_closer, Delta2, Delta_th = get_superoperator_spectral_data(channel, beta, [N, J, h])
        _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J, h])
        mixing_time = get_mixing_time(channel, fixedpoint, eps=eps_fit)
        print(f"iterations for eps={eps_fit:.4g}: {mixing_time}")
        
        
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
                    "get_mixingtime": np.nan if mixing_time is None else float(mixing_time),
                },
            }
        )
        save_sweep_snapshot(sweep_data)

    return sweep_data


def main():
    op_set = construct_opset(N, type="XZ")
    print("\n"+60*'-'+"\n\n")
    ans = input("Load saved sweep-data? [y/n] ")

    if ans not in {"y", "yes", "Y", "Yes"}:
        sweep_data = precompute_sweep(op_set)
        h_grid = np.array([entry["h"] for entry in sweep_data])
        Delta2 = np.array([entry["spectrum_data"]["Delta2"] for entry in sweep_data])
        trace_distance = np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data])
        mixing_time = np.array([entry["spectrum_data"]["get_mixingtime"] for entry in sweep_data])
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
        mixing_time = saved["get_mixingtime"] if "get_mixingtime" in saved.files else None

    gap = np.maximum(Delta2, 1e-16)
    has_mixing_time = mixing_time is not None
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax_gap = axes[0, 0]
    ax_dist = axes[0, 1]
    ax_mix = axes[1, 0]
    ax_info = axes[1, 1]

    ax_gap.semilogy(h_grid, gap, marker="o")
    ax_gap.set_xlabel("h")
    ax_gap.set_ylabel(r"$\Delta_2$")
    ax_gap.set_title(r"$\Delta_2$ vs h")
    ax_gap.grid(True, which="both", linestyle=":")

    ax_dist.plot(h_grid, trace_distance, marker="o")
    ax_dist.set_xlabel("h")
    ax_dist.set_ylabel(r"$\|\rho_{\rm fix} - \rho_\beta\|_1$")
    ax_dist.set_title("fixed point distance")
    ax_dist.grid(True, linestyle=":")

    if has_mixing_time:
        ax_mix.plot(h_grid, mixing_time, marker="o")
        ax_mix.set_xlabel("h")
        ax_mix.set_ylabel(r"$t_{\rm mix}/T$")
        ax_mix.set_title("mixing time (iterations)")
        ax_mix.grid(True, linestyle=":")
    else:
        ax_mix.axis("off")

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

    output_folder = Path(__file__).resolve().parent / "plots"
    output_folder.mkdir(exist_ok=True)
    file_name = f"superoperator_N{N}_h_sweep_{next_running_number(output_folder)}.png"
    fig.savefig(output_folder / file_name, dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()