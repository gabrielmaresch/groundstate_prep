from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib

if "--load" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cooling_channel import construct_opset, transverse_ising_hamiltonian
from superoperator import (
    check_if_TFIM_gibbs,
    get_averaged_channel,
    get_mixing_time,
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
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps_fit", type=float, default=0.05)
    parser.add_argument("--h_min", type=float, default=0.75)
    parser.add_argument("--h_max", type=float, default=1.25)
    parser.add_argument("--h_points", type=int, default=26)
    parser.add_argument("--normalize_Jh", type=bool, default=False)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--npz-number", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"))
    return parser.parse_args()

'''
N = 4
T = 25.0

#alpha = 1.
alpha = .75
#alpha = .5
#alpha = .25

sigma = 2.
omega_max = 20
tau = 0.1
J = 1.0
beta = 1.0
eps_fit = 0.05

#h_values = np.linspace(0.25, 1.0, 4)
h_values = np.linspace(0.75, 1.25, 26)
'''




def save_sweep_snapshot(sweep_data, snapshot_path):
    temp_path = snapshot_path.with_name(snapshot_path.stem + ".tmp" + snapshot_path.suffix)
    np.savez_compressed(
        temp_path,
        h_over_J=np.array([entry["h_over_J"] for entry in sweep_data]),
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
    beta,
    eps_fit,
    h_values,
    normalize_Jh,
    snapshot_path,
):
    sweep_data = []

    for h in h_values:
        print(f"Computing h={h:.4g}", sep="\t")
        J_hot, h_hot = J, h
        if normalize_Jh:
            H_norm = N*(abs(J_hot)+abs(h_hot))
            J_hot = J_hot / H_norm
            h_hot = h_hot / H_norm
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
            method="choi",
        )
        eigvals, fixedpoint, num_closer, Delta2, Delta_th = get_superoperator_spectral_data(channel, beta, [N, J_hot, h_hot])
        _, _, trace_distance = check_if_TFIM_gibbs(fixedpoint, beta, [N, J_hot, h_hot])
        mixing_time = get_mixing_time(channel, fixedpoint, eps=eps_fit)
        print(f"iterations for eps={eps_fit:.4g}: {mixing_time}")
        
        
        sweep_data.append(
            {
                "h": h,
                "h_over_J": h / J,
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
        save_sweep_snapshot(sweep_data, snapshot_path)

    return sweep_data


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

    h_values = np.linspace(args.h_min, args.h_max, args.h_points)

    print("Sweep over h:", h_values)

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
        Delta2 = saved["Delta2"]
        trace_distance = saved["trace_distance"]
        mixing_time = saved["get_mixingtime"] if "get_mixingtime" in saved.files else None
    else:
        sweep_data = precompute_sweep(
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
            h_values=h_values,
            snapshot_path=snapshot_path,
        )
        h_grid = np.array([entry["h_over_J"] for entry in sweep_data])
        Delta2 = np.array([entry["spectrum_data"]["Delta2"] for entry in sweep_data])
        trace_distance = np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data])
        mixing_time = np.array([entry["spectrum_data"]["get_mixingtime"] for entry in sweep_data])
    
    
    #print("\n"+60*'-'+"\n\n")
    #ans = input("Load saved sweep-data? [y/n] ")

    #if ans not in {"y", "yes", "Y", "Yes"}:
    #    sweep_data = precompute_sweep(op_set)
    #    h_grid = np.array([entry["h"] for entry in sweep_data])
    #    Delta2 = np.array([entry["spectrum_data"]["Delta2"] for entry in sweep_data])
    #    trace_distance = np.array([entry["spectrum_data"]["trace_distance"] for entry in sweep_data])
    #    mixing_time = np.array([entry["spectrum_data"]["get_mixingtime"] for entry in sweep_data])
    #else:
    #    fallback = next_running_number(path, "npz") - 1
    #    ans = input(f"running number of .npz? [{fallback}] ")
    #    if ans == "":
    #        ans = fallback
    #    load_npz_number = int(ans)
    #    matches = list(path.glob(f"superoperator_N*_h_sweep_{load_npz_number}.npz"))
    #    if not matches:
    #        matches = list(path.glob(f"superoperator_N*_h_sweep_{fallback}.npz"))
    #    saved = np.load(matches[0])
    #    h_grid = saved["h"]
    #    Delta2 = saved["Delta2"]
    #    trace_distance = saved["trace_distance"]
    #    mixing_time = saved["get_mixingtime"] if "get_mixingtime" in saved.files else None

    gap = np.maximum(Delta2, 1e-16)
    has_mixing_time = mixing_time is not None
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax_gap = axes[0, 0]
    ax_dist = axes[0, 1]
    ax_mix = axes[1, 0]
    ax_info = axes[1, 1]

    ax_gap.semilogy(h_grid, gap, marker="o")
    ax_gap.set_xlabel(r"$h/J$")
    ax_gap.set_ylabel(r"$\Delta_2$")
    ax_gap.set_title(r"$\Delta_2$ vs $h/J$")
    ax_gap.grid(True, which="both", linestyle=":")

    ax_dist.plot(h_grid, trace_distance, marker="o")
    ax_dist.set_xlabel(r"$h/J$")
    ax_dist.set_ylabel(r"$\|\rho_{\rm fix} - \rho_\beta\|_1$")
    ax_dist.set_title("fixed point distance")
    ax_dist.grid(True, linestyle=":")

    if has_mixing_time:
        ax_mix.plot(h_grid, mixing_time, marker="o")
        ax_mix.set_xlabel(r"$h/J$")
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

    plot_number = next_running_number(plot_dir, "png")
    fig.savefig(plot_dir / f"superoperator_N{N}_h_sweep_{plot_number}.png", dpi=200, bbox_inches="tight")
    if args.load:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()