import pennylane as qml
import numpy as np
import random
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Any, List
from cooling_channel import create_circuit_diagram, create_cooling_circuit, transverse_ising_hamiltonian, construct_opset
from scipy.optimize import curve_fit

from ed import get_transverse_ising_gibbsstate


local_path = Path(__file__).resolve().parent


def get_gibbs(N, J, h, beta):
    gibbs_state, energy = get_transverse_ising_gibbsstate(N, J, h, beta)
    return gibbs_state, energy


def nice_print_matrix(A, digits=3):
    A = np.asarray(A)
    fmt = f"{{:.{digits}f}}"
    for row in A:
        print("  ".join(fmt.format(x) for x in row))


def trace_distance(A, B):
    lambdas = np.linalg.eigvalsh(A - B)
    dist = 1 / 2 * sum(abs(ev) for ev in lambdas)
    return dist


def exp_model(x, a, b, c):
    return a + b * np.exp(c * x)


def extract_asymptotics(x, y):
    x_data = np.array(x)
    y_data = np.array(y)

    p_init = y_data[-1], y_data[0] - y_data[-1], -0.1
    p_fit, cov = curve_fit(
        exp_model,
        x_data,
        y_data,
        p0=p_init,
        bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, 0.0]),
    )

    y_fit = exp_model(x_data, *p_fit)

    return y_fit, p_fit, cov


#'____________________________ COOLING CICUIT EXECUTION _____________________________'

def run_TFIM(H_params, T_params, C_params, beta, ops='XZ', *, timesteps=None, mixed=True, show_gibbs=False, silent=False, plots=True, seed=12):

    assert len(H_params) == 3
    J, h, N = H_params
    H_sys = transverse_ising_hamiltonian(J, h, N)
    gibbs_state, energy = get_gibbs(N, J, h, beta)
    

    assert len(T_params) == 2 
    T, tau = T_params

    assert len(C_params) == 3 
    alpha, sigma, omega_max = C_params

    if ops in ['XZ', 'XX2']:
        op_set = construct_opset(N, type=ops)

    params = {
    "N": N,
    "J": J,
    "h": h,
    "beta": beta,
    "T": T,
    "tau": tau,
    "alpha": alpha,
    "sigma": sigma,
    "omega_max": omega_max,
    "ops": ops,
    "mixed": mixed,
    "seed": seed,
    }

    if not silent:
        print("\n" + 80 * "-")
        print("Start calculation")
        print(80 * "-")

        print("\nTransversal Ising XXZ model on", N, "qubits with J =", J, ", h =", h, ":\n")
        print("H_sys =", H_sys, "\n")
        if show_gibbs:
            print(f"\nGibbs state for β = {beta}\n")
            nice_print_matrix(gibbs_state)
            print("\n")


    if timesteps is None:
        s = input("\nEnter number of max. timesteps [100]: ")
        max_timesteps = int(s) if s else 100
        start = 5
        times = range(start, max(max_timesteps, start) + 1, 5) #minimum is 5 timesteps

    elif isinstance(timesteps, int):
        times = [timesteps]
        max_timesteps = timesteps
    elif isinstance(timesteps, (list, tuple, range, np.ndarray)):
        max_timesteps = max(timesteps)
        times = timesteps

    first_run = True
    tr_dist_gibbs = []
    tr_dist_increment = [0]

    for num_timesteps in times:
        
        circuit = create_cooling_circuit(
            num_system_qubits=N,
            num_timesteps=num_timesteps,
            H_sys=H_sys,
            tau=tau,
            beta=beta,
            T=T,
            omega_max=omega_max,
            op_set=op_set,
            alpha=alpha,
            sigma=sigma,
            seed=seed,
            mixed=mixed,
            output=False)

        final_state = circuit()
        
        dist = trace_distance(final_state, gibbs_state)
        if not silent:
            print(num_timesteps, " timesteps:\t", np.round(dist, 3))

        if not first_run:
            dist_increment = trace_distance(previous_state, final_state)
            if not silent:
                print("\t\t\t\t\t trace dist. increment", np.round(dist_increment, 3))
            tr_dist_increment.append(dist_increment)
    
        first_run = False
        previous_state = final_state
        tr_dist_gibbs.append(dist)

    info_H = f"N = {N}\nJ = {J}\nh = {h}"
    info_ch = (
        f"$\\beta$ = {beta}\n"
        f"$T$ = {T}\n"
        f"$\\tau$ = {tau}\n"
        f"$\\omega_{{\\max}}$ = {omega_max}\n"
        f"$\\alpha$ = {alpha}\n"
        f"$\\sigma$ = {sigma}"
    )
    if plots and len(times) > 2:
    
        plt.scatter(times, tr_dist_gibbs)
        plt.xlabel("number of timesteps")
        plt.title("trace distance cooled state vs. Gibbs state")

        plt.text(
            0.8, 0.95, info_H,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.text(
        0.8, 0.65, info_ch,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        dist_fitted, p_fit, cov = extract_asymptotics(times, tr_dist_gibbs)
        plt.plot(times, dist_fitted, color="green", linestyle=":")
        asymptotic_value = p_fit[0]
        plt.text(0.5, 0.85, f"Asymptotic value = {asymptotic_value:.3f}",
         transform=plt.gca().transAxes,
         verticalalignment="top",
         horizontalalignment="center")

        pic_dir = "plots"
        pic_name = "trace_distance_" + str(max_timesteps) + "timesteps.png"
        plt.savefig(local_path / pic_dir / pic_name, dpi=200)
        if not silent:
            plt.show()


    return np.array(times), np.array(tr_dist_gibbs), np.array(tr_dist_increment), params

#############################################################################################



if __name__ == "__main__":

### H_params = [J, h, N]
    H_params = [1., 2., 4]

### T_params = [T, tau]
    T_params = [10., 0.25]

### C_params = [alpha, sigma, omega_max]
    C_params = [0.75, 2., 20.]

    run_TFIM(H_params, T_params, C_params, beta=0.5, timesteps=20)
