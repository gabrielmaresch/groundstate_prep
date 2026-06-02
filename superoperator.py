import pennylane as qml
import pennylane.numpy as np
import random
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from cooling_channel import construct_U_layers, transverse_ising_hamiltonian, construct_opset, sample_omega, sample_operator

from scipy.sparse.linalg import eigs


def get_U_matrix(num_system_qubits, tau, T, sigma, op, omega, H_sys, alpha):
    def U_wrapper():
        construct_U_layers(
            num_qubits=num_system_qubits,
            tau=tau,
            T=T,
            sigma=sigma,
            op=op,
            omega=omega,
            H_sys=H_sys,
            alpha=alpha,
            mixed=False,
        )

    qscript = qml.tape.make_qscript(U_wrapper)()
    return qml.matrix(qscript, wire_order=range(num_system_qubits + 1))

def get_choi_element(i,j, num_system_qubits, U, omega, beta):
    d_sys = 2 ** num_system_qubits
    rho_sys  = np.zeros((d_sys, d_sys), dtype=complex)
    rho_sys[i,j]  = 1 

    Z = np.exp(omega*beta/2) + np.exp(-omega*beta/2)
    rho_env = np.diag([np.exp(omega*beta/2)/Z, np.exp(-omega*beta/2)/Z])
    rho = np.kron(rho_sys, rho_env)

    rho_total = U@rho@U.conj().T
   
    rho_reshaped = rho_total.reshape(d_sys, 2, d_sys, 2)
    rho_sys_out = np.trace(rho_reshaped, axis1=1, axis2=3)
    
    return rho_sys_out


def get_superoperator_matrix(num_system_qubits, tau, T, sigma, op, omega, H_sys, alpha, beta):
    
    U = get_U_matrix(num_system_qubits, tau, T, sigma, op, omega, H_sys, alpha)
    
    d_sys = 2 ** num_system_qubits
    d_choi = d_sys*d_sys

    choi  = np.zeros((d_choi, d_choi), dtype=complex)

    for i in range(d_sys):
        for j in range(d_sys):
            sigma_ij = get_choi_element(i, j, num_system_qubits, U, omega, beta)
            for a in range(d_sys):
                for b in range(d_sys):
                    row = i * d_sys + a
                    col = j * d_sys + b
                    choi[row, col] = sigma_ij[a, b]

    J4 = choi.reshape(d_sys, d_sys, d_sys, d_sys)                       # indices: i, a, j, b
    S = J4.transpose(1, 3, 0, 2).reshape(d_choi, d_choi)   # indices: a, b, i, j

    return S

def get_averaged_channel(N, tau, T, sigma, op, omega, H_sys, alpha, beta, *, averages=50, output=False, k_max = 250):

    S = np.zeros((2**(2*N),2**(2*N)), dtype = complex)

    for i in range(averages):
        A = sample_operator(op_set, seed=3*i+1)
        omega =  sample_omega(omega_max, seed=3*i+2)
        S += get_superoperator_matrix(N, tau, T, sigma, A, omega, H_sys, alpha, beta)
    S = S/averages

    eigvals, eigvecs = eigs(S, k=k_max, which="LM")

    idx = np.argmin(np.abs(eigvals - 1))
    fixedpoint = eigvecs[:, idx]
    target = 0.0 + 1.0j
    degeneracy = np.sum(np.isclose(eigvals, target, atol=1e-4))

    ### sort eigenvals:
    lambda2 = np.sort(abs(eigvals))[1]

    plt.scatter(eigvals.real, eigvals.imag, s=10)
    plt.axhline(0, color="k", lw=0.5)
    plt.axvline(0, color="k", lw=0.5)
    plt.xlabel("Re($\\lambda$)")
    plt.ylabel("Im($\\lambda$)")
    plt.gca().set_aspect("equal", adjustable="box")

    info = f"degeneracy = {degeneracy}\n$\\lambda_2$ = {lambda2:.4f}"

    ax = plt.gca()
    ax.text(
        0.03,
        0.97,
        info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), "k--", lw=1)

    path = Path(__file__).resolve().parent
    file_name = "superoperator_N"+str(N)+".png" 

    plt.savefig(path/file_name, dpi=200)
    
    if output:
        plt.show()

    return S, [fixedpoint, degeneracy]



if __name__ == "__main__":


    # generic parameters for testing
    N = 4
    T = 10
    alpha = 1
    sigma = 0.5
    omega_max = 5
    beta = 1
    tau = 1/2
    
    op_set = construct_opset(N, type="XZ")
    A = sample_operator(op_set, seed=12)
    omega =  sample_omega(omega_max, seed=12)


    J, h = 1, 2
    H_sys  = transverse_ising_hamiltonian(J, h, N)

    averages = 50

