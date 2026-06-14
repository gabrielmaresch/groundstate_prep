import pennylane as qml
import pennylane.numpy as np
import random
import os
import sys
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from cooling_channel import construct_U_layers, transverse_ising_hamiltonian, construct_opset, sample_omega, sample_operator
from analytics import trace_distance
from scipy.sparse.linalg import eigs


################## Helper function for naming logic ###########
def next_running_number(folder, ext="png"):
    numbers = []
    for file in Path(folder).glob("*."+ext):
        match = re.search(rf"(\d+)(?=\.{ext}$)", file.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


##################
from juliacall import Main as jl

local_path = Path(__file__).resolve().parent
file_name = "ed.jl" 
jl.include(str(local_path/file_name))

def get_gibbs(N, J, h, beta):
    gibbs_state, energy = jl.get_transverse_ising_gibbsstate(N, J, h, beta)
    return gibbs_state, energy
####################



def U_parametrized_circuit(num_system_qubits, tau, T, sigma, op, omega, H_sys, alpha):
    def circuit():
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
    return circuit

def get_U_matrix(num_system_qubits, tau, T, sigma, op, omega, H_sys, alpha):
    circuit = U_parametrized_circuit(num_system_qubits, tau, T, sigma, op, omega, H_sys, alpha)
    qscript = qml.tape.make_qscript(circuit)()
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

def get_krausz_blocks(num_system_qubits, U):

    U00 = U[::2,::2] 
    U10 = U[1::2,::2]
    U01 = U[::2,1::2]
    U11 = U[1::2,1::2]

    return [[U00, U01], [U10, U11]]

def get_superoperator_matrix_krausz(num_system_qubits, U_blocks, beta, omega):
    d_sys = 2 ** num_system_qubits

    Z = np.exp(omega*beta/2) + np.exp(-omega*beta/2)
    p = [np.exp(omega*beta/2)/Z, np.exp(-omega*beta/2)/Z]

    S = np.zeros((d_sys**2,d_sys**2,), dtype = complex)  

    for a in range(2):
        for b in range(2):
            S += p[b]*np.kron(U_blocks[a][b],np.conj(U_blocks[a][b])) 

    return S


def get_superoperator_matrix_choi(num_system_qubits, U, omega, beta):
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

    J4 = choi.reshape(d_sys, d_sys, d_sys, d_sys)          # indices: i, a, j, b
    S = J4.transpose(1, 3, 0, 2).reshape(d_choi, d_choi)   # indices: a, b, i, j

    return S



def get_averaged_channel(N, tau, T, sigma, op_set, omega_max, H_sys, alpha, beta, *, omega_quadrature= ('midpoint', 10), method = 'choi'):
    
    rule, n_omega =  omega_quadrature
    if rule == 'midpoint':
        delta_omega = omega_max / n_omega
        omegas = [(k+0.5)*delta_omega for k in range(n_omega)]
    
    S = np.zeros((2**(2*N),2**(2*N)), dtype = complex)

    for op in op_set:
        for omega in omegas:
            U = get_U_matrix(N, tau, T, sigma, op, omega, H_sys, alpha)
            if method == 'choi':
                S += get_superoperator_matrix_choi(N, U, omega, beta)
            elif method == 'krausz':
                U_blocks = get_krausz_blocks(N, U)
                S += get_superoperator_matrix_krausz(N, U_blocks, beta, omega)


    averages = len(op_set)*n_omega
    S = S/averages

    S_params = (N, tau, T, sigma, op_set, omega_max, H_sys, alpha, beta, averages)

    return S, S_params

def get_superoperator_spectral_data(S,*, k_max=100):
    eigvals, eigvecs = eigs(S, k=k_max, which="LM")

    idx = np.argmin(np.abs(eigvals - 1))
    fixedpoint = eigvecs[:, idx]
    target = 1.0 + 0.0j
    degeneracy = np.sum(np.isclose(eigvals, target, atol=1e-3))

    #compute gap
    lambda2 = np.sort(abs(eigvals))[-2]
    return eigvals, fixedpoint, degeneracy, lambda2

def normalize_to_densitymatrix(A):
    A_dens = 0.5 * (A + A.conj().T)
    A_dens = A_dens / np.trace(A_dens)
    return A_dens

def check_if_TFIM_gibbs(test_vector, beta, TFIM_params, tol = 0.025):
    N, J, h = TFIM_params
    thermal, energy = get_gibbs(N, J, h, beta)
    thermal, energy = np.array(thermal), float(energy)
    
    test_state = normalize_to_densitymatrix(test_vector.reshape((2**N, 2**N)))
    dist = trace_distance(test_state, thermal)
    
    return (dist<tol), test_state, dist

def plot_superoperator_spectrum(S, S_params, J, h, output = True):
    
    N, tau, T, sigma, _, omega_max, _, alpha, beta, averages = S_params
    eigvals, _, degeneracy, lambda2 = get_superoperator_spectral_data(S)

    
    #### PLOT ###########
    plt.scatter(eigvals.real, eigvals.imag, s=10)
    plt.axhline(0, color="k", lw=0.5)
    plt.axvline(0, color="k", lw=0.5)
    plt.xlabel("Re($\\lambda$)")
    plt.ylabel("Im($\\lambda$)")
    plt.gca().set_aspect("equal", adjustable="box")

    info_so = f"degeneracy = {degeneracy}\n$|\\lambda_2|$ = {lambda2:.4f}"
    info_H = f"N = {N}\nJ = {J}\nh = {h}"
    info_ch = (
        f"$\\beta$ = {beta}\n"
        f"$T$ = {T}\n"
        f"$\\tau$ = {tau}\n"
        f"$\\omega_{{\\max}}$ = {omega_max}\n"
        f"$\\alpha$ = {alpha}\n"
        f"$\\sigma$ = {sigma}"
    )

    ax = plt.gca()
    ax.text(
        0.97,
        0.97,
        info_so,
        transform=ax.transAxes,
        ha="right",
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
        0.03,
        0.97,
        info_H,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), "k--", lw=1)

    path = Path(__file__).resolve().parent / "plots" 
    running_number = next_running_number(path)
    file_name = "superoperator_N"+str(N)+"_"+str(running_number)+".png" 

    plt.savefig(path/file_name, dpi=200)
    
    if output:
        plt.show()

    return None

def vectorize(rho):
    #we use row stacking
    return rho.reshape(-1)

def apply_channel(S, rho, output='matrix'):
    rho_vec = vectorize(rho)
    
    if output == 'matrix':
        rho_out = (S@rho_vec).reshape(rho.shape)
    elif output == 'vector':
        rho_out = (S@rho_vec)

    return rho_out



if __name__ == "__main__":


    # generic parameters for testing
    N = 4
    T = 10.
    alpha = 0.1
    sigma = 1.
    omega_max = 6.
    beta = 1.
    tau = 0.25
    op_set = construct_opset(N, type="XZ")
    J, h = 1., 4.5
    H_sys  = transverse_ising_hamiltonian(J, h, N)


    S, S_params = get_averaged_channel(N, tau, T, sigma, op_set, omega_max, H_sys, alpha, beta, method = 'choi')
    eigvals, fixedpoint, degeneracy, lambda2 = get_superoperator_spectral_data(S)
    
    correct_fp, test_state, dist_fp  = check_if_TFIM_gibbs(fixedpoint, beta, [N, J, h])
    if correct_fp:
        print("gibbs state is fixed-point")
    else:
        print("wrong fixed-point")
    print(f"tr-dist to thermal state: {dist_fp:.4f}")

    plot_superoperator_spectrum(S, S_params, J, h)


    # initialize rho
    rho = np.zeros((2**N, 2**N), dtype = complex)
    rho[0,0] = 1

    eps = 1e-3
    thermal, _ = get_gibbs(N, J, h, beta)
    thermal = np.array(thermal)
    fixedpoint = normalize_to_densitymatrix(fixedpoint.reshape((2**N, 2**N)))
    dist = trace_distance(rho, fixedpoint)
    num_iter = 0   
    while dist > eps and num_iter < 1000:
        num_iter += 1
        rho = normalize_to_densitymatrix(apply_channel(S, rho))       
        dist = trace_distance(rho, fixedpoint)
        if num_iter%10 == 0:
            print(f"{num_iter}: tr-dist to fixed point state: {dist:.4f}")
    
    dist = trace_distance(rho, thermal)
    print(f"{num_iter}: tr-dist to thermal state: {dist:.4f}")


