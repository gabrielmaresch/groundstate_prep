"""Exact diagonalization helpers for the transverse Ising model."""

from __future__ import annotations

import numpy as np


def _kron_all(operators):
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def transverse_ising_hamiltonian(N: int, J: float, h: float):
    """Construct the transverse Ising Hamiltonian with periodic boundary conditions."""
    if N <= 0:
        raise ValueError("N must be positive")

    H = np.zeros((2**N, 2**N), dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    for i in range(N):
        # Sz-Sz term with periodic boundary conditions
        if i == N - 1:
            z_ops = [Z if (j == N - 1 or j == 0) else I for j in range(N)]
        else:
            z_ops = [Z if (j == i or j == i + 1) else I for j in range(N)]
        H += -J * _kron_all(z_ops)

        # Sx term
        x_ops = [X if j == i else I for j in range(N)]
        H += -h * _kron_all(x_ops)

    return H


def thermal_state(H: np.ndarray, beta: float):
    """Return the Gibbs state and thermal energy for a Hermitian matrix."""
    H = np.asarray(H, dtype=np.complex128)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    evals, evecs = np.linalg.eigh(H)
    shifted_evals = evals - np.min(evals)
    beta_exps = np.exp(-beta * shifted_evals)
    Z = np.sum(beta_exps)
    weights = np.asarray(beta_exps / Z, dtype=np.complex128)
    gibbs = ((evecs.astype(np.complex128) * weights) @ evecs.conj().T).astype(np.complex128)
    energy = np.complex128(np.sum(evals * beta_exps) / Z)
    return gibbs, energy


def thermal_expectation_value(H: np.ndarray, A: np.ndarray, beta: float):
    """Return the thermal expectation value of A."""
    H = np.asarray(H, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    evals, evecs = np.linalg.eigh(H)
    shifted_evals = evals - np.min(evals)
    beta_exps = np.exp(-beta * shifted_evals)
    Z = np.sum(beta_exps)

    expectation = np.complex128(0.0)
    for i in range(H.shape[0]):
        ket = evecs[:, i]
        expectation += np.complex128(beta_exps[i] / Z) * np.complex128(ket.conj().T @ A @ ket)
    return np.complex128(expectation)


def ground_state(H: np.ndarray):
    """Return the ground-state vector and energy."""
    H = np.asarray(H, dtype=np.complex128)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    evals, evecs = np.linalg.eigh(H)
    return evecs[:, 0].astype(np.complex128), np.complex128(evals[0])


def ground_state_expectation_value(H: np.ndarray, A: np.ndarray):
    """Return the ground-state expectation value of A."""
    H = np.asarray(H, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    _, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0].astype(np.complex128)
    return np.complex128(gs.conj().T @ A @ gs)



def get_transition_generator_single(H0: np.ndarray, A: np.ndarray, beta: float, omega_max: float, sigma: float, quadrature_points=100):
    H0 = np.asarray(H0, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)
    if H0.ndim != 2 or H0.shape[0] != H0.shape[1]:
        raise ValueError("H0 must be square")
    if A.shape != H0.shape:
        raise ValueError("A must have the same shape as H0")
    if omega_max <= 0:
        raise ValueError("omega_max must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if quadrature_points < 2:
        raise ValueError("quadrature_points must be at least 2")

    E, U = np.linalg.eigh(H0)
    M = np.conj(U).T @ A @ U
    T = np.zeros_like(M, dtype=float) 

    n = H0.shape[0]
    omegas = np.linspace(0,omega_max, quadrature_points)
    for i in range(n):
        for j in range(n):
            def f(omega):
                return np.exp(-2*sigma**2*(E[i]-E[j]-omega)**2)
            
            int_plus = np.trapezoid([f(w)*(1+np.exp(-beta*w))**(-1) for w in omegas], omegas)
            int_minus = np.trapezoid([f(-w)*(1+np.exp(beta*w))**(-1) for w in omegas], omegas)
            if i != j:
                T[i,j] = np.abs(M[i,j])**2* int_plus + np.abs(M)[j,i]**2* int_minus
    for i in range(n):
        T[i,i] = -sum(T[:,i])

    T *= np.sqrt(8*np.sqrt(np.pi)) * sigma

    return T


def get_transition_generator_average(H0: np.ndarray, operators, beta: float, omega_max: float, sigma: float, quadrature_points=100):
    operators = list(operators)
    if not operators:
        raise ValueError("operators must not be empty")

    return sum(
        get_transition_generator_single(H0, A, beta, omega_max, sigma, quadrature_points)
        for A in operators
    ) / len(operators)
