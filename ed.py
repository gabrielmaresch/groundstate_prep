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

    H = np.zeros((2**N, 2**N), dtype=np.complex64)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex64)

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
    H = np.asarray(H, dtype=np.complex64)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    evals, evecs = np.linalg.eigh(H)
    beta_exps = np.exp(-beta * evals)
    Z = np.sum(beta_exps)
    weights = np.asarray(beta_exps / Z, dtype=np.complex64)
    gibbs = ((evecs.astype(np.complex64) * weights) @ evecs.conj().T).astype(np.complex64)
    energy = np.complex64(np.sum(evals * beta_exps) / Z)
    return gibbs, energy


def thermal_expectation_value(H: np.ndarray, A: np.ndarray, beta: float):
    """Return the thermal expectation value of A."""
    H = np.asarray(H, dtype=np.complex64)
    A = np.asarray(A, dtype=np.complex64)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    evals, evecs = np.linalg.eigh(H)
    beta_exps = np.exp(-beta * evals)
    Z = np.sum(beta_exps)

    expectation = np.complex64(0.0)
    for i in range(H.shape[0]):
        ket = evecs[:, i]
        expectation += np.complex64(beta_exps[i] / Z) * np.complex64(ket.conj().T @ A @ ket)
    return np.complex64(expectation)


def ground_state(H: np.ndarray):
    """Return the ground-state vector and energy."""
    H = np.asarray(H, dtype=np.complex64)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    evals, evecs = np.linalg.eigh(H)
    return evecs[:, 0].astype(np.complex64), np.complex64(evals[0])


def ground_state_expectation_value(H: np.ndarray, A: np.ndarray):
    """Return the ground-state expectation value of A."""
    H = np.asarray(H, dtype=np.complex64)
    A = np.asarray(A, dtype=np.complex64)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    _, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0].astype(np.complex64)
    return np.complex64(gs.conj().T @ A @ gs)


def get_transverse_ising_groundstate(N: int, J: float, h: float):
    """Convenience wrapper returning the ground state and its energy."""
    H = transverse_ising_hamiltonian(N, J, h)
    return ground_state(H)


def get_transverse_ising_gibbsstate(N: int, J: float, h: float, beta: float):
    """Convenience wrapper returning the Gibbs state and its energy."""
    H = transverse_ising_hamiltonian(N, J, h)
    return thermal_state(H, beta)


def get_spectrum(H: np.ndarray):
    """Return the eigenvalues of H in ascending order."""
    H = np.asarray(H, dtype=np.complex64)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    return np.linalg.eigvalsh(H)