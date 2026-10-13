"""Small deterministic checks of channel representations and physical properties."""

import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm

from superoperator import (
    apply_channel,
    get_averaged_channel,
    get_averaged_channel_matrix,
    get_classical_populations,
    get_kraus_blocks,
    get_normality_residual,
    get_superoperator_matrix,
    get_superoperator_matrix_kraus,
    get_superoperator_spectral_data,
    get_U_matrix,
    linear_operator_to_dense,
    normalize_to_densitymatrix,
    superoperator_as_linop,
    vectorize,
)


@pytest.fixture(params=[1, 2], ids=["one-qubit", "two-qubit"])
def channel(request):
    """A generic complex unitary catches conjugation and wire-order mistakes."""
    n = request.param
    d = 2**n
    rng = np.random.default_rng(123 + n)
    a = rng.normal(size=(2*d, 2*d)) + 1j*rng.normal(size=(2*d, 2*d))
    u = expm(-0.2j * (a + a.conj().T))
    beta, omega = 0.7, 1.3
    dense = get_superoperator_matrix(n, u, omega, beta)
    linop = superoperator_as_linop(n, u, omega, beta)
    return n, u, beta, omega, dense, linop


def test_representations_and_partial_trace(channel):
    n, u, beta, omega, dense, linop = channel
    d = 2**n
    kraus = get_superoperator_matrix_kraus(n, get_kraus_blocks(n, u), beta, omega)
    np.testing.assert_allclose(kraus, dense, atol=1e-12)
    np.testing.assert_allclose(linear_operator_to_dense(linop), dense, atol=1e-12)
    # A complex, non-Hermitian input checks the full linear map, not just states.
    rng = np.random.default_rng(42)
    rho = rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d))
    p0 = 1 / (1 + np.exp(-beta * omega))
    joint = u @ np.kron(rho, np.diag([p0, 1-p0])) @ u.conj().T
    expected = np.trace(joint.reshape(d, 2, d, 2), axis1=1, axis2=3)
    np.testing.assert_allclose(apply_channel(dense, rho), expected, atol=1e-12)
    np.testing.assert_allclose(apply_channel(linop, rho, output="vector"),
                               vectorize(expected), atol=1e-12)
    np.testing.assert_allclose(linop.H @ vectorize(rho),
                               dense.conj().T @ vectorize(rho), atol=1e-12)


def test_channel_is_completely_positive_and_trace_preserving(channel):
    n, _, _, _, dense, linop = channel
    d = 2**n
    identity = vectorize(np.eye(d))
    np.testing.assert_allclose(linop.H @ identity, identity, atol=1e-12)
    # Choi block (i,j) is E(|i><j|), using row-stacked vectorization.
    choi = dense.reshape(d, d, d, d).transpose(2, 0, 3, 1).reshape(d*d, d*d)
    np.testing.assert_allclose(choi, choi.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(choi).min() >= -1e-12
    np.testing.assert_allclose(np.trace(choi), d, atol=1e-12)
    rho = np.eye(d) / d
    out = apply_channel(dense, rho)
    np.testing.assert_allclose(out, out.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(out), 1, atol=1e-12)
    assert np.linalg.eigvalsh(out).min() >= -1e-12
    assert np.max(np.abs(np.linalg.eigvals(dense))) <= 1 + 1e-12
    populations = get_classical_populations(linop, np.eye(d))
    assert populations.min() >= -1e-12
    np.testing.assert_allclose(populations.sum(axis=0), 1, atol=1e-12)


def test_spectrum_and_fixed_point_for_depolarizing_channel():
    d, contraction = 4, 0.6
    identity = vectorize(np.eye(d))
    channel = contraction*np.eye(d*d) + (1-contraction)*np.outer(identity, identity)/d
    eigvals, fixed, num_closer, separation, gap, thermal_dist = (
        get_superoperator_spectral_data(channel, 0, [2, 1, 0.4], full_spectrum=True)
    )
    np.testing.assert_allclose(np.sort(eigvals.real), [contraction]*15 + [1], atol=1e-12)
    np.testing.assert_allclose(channel @ fixed, fixed, atol=1e-12)
    np.testing.assert_allclose(normalize_to_densitymatrix(fixed.reshape(d, d)),
                               np.eye(d)/d, atol=1e-12)
    np.testing.assert_allclose([separation, gap, thermal_dist], [0.4, 0.4, 0], atol=1e-12)
    assert num_closer == 0
    assert get_normality_residual(channel) < 1e-12


def test_normality_residual_detects_non_normal_channel(channel):
    _, _, _, _, dense, linop = channel
    residual = get_normality_residual(dense, seed=5)
    assert residual > 1e-6
    np.testing.assert_allclose(get_normality_residual(linop, seed=5), residual, atol=1e-12)


def test_circuit_unitarity_and_averaged_channel_agreement():
    h_sys = qml.Hamiltonian([0.3], [qml.PauliZ(0)])
    args = (1, 0.1, 0.1, 1.0, [qml.PauliX(0), qml.PauliZ(0)], 1.5, h_sys, 0.4, 0.7)
    u = get_U_matrix(1, 0.1, 0.1, 1.0, qml.PauliX(0), 0.75, h_sys, 0.4)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-12)
    kwargs = {"omega_quadrature": ("midpoint", 2)}
    dense, _ = get_averaged_channel_matrix(*args, **kwargs)
    kraus, _ = get_averaged_channel_matrix(*args, method="kraus", **kwargs)
    linop, _ = get_averaged_channel(*args, **kwargs)
    np.testing.assert_allclose(kraus, dense, atol=1e-12)
    np.testing.assert_allclose(linear_operator_to_dense(linop), dense, atol=1e-12)
    np.testing.assert_allclose(linop.H @ vectorize(np.eye(2)),
                               vectorize(np.eye(2)), atol=1e-12)
