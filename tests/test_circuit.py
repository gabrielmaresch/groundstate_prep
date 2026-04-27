"""
Tests for the basic PennyLane circuit template (circuit.py).
"""

import numpy as np
import pennylane as qml
import pytest

from circuit import (
    channel_step_unitary,
    make_energy_circuit,
    transverse_field_ising_hamiltonian,
)


# ---------------------------------------------------------------------------
# transverse_field_ising_hamiltonian
# ---------------------------------------------------------------------------

class TestTransverseFieldIsingHamiltonian:
    def test_returns_hamiltonian(self):
        H = transverse_field_ising_hamiltonian(2)
        assert isinstance(H, qml.Hamiltonian)

    def test_single_qubit_has_only_field_terms(self):
        H = transverse_field_ising_hamiltonian(1, J=1.0, h=0.5)
        # 1 qubit: no ZZ term, only one X term
        assert len(H.coeffs) == 1
        assert H.coeffs[0] == pytest.approx(-0.5)

    def test_two_qubit_term_count(self):
        H = transverse_field_ising_hamiltonian(2, J=1.0, h=0.5)
        # 1 ZZ term + 2 X terms = 3
        assert len(H.coeffs) == 3

    def test_three_qubit_term_count(self):
        H = transverse_field_ising_hamiltonian(3, J=1.0, h=0.5)
        # 2 ZZ terms + 3 X terms = 5
        assert len(H.coeffs) == 5

    def test_coefficients_reflect_J_and_h(self):
        J, h = 2.0, 3.0
        H = transverse_field_ising_hamiltonian(2, J=J, h=h)
        assert -J in H.coeffs
        assert -h in H.coeffs

    def test_wires_match_n_qubits(self):
        for n in [2, 3, 4]:
            H = transverse_field_ising_hamiltonian(n)
            assert len(H.wires) == n


# ---------------------------------------------------------------------------
# channel_step_unitary
# ---------------------------------------------------------------------------

class TestChannelStepUnitary:
    def test_runs_without_error(self):
        """channel_step_unitary should execute inside a QNode without error."""
        n = 2
        system_wires = list(range(n))
        ancilla_wires = list(range(n, 2 * n))
        dev = qml.device("default.qubit", wires=system_wires + ancilla_wires)

        @qml.qnode(dev)
        def circuit(params):
            channel_step_unitary(system_wires, ancilla_wires, params)
            return qml.state()

        params = np.zeros(2 * n)
        state = circuit(params)
        assert state.shape == (2 ** (2 * n),)

    def test_params_affect_output(self):
        """Different parameters should produce different states."""
        n = 2
        system_wires = list(range(n))
        ancilla_wires = list(range(n, 2 * n))
        dev = qml.device("default.qubit", wires=system_wires + ancilla_wires)

        @qml.qnode(dev)
        def circuit(params):
            channel_step_unitary(system_wires, ancilla_wires, params)
            return qml.state()

        rng = np.random.default_rng(0)
        p1 = rng.uniform(-np.pi, np.pi, size=2 * n)
        p2 = rng.uniform(-np.pi, np.pi, size=2 * n)
        s1 = circuit(p1)
        s2 = circuit(p2)
        assert not np.allclose(s1, s2)


# ---------------------------------------------------------------------------
# make_energy_circuit
# ---------------------------------------------------------------------------

class TestMakeEnergyCircuit:
    def test_returns_callable(self):
        H = transverse_field_ising_hamiltonian(2)
        circuit = make_energy_circuit(H, n_steps=1)
        assert callable(circuit)

    def test_energy_is_scalar(self):
        H = transverse_field_ising_hamiltonian(2)
        circuit = make_energy_circuit(H, n_steps=1)
        n_system = len(H.wires)
        params = np.zeros((1, 2 * n_system))
        energy = circuit(params)
        assert np.ndim(energy) == 0

    def test_energy_above_ground_state(self):
        """The expectation value must be >= the exact ground state energy."""
        n = 2
        H = transverse_field_ising_hamiltonian(n, J=1.0, h=0.5)
        circuit = make_energy_circuit(H, n_steps=1)

        eigenvalues = np.linalg.eigvalsh(qml.matrix(H, wire_order=list(range(n))))
        ground_energy = float(eigenvalues[0])

        rng = np.random.default_rng(7)
        params = rng.uniform(-np.pi, np.pi, size=(1, 2 * n))
        energy = float(circuit(params))
        assert energy >= ground_energy - 1e-6

    def test_multi_step_circuit(self):
        """Circuit with multiple channel steps should return a valid energy."""
        n = 2
        n_steps = 3
        H = transverse_field_ising_hamiltonian(n)
        circuit = make_energy_circuit(H, n_steps=n_steps)
        params = np.zeros((n_steps, 2 * n))
        energy = circuit(params)
        assert np.isfinite(energy)

    def test_zero_params_reproducible(self):
        """With all-zero params the energy should be reproducible."""
        n = 2
        H = transverse_field_ising_hamiltonian(n)
        circuit = make_energy_circuit(H, n_steps=1)
        params = np.zeros((1, 2 * n))
        e1 = float(circuit(params))
        e2 = float(circuit(params))
        assert e1 == pytest.approx(e2)

    def test_different_hamiltonians(self):
        """Template should work for different Hamiltonian sizes."""
        for n in [2, 3]:
            H = transverse_field_ising_hamiltonian(n)
            circuit = make_energy_circuit(H, n_steps=1)
            params = np.zeros((1, 2 * n))
            energy = circuit(params)
            assert np.isfinite(energy)
