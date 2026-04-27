"""
Basic PennyLane circuit template for ground state preparation.

Implements the quantum-channel approach to ground state preparation following
arXiv:2508.05703 by Ding et al.

The core idea is to repeatedly apply a quantum channel to a system register.
Each channel step couples the system to a fresh ancilla register, applies a
joint unitary, and then discards (traces out) the ancilla.  After sufficiently
many steps the system's reduced density matrix converges toward the ground
state of the target Hamiltonian.
"""

import pennylane as qml
import numpy as np


# ---------------------------------------------------------------------------
# Hamiltonian helpers
# ---------------------------------------------------------------------------

def transverse_field_ising_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> qml.Hamiltonian:
    """Return the transverse-field Ising Hamiltonian on a 1-D chain.

    H = -J * sum_{i} Z_i Z_{i+1}  -  h * sum_{i} X_i

    Args:
        n_qubits: Number of system qubits.
        J: Ising coupling strength (default 1.0).
        h: Transverse field strength (default 0.5).

    Returns:
        A :class:`pennylane.Hamiltonian` object.
    """
    coeffs = []
    obs = []

    # ZZ interaction terms
    for i in range(n_qubits - 1):
        coeffs.append(-J)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

    # Transverse field terms
    for i in range(n_qubits):
        coeffs.append(-h)
        obs.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, obs)


# ---------------------------------------------------------------------------
# Quantum channel step
# ---------------------------------------------------------------------------

def channel_step_unitary(system_wires, ancilla_wires, params):
    """Apply one step of the quantum-channel circuit on system + ancilla wires.

    This subroutine implements the unitary part of one channel step:

    1. Entangle ancilla qubits with the system via CNOT gates (encoding the
       system state into the environment).
    2. Apply single-qubit rotations on the ancilla (energy-exchange layer).
    3. Apply parameterised RY rotations on all system qubits (variational layer).

    After calling this function the ancilla is discarded by the caller (i.e.
    the reduced state of the system register is retained).

    Args:
        system_wires: Wire labels for the system register.
        ancilla_wires: Wire labels for the ancilla register.  Must have the
            same length as *system_wires*.
        params: 1-D array of rotation angles.  Expected length:
            ``len(system_wires)`` angles for the ancilla RY layer plus
            ``len(system_wires)`` angles for the system RY layer.
    """
    n = len(system_wires)
    ancilla_params = params[:n]
    system_params = params[n:2 * n]

    # Initialise ancilla in the |0> state (already done by device reset)
    # Couple each system qubit to its paired ancilla qubit
    for s, a in zip(system_wires, ancilla_wires):
        qml.CNOT(wires=[s, a])

    # Parameterised rotations on ancilla (controls energy exchange)
    for i, a in enumerate(ancilla_wires):
        qml.RY(ancilla_params[i], wires=a)

    # Parameterised rotations on system (variational ansatz layer)
    for i, s in enumerate(system_wires):
        qml.RY(system_params[i], wires=s)

    # Entangle back: ancilla -> system (feedback)
    for s, a in zip(system_wires, ancilla_wires):
        qml.CNOT(wires=[a, s])


# ---------------------------------------------------------------------------
# QNode: energy expectation value
# ---------------------------------------------------------------------------

def make_energy_circuit(hamiltonian: qml.Hamiltonian, n_steps: int = 1):
    """Build and return a QNode that computes the energy expectation value.

    The circuit applies *n_steps* sequential quantum-channel steps to the
    system register.  Each step uses a dedicated set of ancilla wires
    (initialised to |0> by the device reset).  At the end it measures the
    expectation value of *hamiltonian* on the system register.

    Args:
        hamiltonian: Target Hamiltonian whose ground state we want to prepare.
        n_steps: Number of channel steps to apply (default 1).

    Returns:
        A callable ``energy(params)`` QNode where ``params`` is a 2-D array of
        shape ``(n_steps, 2 * n_system_qubits)``.
    """
    n_system = len(hamiltonian.wires)
    system_wires = list(range(n_system))
    # Allocate a fresh set of ancilla wires for each step so no mid-circuit
    # reset is needed — each ancilla set starts in |0> by device initialisation.
    all_ancilla = [
        list(range(n_system + step * n_system, n_system + (step + 1) * n_system))
        for step in range(n_steps)
    ]
    total_wires = system_wires + [w for anc in all_ancilla for w in anc]

    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def energy(params):
        """QNode: apply channel steps and return energy expectation value.

        Args:
            params: Array of shape ``(n_steps, 2 * n_system_qubits)``.

        Returns:
            Expectation value of the Hamiltonian.
        """
        for step in range(n_steps):
            channel_step_unitary(system_wires, all_ancilla[step], params[step])

        # Measure energy on the system register only
        return qml.expval(hamiltonian)

    return energy


# ---------------------------------------------------------------------------
# Simple demo / entry point
# ---------------------------------------------------------------------------

def main():
    """Demonstrate the basic circuit template on a 2-qubit TFIM Hamiltonian."""
    n_qubits = 2
    n_steps = 2

    hamiltonian = transverse_field_ising_hamiltonian(n_qubits, J=1.0, h=0.5)
    print("Hamiltonian:", hamiltonian)

    energy_circuit = make_energy_circuit(hamiltonian, n_steps=n_steps)

    # Random initial parameters: shape (n_steps, 2 * n_qubits)
    rng = np.random.default_rng(42)
    params = rng.uniform(-np.pi, np.pi, size=(n_steps, 2 * n_qubits))

    energy_value = energy_circuit(params)
    print(f"Energy expectation value: {energy_value:.6f}")

    # Compute the exact ground state energy for reference
    eigenvalues = np.linalg.eigvalsh(qml.matrix(hamiltonian, wire_order=list(range(n_qubits))))
    ground_energy = eigenvalues[0]
    print(f"Exact ground state energy: {ground_energy:.6f}")


if __name__ == "__main__":
    main()
