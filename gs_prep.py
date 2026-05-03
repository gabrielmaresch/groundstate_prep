import pennylane as qml
import pennylane.numpy as np
import random

def transverse_ising_hamiltonian(J:float, h:float, N:int, boundary_condition: str='periodic'):
    coeffs = []
    ops = []
    for i in range(N-1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i)@qml.PauliZ(i+1))
    
    if boundary_condition == 'periodic':    
        coeffs.append(-J)
        ops.append(qml.PauliZ(N-1)@qml.PauliZ(0))
   
    for i in range(N):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))


    return qml.Hamiltonian(coeffs, ops) 




def construct_interaction_hamiltonian(num_qubits:int, sigma:float, alpha = 1, rng = 42):
    random.seed(rng)
    op_set = []
    for i in range(num_qubits-2):
        for j in range(i, num_qubits-2):
            op_set.append(qml.PauliX(i)@qml.PauliX(j))

    sign = random.choice([-1,1])
    A = np.random.choice(op_set)
    B = (qml.PauliX(num_qubits-1) - 1j * qml.PauliY(num_qubits-1)) / 2
    f = random.gauss(0, sigma)

    ops = [A@B, A.dag()@B.dag()]
    coeffs = [sign*alpha*f, sign*alpha*f]

    return qml.Hamiltonian(coeffs, ops)

def construct_environmental_hamiltonian(num_qubits:int, omega_max):
    omega = random.uniform(0, omega_max)
    return qml.Hamiltonian([-omega/2], [qml.PauliZ(num_qubits-1)])





N = 3
t = 1
dev = qml.device("default.qubit", wires=N)
@qml.qnode(dev)

def circuit(t):
    H = transverse_ising_hamiltonian(1, 1, N)
    qml.TimeEvolution(H, t)
    return qml.state()

print(circuit(1.0))
print(qml.draw(circuit)(1.0))