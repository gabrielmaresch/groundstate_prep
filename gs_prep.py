import pennylane as qml
import pennylane.numpy as np
import random
import matplotlib.pyplot as plt

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

def evaluate_sqrt_gaussian(sigma:float, t:float):
    N = (2*np.pi*sigma**2)**(1/4)
    return 1/N*np.exp(-1/4*t**2/sigma**2)


### this is a very particular op_set for testing 
def construct_opset(num_qubits:int, *, output = False):
    with qml.QueuingManager.stop_recording():
        op_set = []
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if output:
                    print(i,j)
                op_set.append(qml.PauliX(i)@qml.PauliX(j))
                op_set.append(-(qml.PauliX(i)@qml.PauliX(j)))
    return op_set




def construct_interaction_hamiltonian(num_qubits:int, sigma:float, t:float,  A, alpha:float = 1, rng = 42):
    # num_qubits is system qubits without environment
    random.seed(rng)
    
    with qml.QueuingManager.stop_recording():
        # environment
        B = (qml.PauliX(num_qubits) - 1j * qml.PauliY(num_qubits)) / 2
        f = evaluate_sqrt_gaussian(sigma, t)

        ops = [A@B, qml.adjoint(A@B)]
        coeffs = [alpha*f, alpha*f]
        H_int = qml.Hamiltonian(coeffs, ops)
    #return f(AB*A†B)
    return H_int

def construct_environmental_hamiltonian(num_qubits:int, omega):
    return qml.Hamiltonian([-omega/2], [qml.PauliZ(num_qubits)])


def construct_W_layer(num_qubits:int, m:int, tau:float, T:float, sigma:float, op, alpha:float = 1):
    t = (m+1)/2*tau-T
    H_int = construct_interaction_hamiltonian(num_qubits, sigma, t, op, alpha)
    qml.evolve(H_int, coeff=tau/2)
    return None

def construct_U_layer(num_qubits:int, m:int, tau:float, T:float, sigma:float, op, omega, H_sys, alpha:float = 1):

    construct_W_layer(N, 1, tau, T, sigma, op, alpha )
    
    H_env = construct_environmental_hamiltonian(num_qubits, omega)
    qml.evolve(H_env, coeff=tau)
    # this should later be trotterized
    qml.evolve(H_sys, coeff=tau)

    construct_W_layer(N, 1, tau, T, sigma, op, alpha )
    
    return None


def initialize_rho_env(wire, beta, omega, *, first=False):
    if not first: 
        _ = qml.measure(wires=wire, reset=True)
    Z = np.exp(omega*beta/2) + np.exp(-omega*beta/2)
    rho_env = np.array([(np.exp(omega*beta/2)/Z)**(1/2), (np.exp(-omega*beta/2)/Z)**(1/2)], dtype=complex)  
    qml.StatePrep(rho_env, wires=[wire])

    return None

if __name__ == "__main__":

    print("hello world")
    N = 3
    T = 1
    alpha = 1
    sigma = 1
    omega_max = 5
    beta = 10 


    tau = 0.5
    M = int(np.ceil(2*T/tau))
    print(M, "substeps")

    op_set = construct_opset(N)
    #print(op_set)
    H_sys  = transverse_ising_hamiltonian(-1, 0, N)

    dev = qml.device("default.qubit", wires=N+1)
    @qml.qnode(dev)

    def circuit():
        A = random.choice(op_set)
        omega = random.uniform(0, omega_max)

        print("chosen operator: ", A)
        print("chosen omega:", np.round(omega, 2))
        #construct_W_layer(N, 1, tau, T, sigma, op_set, alpha )

        initialize_rho_env(N, beta, omega, first=True)
        qml.Barrier(wires=range(N + 1), only_visual=True)

        for _ in range(M):
            construct_U_layer(N, 1, tau, T, sigma, A, omega, H_sys, alpha)
            qml.Barrier(wires=range(N + 1), only_visual=True)

    
        return qml.state()

    #final_state = circuit()
    
    #print(np.round(final_state, 3))





    ###################### display circuit nicely ###################
    from matplotlib.text import Text

    fig, ax = qml.draw_mpl(circuit)()

    # Replace the default evolve label with a custom one
    for artist in fig.findobj(Text):
        txt = artist.get_text()
        if "Exp(-"+str(np.round(tau/2, 2)) in txt:
            artist.set_text(r"$W(\tau/2)$")
        elif "Exp(-"+str(np.round(tau, 2)) in txt:
            artist.set_text(r"$e^{-i\tau H_{\alpha}}$")      
    fig.savefig("circuit.png", dpi=100, bbox_inches=None)
    plt.close(fig)
