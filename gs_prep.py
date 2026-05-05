import pennylane as qml
import pennylane.numpy as np

import random
from typing import Any, List
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
def construct_opset(num_qubits:int, *, output:bool = False):
    with qml.QueuingManager.stop_recording():
        op_set = []
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if output:
                    print(i,j)
                op_set.append(qml.PauliX(i)@qml.PauliX(j))
                op_set.append(-(qml.PauliX(i)@qml.PauliX(j)))
    return op_set




def construct_interaction_hamiltonian(num_qubits:int, sigma:float, t:float,  A:"qml.Operator", alpha:float = 1, *, hermitian:bool = True):
    # num_qubits is system qubits without environment

    
    with qml.QueuingManager.stop_recording():
        # environment
        B = (qml.PauliX(num_qubits) - 1j * qml.PauliY(num_qubits)) / 2
        # B_dagger = (qml.PauliX(num_qubits) + 1j * qml.PauliY(num_qubits)) / 2
        f = evaluate_sqrt_gaussian(sigma, t)
        
        if hermitian:
            return qml.Hamiltonian([alpha*f], [A@qml.PauliX(num_qubits)]) 
        else:
            ops = [A@B, qml.adjoint(A@B)]
        
        coeffs = [alpha*f, alpha*f]
        H_int = qml.Hamiltonian(coeffs, ops)
    
    return H_int

def construct_environmental_hamiltonian(num_qubits:int, omega:float):
    return qml.Hamiltonian([-omega/2], [qml.PauliZ(num_qubits)])


def construct_W_layer(num_qubits:int, m:int, tau:float, T:float, sigma:float, op:"qml.Operator", alpha:float = 1):
    t = (m+1)/2*tau-T
    H_int = construct_interaction_hamiltonian(num_qubits, sigma, t, op, alpha)
    
    #probably better to make this time evolution explicit
    #qml.ApproxTimeEvolution(H_int, tau, 1)
    qml.evolve(H_int, coeff=tau)
    
    return None

def construct_U_layers(num_qubits:int, tau:float, T:float, sigma:float, op:"qml.Operator", omega:float, H_sys:qml.Hamiltonian, alpha:float = 1):
    
    M = int(np.ceil(2*T/tau)) # number of substeps  
    # print(M, "substeps")
 
    H_env = construct_environmental_hamiltonian(num_qubits, omega)
    
    construct_W_layer(N, 1, tau/2, T, sigma, op, alpha )
    for i in range(M):
        #qml.ApproxTimeEvolution(H_env, tau, 1)
        qml.evolve(H_env, coeff=tau)
        # this should later be trotterized

        #qml.ApproxTimeEvolution(H_sys, tau, 1)
        qml.evolve(H_sys, coeff=tau)
        if i < M-1:
            construct_W_layer(N, 1, tau, T, sigma, op, alpha )
        else:
            construct_W_layer(N, 1, tau/2, T, sigma, op, alpha )
    
    return None


def initialize_rho_env(wire:int, beta:float, omega:float, rng:int = 42, *, first:bool = False):
    
    if not first:
        qml.measure(wires=[wire], reset=True)
    
    Z = np.exp(omega*beta/2) + np.exp(-omega*beta/2)
    p0, p1 = np.exp(omega*beta/2)/Z, np.exp(-omega*beta/2)/Z
    

    # exp(beta Z) is diagonal, so we sample from the density matrix
    random.seed(rng)
    if random.uniform(0.0, 1.0) > p0:
        qml.PauliX(wires=wire)
    #rho_env = np.diag(np.array([p0, p1], dtype=complex)) 
    #qml.QubitDensityMatrix(rho_env, wires=[wire])

    return None

def sample_operator(op_set:List["qml.Operator"], rng:int = 24):
    random.seed(rng)
    A = random.choice(op_set)
    print("chosen operator: ", A)
    return A

def sample_omega(omega_max:float, rng:int = 24):
    random.seed(rng)
    omega = random.uniform(0, omega_max)
    print("chosen omega:", np.round(omega, 2))
    return omega


############### helper function ###########
def get_evolutiontime(op:"qml.Hamiltonian"):
    if isinstance(op, qml.ApproxTimeEvolution):
        evo_time = float(op.parameters[0])
        print(evo_time)
    elif isinstance(op, qml.ops.op_math.Evolution):
        evo_time = np.abs(op.coeff)
    return evo_time



if __name__ == "__main__":

    print("hello world")
    N = 3
    T = 1
    alpha = 1
    sigma = 1
    omega_max = 5
    beta = 10 

    tau = 1/2
    num_timesteps = 2

    op_set = construct_opset(N)
    A, omega = sample_operator(op_set), sample_omega(omega_max)

    H_sys  = transverse_ising_hamiltonian(-1, 0, N)

    # we might need a mixed device, because the auxilary qubit is thermal 
    dev = qml.device("default.qubit", wires=N+num_timesteps)
    @qml.qnode(dev) #we need extra qubits for deferred mid circuit measurment

    def circuit():
        
        for i in range(num_timesteps):
            is_first = (i==0)
            initialize_rho_env(N, beta, omega, first=is_first)
            construct_U_layers(N, tau, T, sigma, A, omega, H_sys)
            #qml.measure(wires=N_env)
            qml.Barrier(wires=range(N + 1), only_visual=True)
        
        return qml.density_matrix(wires=range(N))
            
    final_state = circuit()
    print(np.round(final_state, 4))


    


    ###################### display circuit nicely ###################
    display_circuit = True
    if display_circuit:
        from matplotlib.text import Text
            
        qscript = qml.tape.make_qscript(circuit.func)()
        all_ops = list(qscript.operations)

        evolv_ops = {"W_tau": [], "W_tau_half": [], "H_env": [], "H_sys": []}
        for op in all_ops:
            #print(op.name, list(op.wires), getattr(op, "coeff", None))
            if "Evolution" in op.name:
                if [N] == list(op.wires):
                    evolv_ops["H_env"].append(op)
                elif N not in list(op.wires):
                    evolv_ops["H_sys"].append(op)
                elif np.isclose(abs(get_evolutiontime(op)), tau):
                    evolv_ops["W_tau"].append(op)
                else:
                    evolv_ops["W_tau_half"].append(op)
    

        fig, ax = qml.draw_mpl(circuit)()

        exp_texts = [artist for artist in fig.findobj(Text) if "Exp(" in artist.get_text()]
        evolve_ops_in_order = [op for op in all_ops if "Evolution" in op.name]



        for artist, op in zip(exp_texts, evolve_ops_in_order):
            if op in evolv_ops["H_env"]:
                artist.set_text(r"$e^{i\omega\tau Z/2}$")
            elif op in evolv_ops["W_tau_half"]:
                artist.set_text(r"$W(\tau/2)$")
            elif op in evolv_ops["W_tau"]:
                artist.set_text(r"$W(\tau)$")
            else:
                artist.set_text(r"$e^{-i\tau H_{\mathrm{sys}}}$")

        fig.savefig("circuit_"+str(num_timesteps)+"_steps.png", dpi=100, bbox_inches=None)
        plt.close(fig)