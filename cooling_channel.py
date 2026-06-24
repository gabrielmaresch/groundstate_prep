import pennylane as qml
import pennylane.numpy as np

import random
from typing import Any, List
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path


""" 
H = -J ∑_<i,j> Z_i Z_j - h ∑_i X_i  
"""    
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
def construct_opset(num_qubits:int, *, type:str = "XX2", output:bool = False):
    with qml.QueuingManager.stop_recording():
        op_set = []

        if type == "XZ2":
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    if output:
                        print(i,j)
                    op_set.append(qml.PauliX(i)@qml.PauliX(j))
                    op_set.append(-(qml.PauliX(i)@qml.PauliX(j)))
                    op_set.append(qml.PauliZ(i)@qml.PauliZ(j))
                    op_set.append(-(qml.PauliZ(i)@qml.PauliZ(j)))

        elif type == "XZ":
            for i in range(num_qubits):
                op_set.extend([qml.PauliX(i), -qml.PauliX(i)])
                op_set.extend([qml.PauliZ(i), -qml.PauliZ(i)])
        
        elif type == "XYZ":
            for i in range(num_qubits):
                op_set.extend([qml.PauliX(i), -qml.PauliX(i)])
                op_set.extend([qml.PauliY(i), -qml.PauliY(i)])
                op_set.extend([qml.PauliZ(i), -qml.PauliZ(i)])

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


def construct_W_layer(num_qubits:int, m:int, tau_half:float, T:float, sigma:float, op:"qml.Operator", alpha:float = 1, *, mixed:bool):
    t = (2*m+1.)*tau_half-T
    H_int = construct_interaction_hamiltonian(num_qubits, sigma, t, op, alpha)
    
    #probably better to make this time evolution explicit
    
    if mixed:
        order = 2
        qml.ApproxTimeEvolution(H_int, tau_half, order)
    else:
        qml.evolve(H_int, coeff=tau_half)
    
    return None

def construct_U_layers(num_qubits:int, tau:float, T:float, sigma:float, op:"qml.Operator", omega:float, H_sys:qml.Hamiltonian, alpha:float = 1, *, mixed:bool):
    
    M = int(np.ceil(2*T/tau)) # number of substeps  
    # print(M, "substeps")
 
    # H_env = construct_environmental_hamiltonian(num_qubits, omega)
    
    for m in range(M):
        construct_W_layer(num_qubits, m, tau/2, T, sigma, op, alpha, mixed=mixed)
        #-------------------------------
        if mixed:
            order = 2
            qml.ApproxTimeEvolution(H_sys, tau, order)
        else:
            qml.evolve(H_sys, coeff=tau)
        # time evolution for H_env is just Z-rotation 
        qml.RZ(-tau*omega, wires=num_qubits)
        #--------------------------------
        construct_W_layer(num_qubits, m, tau/2, T, sigma, op, alpha, mixed=mixed)
    
    return None


def initialize_rho_env(wire:int, beta:float, omega:float, *, seed:int, mixed:bool = True):

    Z = np.exp(omega*beta/2) + np.exp(-omega*beta/2)
    p0, p1 = np.exp(omega*beta/2)/Z, np.exp(-omega*beta/2)/Z
    assert np.isclose(p0+p1, 1)

    if mixed:
        K0 = np.sqrt(p0) * np.array([[1, 0], [0, 0]], dtype=complex)
        K1 = np.sqrt(p0) * np.array([[0, 1], [0, 0]], dtype=complex)
        K2 = np.sqrt(p1) * np.array([[0, 0], [1, 0]], dtype=complex)
        K3 = np.sqrt(p1) * np.array([[0, 0], [0, 1]], dtype=complex)
        qml.QubitChannel([K0, K1, K2, K3], wires=wire)
    else:
        # exp(beta Z) is diagonal, so we sample as classical mixture
        qml.Reset(wires=wire)
        rng = random.Random(seed)
        if rng.uniform(0.0, 1.0) > p0:
            qml.PauliX(wires=wire)

    return None

def sample_operator(op_set:List["qml.Operator"], *, seed:int, output:bool = False):
    rng = random.Random(seed)
    A = rng.choice(op_set)
    if output:
        print("chosen operator: ", A)
    return A

def sample_omega(omega_max:float, *, seed:int, output:bool = False):
    rng = random.Random(seed)
    omega = rng.uniform(0, omega_max)
    if output:
        print("chosen omega:", np.round(omega, 2))
    return omega

############### this is the main function of this file #######

def create_cooling_circuit(
num_system_qubits:int,
num_timesteps:int, 
H_sys:"qml.Hamiltonian",    #system hamiltonian        
tau:int,                    #internal timestep for W-layer
beta:int,                   #inverse temperature
T:int,                      #time increment per timestep, i.e. applying the channel once
omega_max,                  #maximal frequency for up- and downsampling
op_set:List["qml.Operator"],#list of interaction operators, needs to be closed under sign-change and dagger
alpha:int=1,                #coupling parameter system-environment
sigma:int=1,                #inverse frequency width for gaussian filter
*,              
seed:int=42,                #answer to life, universe and everything
mixed:bool=True,            #circuit transforms either density matrices or pure states
output:bool = False):
    
    num_auxiliary_qubits = 1
    N = num_system_qubits + num_auxiliary_qubits
    
    device_type = "mixed" if mixed else "qubit"

    dev = qml.device("default."+device_type, wires=N)
    @qml.qnode(dev) 
    def circuit():

        for i in range(num_timesteps):
            if output:
                print("\ntimestep", i, ": ")

            A = sample_operator(op_set, seed=seed+3*i, output=output)
            omega =  sample_omega(omega_max, seed=seed+3*i+1, output=output)
            
            initialize_rho_env(N-1, beta, omega, seed=seed+3*i+2, mixed=mixed)
            construct_U_layers(num_system_qubits, tau, T, sigma, A, omega, H_sys, alpha, mixed=mixed)
          
            qml.Barrier(wires=range(N), only_visual=True)
        
        # print("\nsystem density matrix (N=", N, "):\n", sep='')
    
        return qml.density_matrix(wires=range(num_system_qubits))
    
    return circuit


############### helper function for circuit diagram ###########
def get_evolutiontime(op:"qml.Hamiltonian"):
    if isinstance(op, qml.ApproxTimeEvolution):
        evo_time = float(op.parameters[0])
        print(evo_time)
    elif isinstance(op, qml.ops.op_math.Evolution):
        evo_time = np.abs(op.coeff)
    else:
        evo_time=None
    return evo_time

###################### display circuit nicely ###################
def create_circuit_diagram(qc_name:str, path, file_name:str = "", *, num_timesteps:int, tau:float, mixed:bool):
    from matplotlib.text import Text
    mpl.rcParams["mathtext.fontset"] = "stix" # use a font that doesnt throw warnings
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["font.size"] = 20   # or 20, 22, ...
        
    qscript = qml.tape.make_qscript(qc_name.func)()
    all_ops = list(qscript.operations)
    N = len(qc_name.device.wires) - 1  

    evolv_ops = {"W_tau": [], "W_tau_half": [], "H_env": [], "H_sys": []}
    for op in all_ops:
        op_condition = ("Evolution" in op.name) or ("ApproxTimeEvolution" in op.name) or ("RZ" in op.name) or ("X" in op.name)
        #print(op.name, list(op.wires), getattr(op, "coeff", None))
        if op_condition:
            if [N] == list(op.wires):
                evolv_ops["H_env"].append(op)
            elif N not in list(op.wires):
                evolv_ops["H_sys"].append(op)
            elif np.isclose(abs(get_evolutiontime(op)), tau):
                evolv_ops["W_tau"].append(op)
            else:
                evolv_ops["W_tau_half"].append(op)


    fig, ax = qml.draw_mpl(qc_name)()

    op_texts = [artist for artist in fig.findobj(Text) if ("Exp(" in artist.get_text()) or ("Approx" in artist.get_text()) or ("RZ" in artist.get_text())]
    evolve_ops_in_order = [op for op in all_ops if ("Evolution" in op.name) or ("ApproxTimeEvolution" in op.name) or ("RZ" in op.name) or ("X" in op.name) ]

    for artist, op in zip(op_texts, evolve_ops_in_order):
        if op in evolv_ops["H_env"]:
            artist.set_text(r"$e^{i\omega\tau Z/2}$")
        elif op in evolv_ops["W_tau_half"]:
            artist.set_text(r"$W(\tau/2)$")
        elif op in evolv_ops["W_tau"]:
            artist.set_text(r"$W(\tau)$")
        else:
            artist.set_text(r"$e^{-i\tau H_{\mathrm{sys}}}$")

    # replace the reset labels  
    all_texts = fig.findobj(Text)
    for artist in all_texts:
        txt = artist.get_text().strip()
        if (not mixed and txt == "X") or (mixed and txt == "QubitChannel"):
            artist.set_text(r"$\rho_E$")
            artist.set_fontsize(20) 

    file_name = "cooling_circuit_"+str(num_timesteps)+"_steps.png" if (file_name == "") else file_name 

    fig.savefig(path+file_name, dpi=200)
    plt.close(fig)

def main():
    # generic parameters for testing
    output = True
    N = 3
    T = 1
    alpha = 1
    sigma = 1
    omega_max = 5
    beta = 10 
    tau = 1/2
    
    assert tau < T

    op_set = construct_opset(N, type="XZ")
    J, h = -1, 1
    H_sys  = transverse_ising_hamiltonian(J, h, N)

    seed = 123
    mixed = True
    circuit_diagram = True
    
    print("\n" + 80 * "-")
    print("Start calculation")
    print(80 * "-")

    print("\nTransversal Ising model on", N,"qubits with J=", J, ", h=", h, ":\n")
    print("H_sys =", H_sys, "\n")

    s = input("Enter number of timesteps [10]: ")
    num_timesteps = int(s) if s else 10

    circuit = create_cooling_circuit(
        num_system_qubits=N,
        num_timesteps=num_timesteps,
        H_sys=H_sys,
        tau=tau,
        beta=beta,
        T=T,
        omega_max=omega_max,
        op_set=op_set,
        alpha=alpha,
        sigma=sigma,
        seed=seed,
        mixed=mixed,
        output=output)
    
            
    final_state = circuit()
    if output:
        print(np.round(final_state, 3))
        print("\ntrace =", np.round(np.real(np.trace(final_state)),3))
        is_diagonal = np.allclose(final_state, np.diag(np.diagonal(final_state)))
        print("is diagonal:", is_diagonal)

    if circuit_diagram:
        local_path = Path(__file__).resolve().parent
        create_circuit_diagram(circuit, local_path, num_timesteps=num_timesteps, tau=tau, mixed=mixed)
    
if __name__ == "__main__":
    main()