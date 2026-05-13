using LinearAlgebra
using Test
using Pkg

##### This file uses code from the QuantumSimulation course assignments ######


"""
    transverse_ising_hamiltonian(N::Int, J::Real, h::Real)

Construct the Hamiltonian matrix of the spin-1/2 transverse Ising model
for a 1D chain with periodic boundary conditions.

H = -J ∑_<i,j> Z_i Z_j - h ∑_i X_i      

# Arguments
- `N::Int`: Number of spins in the chain.
- `J::Real`: Coefficient controlling the strength of the `S^z S^z` interaction.
- `h::Real`: Strength of the external magnetic field.

# Returns
- `H::AbstractMatrix`: A `2^N × 2^N` dense matrix representing the Hamiltonian
  in the computational basis.
"""
function transverse_ising_hamiltonian(N::Int, J::Real, h::Real)
    H = zeros(Float64, 2^N, 2^N)
    X = [0.0 1.0; 1.0 0.0] 
    Z = [1.0 0.0; 0.0 -1.0]
    id2 =[1.0 0.0; 0.0 1.0]

    for i in 1:N
        # construct Sz-Sz term
        if i == N
            v_z = [(j==N || j==1 ? Z : id2)  for j in 1:N]
        # periodic boundary condition (not elegant, but easily readable)
        else
            v_z = [(j==i || j==i+1 ? Z : id2)  for j in 1:N]
        end
        H += -J.*kron(v_z...)
        
        # construct Sx term
        v_x = [(j==i ? X : id2)  for j in 1:N]
        H += -h.*kron(v_x...)
    end
    return Hermitian(H)
end

function thermal_state(H::AbstractMatrix, beta::Real)
    eig = eigen(H)
    N, M = size(H)
    @assert N==M

    evals, V = eig.values, eig.vectors
    #could be shifted by GS energy if necessary (large beta)
    beta_exps = [exp(-beta*E) for E in evals]
    Z = sum(p for p in beta_exps)
    D = Diagonal(beta_exps./Z)
    energy = sum(evals[i]*beta_exps[i] for i in 1:N)/Z
    gibbs =  V * D * V'
    return gibbs, energy
end

function thermal_expectation_value(H::AbstractMatrix, A::AbstractMatrix, beta::Real)
    eig = eigen(H)
    N, M = size(H)
    @assert N==M

    evals, V = eig.values, eig.vectors
    #could be shifted by GS energy if necessary (large beta)
    beta_exps = [exp(-beta*E) for E in evals]
    Z = sum(p for p in beta_exps)
    
    expectation = 0

    for i in 1:N
        i_ket = V[:,i]
        i_bra = i_ket'
        matrix_element = i_bra * A * i_ket 
        expectation += beta_exps[i]/Z * matrix_element
    end
    return expectation
end

function ground_state(H::AbstractMatrix)
    eig = eigen(H)
    N, M = size(H)
    @assert N==M

    gs_energy, gs = eig.values[1], eig.vectors[:,1]
    return gs, gs_energy
end


function ground_state_expectation_value(H::AbstractMatrix, A::AbstractMatrix)
    eig = eigen(H)
    N, M = size(H)
    @assert N==M

    gs = eig.vectors[:,1]
    return gs'*A*gs
end

function get_transverse_ising_groundstate(N::Int, J::Real, h::Real)
    H = transverse_ising_hamiltonian(N, J, h)
    gs, gs_energy = ground_state(H)
    return gs, gs_energy
end

function get_transverse_ising_gibbsstate(N::Int, J::Real, h::Real, beta::Real)
    H = transverse_ising_hamiltonian(N, J, h)
    gibbs, gibbs_energy = thermal_state(H, beta)
    return gibbs, gibbs_energy
end

function get_spectrum(H::AbstractMatrix)
    return eigen(H).values
end