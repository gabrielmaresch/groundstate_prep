import pennylane.numpy as np
from pathlib import Path



local_path = Path(__file__).resolve().parent.parent
file_name = "ed.jl" 

def test_energies():
    from juliacall import Main as jl
    jl.include(local_path / file_name)
    
    
    #no transverse field
    _, energy = jl.get_transverse_ising_groundstate(3, 1, 0)
    assert np.isclose(energy, -3.0)

    #comparable transverse field:
    _, energy = jl.get_transverse_ising_groundstate(3, 1, 1)
    assert np.isclose(energy, -4.0)

    #strong transverse field:
    _, energy = jl.get_transverse_ising_groundstate(3, 1, 10)
    assert np.isclose(np.round(energy,2), -30.08)


    #only transverse field:
    _, energy = jl.get_transverse_ising_groundstate(3, 0, 1)
    assert np.isclose(energy, -3.0)
