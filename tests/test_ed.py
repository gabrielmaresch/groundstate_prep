import numpy as np


local_path = "/Users/gab/Library/Mobile Documents/com~apple~CloudDocs/QuIST/Projekt-Praktikum/codebase/groundstate_prep/"
file_name = "ed.jl" 

def test_energies():
    from juliacall import Main as jl
    jl.include(local_path+file_name)
    
    
    #no transverse field
    _, energy = jl.get_transverse_ising_groundstate(3, 1, 0)
    assert np.isclose(energy, -0.75)

    #comparable transverse field:
    _, energy = jl.get_transverse_ising_groundstate(3, 1, 1)
    assert np.isclose(energy, -0.75-np.sqrt(3)/2)

    #strong transverse field:
    _, energy = jl.get_transverse_ising_groundstate(3, 1, 10)
    assert np.isclose(np.round(energy,2), -15.01)


    #only transverse field:
    _, energy = jl.get_transverse_ising_groundstate(3, 0, 1)
    assert np.isclose(energy, -1.5)
