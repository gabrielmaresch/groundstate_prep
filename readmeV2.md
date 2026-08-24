# Ground-State Preparation via Quantum Channels

Research implementation of quantum-channel algorithms for ground- and
thermal-state preparation. The repository develops and evaluates cooling-based
protocols for small quantum many-body systems, using exact diagonalization and
explicit channel construction as reference tools.

This is a project for the QIST master's programme at TU Wien, supervised by
S. Andergassen and T. Ayral.

## Scope

The current implementation focuses on the transverse-field Ising model (TFIM).
It constructs an ancilla-assisted cooling circuit, averages the resulting
channels over interaction operators and frequencies, and studies convergence
towards a Gibbs state through trajectory and spectral diagnostics.

The code is intended for exploratory research on small systems. It is not a
general-purpose quantum-simulation library.

## Installation

The Python code requires Python 3.11+ and the dependencies listed in
[`requirements.txt`](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some validation and reference calculations use Julia through `juliacall`.
From the repository root, instantiate the Julia environment with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick check

Run the exact-diagonalization regression test:

```bash
pytest tests/test_ed.py
```

## Main workflows

### Cooling trajectories

`path_analysis.py` provides `run_TFIM(...)` for simulating the cooling circuit
on a small TFIM system. The routine compares the output state with the exact
Gibbs state using trace distance and can generate convergence plots.

The circuit construction itself lives in `cooling_channel.py`:

- `transverse_ising_hamiltonian(...)` builds the TFIM Hamiltonian.
- `construct_opset(...)` defines the sampled system--ancilla interactions.
- `create_cooling_circuit(...)` returns a PennyLane QNode for repeated cooling
  steps.

### Averaged-channel analysis

`superoperator.py` constructs the averaged channel and provides spectral and
fixed-point diagnostics. The channel can be built either from Choi elements or
Kraus blocks; sweep scripts select the appropriate implementation for their
use case.

For a sweep over the transverse field, run for example:

```bash
python parallel_h_sweep.py --N 4 --h-min 0.1 --h-max 2.0 --h-points 20
```

For a grid over `h`, `alpha`, `sigma`, and `omega_max`:

```bash
python grid.py --N 4 --h-points 10 --alpha-points 6 --sigma-points 5 --omega-points 2
```

Results are saved as `.npz` files under `data/` by default. Use `--data-dir`
to choose another output directory. The notebooks `sweep_analysis.ipynb` and
`grid_analysis.ipynb` inspect the resulting spectral gaps, trace distances,
mixing-time estimates, and parameter dependence.

> **Scaling note:** a dense channel matrix has dimension
> \(4^N \times 4^N\). Dense workflows are therefore practical only for small
> `N`. Scripts protect against saving very large dense matrices with
> `--save-channel`.

## Preliminary results

For the explicitly constructed averaged channel of a normalized four-qubit
TFIM, the tested cooling trajectories converge towards the channel fixed point
across the sampled field range. The iterations needed to reach trace distance
below `0.05` from that fixed point generally decrease in the paramagnetic
regime.

The experiments show an accuracy--speed trade-off in the coupling `alpha`:
weaker coupling yields fixed points closer to the exact Gibbs state but takes
more iterations. In the reported `N=4`, `beta=1` sweep, `alpha=0.25` and
`0.5` produced fixed points within `0.1` trace distance of the Gibbs state;
stronger coupling converged faster but with larger fixed-point error.

These are small-system, noiseless numerical observations. They depend on the
chosen initial state, Trotterization, and frequency quadrature, and do not yet
establish performance for larger systems or the planned impurity model.

## Repository guide

- `cooling_channel.py` — cooling-channel and circuit construction.
- `superoperator.py` — averaged quantum channels and spectral analysis.
- `ed.py`, `ed.jl` — exact diagonalization and Gibbs-state reference methods.
- `path_analysis.py` — cooling trajectories and convergence metrics.
- `parallel_h_sweep.py`, `parallel_beta_sweep.py`, `grid.py` — parameter-sweep
  entry points.
- `*_analysis.ipynb` — notebook-based data analysis.
- `data/` — generated numerical results.
- `docu/` — technical documentation and figures.

## Research direction

Planned work includes applying the method to a 1+2-site impurity/bath DMFT
model and investigating the effects of noise and resource constraints. Open
questions include effective-temperature descriptions of noise, the resources
needed to resolve phase transitions, and hybrid cooling/filtering strategies
for impurity Green's functions.

## References

- Ding et al., [arXiv:2508.05703](https://arxiv.org/abs/2508.05703)
- Cruz et al., [arXiv:2505.05411](https://arxiv.org/abs/2505.05411)

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
