# groundstate_prep
Implementation of ground- resp. thermal- state preparation algorithm following the quantum-channel approach presented in arXiv:2508.05703 from Ding et al. The project investigates cooling-based preparation methods for small quantum many-body systems, with a focus on toy models that are small enough for exact simulation and algorithmic testing.

Project for QIST master program at TU Wien, supervised by S. Andergassen and T. Ayral.

## Context

The implementation follows the quantum-channel approach to ground- and thermal-state preparation resp. the filtering algorithm described in:

- Ding et al., arXiv:2508.05703
- Cruz et al., arXiv:2505.05411



## Milestones
1.) Implementation of cooling algorithm for transverse Ising model as toy example

2.) Implementation of cooling algorithm for 1+2 (impurity/bath) site DMFT model.

## Questions:
- How does cooling work if run on a NISQ QC or with noise simulation?
- Can we assign the noise an effective temperature and what T is necessary to be able to detect a particular phase-transition (e.g. superconducting phase)?
- What computational budget is necessary to do so?
- Resource estimation for impurity model
- Can a hybrid algorithm (first cooling, then filtering) compute a Green's function for the impurity model effectively?
- What is the optimal ressource allocation between cooling and filtering algorithm?

## Current status

Work in progress.

This repository is an exploratory research implementation. It is intended to test algorithmic ideas on small model systems, not to provide a general-purpose quantum simulation package.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).