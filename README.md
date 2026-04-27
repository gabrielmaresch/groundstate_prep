# groundstate_prep
Implementation of ground resp. thermal state preparation algorithm following the quantum-channel approach presented in arXiv:2508.05703 from Ding et.al.
Project for QIST master program at TU Wien, supervised by S. Andergassen and T. Ayral.


### Milestones
1.) Implementation of cooling algorithm for transverse Ising model as toy example

2.) Implementation of cooling algorithm for 1+2 (impurity/bath) site DMFT model.

### Questions:
- How does cooling work if run on a NISQ QC or with noise simulation?
- Can we assign the noise an effective temperature and what T is necessary to be able to detect a particular phase-transition (e.g. superconducting phase)?
- What computational budget is necessary to do so?
- Ressource estimation for impurity model
- Can a hybrid algorithm (first cooling, then filtering) compute a Green's function for the impurity model effectively?
- What is the optimal ressource allocation between cooling and filtering-algorithm?
