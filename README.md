# SGAN-ME

This is a python implementation of the SGAN-ME approach presented in "Using deep generative neural networks to account for model errors in Markov chain Monte Carlo inversion" (submitted to geophysical journal international and currently under review). The folder includes the SGAN network training (*SGAN* folder) and MCMC inversion (*DREAM(ZS)* folder). The *DREAM(ZS)* folder includes two subsurface-model examples (in the paper referred as 1 and 2) for a model error between a finite difference time domain and straight-ray solvers in a ground-penetrating-radar cross-borehole numerical experiment.

## Scirpts:

**SGAN**
- train_SGAN.py
- utils.py
- nnmodels.py
- MSN.py
- torchsummary.py

**DREAM(ZS)**
- run_mcmc.py
- mcmc.py
- mcmc_func.py
- generator.py
- gen_from_z.py
- tomokernel_straight.py

## Citation :

Levy S., Hunziker J., Laloy E., Irving J., Linde N. under review. Using deep generative neural networks to account for model errors in Markov chain Monte Carlo inversion [submitted to Geophysical Journal International on June 2021].

## License:

See particular license specifications in folders.

## Contact:

Shiran Levy (shiran.levy@unil.ch)
