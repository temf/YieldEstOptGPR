# Yield Estimation and Optimization with Gaussian Process Regression (YieldEstOptGPR)

This repository contains the main source code and data of the yield estimation procedure documented in the paper "A blackbox yield estimation workflow with Gaussian process regression applied to the design of electromagnetic devices", see https://arxiv.org/abs/2003.13278, or

@article{FuhrlanderSchops2020,
	author = {Fuhrländer, Mona and Schöps, Sebastian},
	journal = {Journal of Mathematics in Industry},
	year = {2020},
	pubstate = {forthcoming},
	title = {A Blackbox Yield Estimation Workflow with {Gaussian} Process Regression Applied to the Design of Electromagnetic Devices},
}


## Content

- This is an algorithm for the efficient and reliable estimation of a yield (= percentage of accepted realizations in a manufacturing process under uncertainties).

- For yield estimation a hybrid method combining pure Monte Carlo (MC) with a surrogate model approach based on Gaussian process regression (GPR) is used.

- As benchmark problems a simple dielectrical waveguide and a lowpass filter are considered.


## Running the examples

- The main files to run the yield estimation are Run_YieldEst_Waveguide.py (for the waveguide problem) and Run_YieldEst_Lowpass.py (for the lowpass filter problem, respectively).


## Data origin

- The model of the waveguide comes from Dimitrios Loukrezis (https://github.com/dlouk/UQ_benchmark_models/tree/master/rectangular_waveguides) and is evaluated in Eval_Waveguide.py.

- The model of the Lowpass Filter comes from the CST Studio Suite Examples Library (https://www.3ds.com/de/produkte-und-services/simulia/produkte/cst-studio-suite/) and is evaluated using the Frequency Domain Solver of CST. The results for the S-Parameter are given in Lowpass_data_QoI.py, for the sample in Lowpass_data_sample.py. Eval_Lowpass.py accesses the lists.

## Licence

This project is licensed under the terms of the GNU General Public License (GPL).
