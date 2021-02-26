# Yield Estimation and Optimization with Gaussian Process Regression (YieldEstOptGPR)

This repository contains the main source code and data of the yield estimation and optimization procedures documented in the following papers 

@article{FuhrlanderSchops2020,
  title={A blackbox yield estimation workflow with Gaussian process regression applied to the design of electromagnetic devices},
  author={Fuhrländer, Mona and Schöps, Sebastian},
  journal={Journal of Mathematics in Industry},
  volume={10},
  number={1},
  pages={1--17},
  year={2020},
  publisher={Springer},
  url={https://doi.org/10.1186/s13362-020-00093-1}
}

and

@article{FuhrlanderSchops2021,
  title={Yield Optimization using Hybrid Gaussian Process Regression and a Genetic Multi-Objective Approach},
  author={Fuhrländer, Mona and Schöps, Sebastian},
  journal={Advances in Radio Science},
  publisher={Copernicus GmbH},
  pubstate={forthcoming},
  year={2021},
  url={https://arxiv.org/abs/2010.04028}
}

## Content

- This is an algorithm for the efficient and reliable estimation of a yield (= percentage of accepted realizations in a manufacturing process under uncertainties).

- For yield estimation a hybrid method combining pure Monte Carlo (MC) with a surrogate model approach based on Gaussian process regression (GPR) is used.

- For yield optimization an adaptive Newton-MC method is used, which is a modification of a globalized Newton method allowing adaptive sample size increase.

- As benchmark problems a simple dielectrical waveguide and a lowpass filter (only for estimation) are considered.

## Running the examples

- The main files to run the yield estimation are Run_YieldEst_Waveguide.py (for the waveguide problem) and Run_YieldEst_Lowpass.py (for the lowpass filter problem, respectively).

- The main file to run the yield optimization is Run_YieldOpt_Waveguide.py.

## Data origin

- The model of the waveguide comes from Dimitrios Loukrezis (https://github.com/dlouk/UQ_benchmark_models/tree/master/rectangular_waveguides) and is evaluated in Eval_Waveguide.py.

- The model of the Lowpass Filter comes from the CST Studio Suite Examples Library (https://www.3ds.com/de/produkte-und-services/simulia/produkte/cst-studio-suite/) and is evaluated using the Frequency Domain Solver of CST. The results for the S-Parameter are given in Lowpass_data_QoI.py, for the sample in Lowpass_data_sample.py. Eval_Lowpass.py accesses the lists.

## Licence

This project is licensed under the terms of the GNU General Public License (GPL).
