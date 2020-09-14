# YieldEstOptGPR

- This is an algorithm for the efficient and reliable estimation of a yield (= percentage of accepted realizations in a manufacturing process under uncertainties). 

- As benchmark problems a simple dielectrical waveguide and a lowpass filter are considered.

- For yield estimation a hybrid method combining pure Monte Carlo (MC) with a surrogate model approach based on Gaussian process regression (GPR) is used.

- The main files to run the yield estimation are Run_YieldEst_Waveguide.py (for the waveguide problem) and Run_YieldEst_Lowpass.py (for the lowpass filter problem, respectively).

- A detailed description of the method and numercial results can be found in arXiv:2003.13278.

- The model of the waveguide comes from... and is evaluated in Eval_Waveguide.py.

- The model of the Lowpass Filter comes from the CST Studio Suite Examples Library and is evaluated using the Frequency Domain Solver of CST. The results for the S-Parameter are given in Lowpass_data_QoI.py, for the sample in Lowpass_data_sample.py. Eval_Lowpass.py accesses the lists.
