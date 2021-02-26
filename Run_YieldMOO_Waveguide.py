# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 08:16:40 2020

@author: Mona Fuhrländer (Technische Universität Darmstadt, mona.fuhrlaender@tu-darmstadt.de)

Genetic Multi-objective optimization based on Pymoo (see https://pymoo.org/) 
applied to simple dielectrical waveguide with 4 uncertain parameters
- maximization of the yield
- robust minimization of the width of the waveguide
"""


import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Import Problem Formulation
from Opt_ProbDefMOO_Waveguide import MyProblem



# =============================================================================
#     #Set the problem
# =============================================================================
problem = MyProblem()


# =============================================================================
#     #Set the algorithm and choose the settings
# =============================================================================
#https://pymoo.org/algorithms/index.html
algorithm = NSGA2(
    pop_size=200, #100,#40, # initial population size
    n_offsprings=100,#50,#10, # new individuals per generation
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)


# =============================================================================
#     #Define termination criteria
# =============================================================================
#https://pymoo.org/interface/termination.html
termination = MultiObjectiveDefaultTermination(
    x_tol=1e-8,
    cv_tol=1e-6,
    f_tol=0.0025,
    nth_gen=5,
    n_last=30,
    n_max_gen=30, # maximal number of generations
    n_max_evals=100000
)


# =============================================================================
#     #Solve the problem (Optimization)
# =============================================================================
#https://pymoo.org/interface/result.html
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               #pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True
               )


# =============================================================================
#     #Visualization in design and objective space
# =============================================================================
#https://pymoo.org/visualization/index.html
# Design Space
plot = Scatter(title = "Design Space")
plot.add(res.X, s=30, facecolors='none', edgecolors='r')
plot.do()
plot.show()

# Objective Space
plot = Scatter(title = "Objective Space")
plot.add(res.F)
plot.show()


# =============================================================================
#     #Output lists for solution, objective functions and constraint
# =============================================================================
Output = np.array([res.X[:,0],res.X[:,1],res.X[:,2],res.X[:,3],res.F[:,0],res.F[:,1],res.G[:,0],res.CV[:,0]]).T
Output_posYield = np.array([res.X[:,0],res.X[:,1],res.X[:,2],res.X[:,3],res.F[:,0],-res.F[:,1],res.G[:,0],res.CV[:,0]]).T
print('\n Output: x1, x2, x3, x4, f1 (size), -f2 (Yield), g1 (Yield>Y_min), violation g1')
np.savetxt('MOOsolutionA.csv', Output, delimiter=',',header='x1,x2,x3,x4,f1,f2,g1,diff')
np.savetxt('MOOsolutionB.csv', Output_posYield, delimiter=',',header='x1,x2,x3,x4,f1,-f2,g1,diff')

print(Output_posYield)