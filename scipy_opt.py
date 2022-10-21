import numpy as np
from scipy.optimize import minimize
from cost_functions import *

# Unconstrained Problem A
x0 = np.zeros((6,1))
res_a = minimize(V_a, x0)
minimum = V_a(res_a.x).item()
print(f'Unconstrained A:\n x={res_a.x}, V(x)={minimum:.5f}\n')

# Unconstrained Problem B
x0 = np.ones((2,1))
res_b = minimize(V_b, x0)
minimum = V_b(res_b.x).item()
print(f'Unconstrained B:\n x={res_b.x}, V(x)={minimum:.5f}\n')

# Unconstrained Problem C
x0 = np.ones((2,1))
res_c = minimize(V_c, x0, method='CG')
minimum = V_c(res_c.x).item()
print(f'Unconstrained C:\n x={res_c.x}, V(x)={minimum:.5f}\n')

# Constrained Problem 1
x0 = np.array([[0.1],[0.7]])
ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([x[0]-x[1]**2]),
             'jac' : lambda x: np.array([[1.0, -2.0*x[1]]])}
eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([x[0]**2 + x[1]**2 - 1]),
           'jac' : lambda x: np.array([2.0*x[0], 2.0*x[1]])}
res_1 = minimize(V_1, x0, method='SLSQP', constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9})
minimum = V_1(res_1.x).item()
print(f'Constrained 1:\n x={res_1.x}, V(x)={minimum:.5f}\n')

# Constrained Problem 2
x0 = np.array([[0.6],[0.6]])
ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([-x[0]-x[1]**2 + 1,
                                          x[0] + x[1]]),
             'jac' : lambda x: np.array([[-1.0, -2.0*x[1]], 
                                          [1.0, 1.0]])}
res_2 = minimize(V_2, x0, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-9})
minimum = V_2(res_2.x).item()
print(f'Constrained 2:\n x={res_2.x}, V(x)={minimum:.5f}\n')

# Constrained Problem 3
x0 = np.array([[4.],[2.]])
ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([x[0]-1]),
             'jac' : lambda x: np.array([[1.0, 0.0]])}
eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([x[0]**2 + x[1]**2 - 4]),
           'jac' : lambda x: np.array([2.0*x[0], 2.0*x[1]])}
res_3 = minimize(lambda x: np.log(x[0])-x[1], x0, method='SLSQP', constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9})
minimum = V_3(res_3.x).item()
print(f'Constrained 3:\n x={res_3.x}, V(x)={minimum:.5f}\n')
