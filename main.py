import numpy as np
from algorithms import *
from cost_functions import *

if __name__ == '__main__':
    
    #____________PROBLEM A________________________

    print('\n---- TESTING PROBLEM A ----\n')
    
    x0 = np.zeros((6,1))

    # test steepest descent
    x, minimum =  steepest_descent(x0,
                                V_a,
                                gradV_a,
                                step_size = 1e-4,
                                threshold = 1e-8, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Steepest descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test conjugate gradient descent
    x, minimum =  conjugate_gradient(x0,
                                V_a,
                                gradV_a,
                                step_size = 1e-4,
                                threshold = 1e-8, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Conjugate descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test secant function
    x, minimum =  secant(x0,
                        V_a,
                        gradV_a,
                        step_size = 1e-4,
                        threshold = 1e-6, 
                        log = False, 
                        h = 1e-8, 
                        max_iter = 1e7, 
                        fd_method = 'central', 
                        track_history = False)
    print(f'Secant Method:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________PROBLEM B________________________

    print('\n---- TESTING PROBLEM B ----\n')
    x0 = np.ones((2,1))

    # test steepest descent
    x, minimum =  steepest_descent(x0,
                                V_b,
                                gradV_b,
                                step_size = 1e-4,
                                threshold = 1e-8, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Steepest descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test conjugate gradient descent
    x, minimum =  conjugate_gradient(x0,
                                V_b,
                                gradV_b,
                                step_size = 1e-4,
                                threshold = 1e-8, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Conjugate descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test secant function
    x, minimum =  secant(x0,
                        V_b,
                        gradV_b,
                        step_size = 1e-4,
                        threshold = 1e-4, 
                        log = False, 
                        h = 1e-8, 
                        max_iter = 1e7, 
                        fd_method = 'central', 
                        track_history = False)
    print(f'Secant Method:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________PROBLEM C________________________

    print('\n---- TESTING PROBLEM C ----\n')
    x0 = np.zeros((2,1))

    # test steepest descent
    x, minimum =  steepest_descent(x0,
                                V_c,
                                gradV_c,
                                step_size = 1e-4,
                                threshold = 1e-6, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Steepest descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test conjugate gradient descent
    x, minimum =  conjugate_gradient(x0,
                                V_c,
                                gradV_c,
                                step_size = 1e-4,
                                threshold = 1e-6, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Conjugate descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test secant function
    x, minimum =  secant(x0,
                        V_c,
                        gradV_c,
                        step_size = 1e-4,
                        threshold = 1e-4, 
                        log = False, 
                        h = 1e-8, 
                        max_iter = 1e7, 
                        fd_method = 'central', 
                        track_history = False)
    print(f'Secant Method:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________CONSTRAINED PROBLEM 1________________________

    print('\n---- TESTING CONSTRAINED PROBLEM 1 ----\n')
    x0 = np.array([[0.1],[0.7]])

    # test penalty function
    x, minimum =  penalty_fn(x0,
                                V_1,
                                gradient_function=None,
                                ecp=h2_1,
                                icp=[h1_1],
                                step_size = 1e-4,
                                threshold = 1e-3, 
                                conv_threshold=1e-3,
                                log = False, 
                                h = 1e-5, 
                                max_iter = 1e5, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Penalty function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[0.6],[0.6]])

    # test barrier function
    x, minimum =  barrier_fn(x0,
                                V_1,
                                gradient_function=None,
                                mode='inv',
                                ecp=h2_1,
                                icp=[h1_1],
                                step_size = 1e-4,
                                threshold = 1e-3,
                                conv_threshold = 1e-6,  
                                log = False, 
                                h = 1e-5, 
                                max_iter = 1e5, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Barrier function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[0.1],[0.7]])
    
    # test augmented lagrangian
    x, minimum = augmented_lagrangian(x0,
                                        V_1, 
                                        [h2_1], 
                                        [h1_1],
                                        log = False,
                                        track_history = False)
    print(f'Augmented Lagrangian function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[0.1],[0.7]])
    # test lagrange-newton
    x, minimum = lagrange_newton(x0,
                                V_1, 
                                equality_constraints = [h2_1], 
                                inequality_constraints = [h1_1],
                                log = False,
                                track_history = False)
    print(f'Lagrange-Newton function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________CONSTRAINED PROBLEM 2________________________

    print('\n---- TESTING CONSTRAINED PROBLEM 2 ----\n')
    x0 = np.array([[1.],[-1.]])

    # test penalty function
    x, minimum =  penalty_fn(x0,
                                V_2,
                                gradient_function=None,
                                ecp=None,
                                icp=[h1_2,h2_2],
                                step_size = 1e-4,
                                threshold = 1e-3,
                                conv_threshold=1e-3,
                                log = False, 
                                h = 1e-7, 
                                max_iter = 1e5, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Penalty function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[0.5],[0.5]])

    # test barrier function
    x, minimum =  barrier_fn(x0,
                                V_2,
                                gradient_function=None,
                                mode='inv',
                                ecp=None,
                                icp=[h1_2,h2_2],
                                step_size = 1e-4,
                                threshold = 1e-7,
                                conv_threshold = 1e-6, 
                                log = False, 
                                h = 1e-5, 
                                max_iter = 1e5, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Barrier function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[0.6],[0.6]])
    # test augmented lagrangian
    x, minimum = augmented_lagrangian(x0,
                                        V_2, 
                                        [], 
                                        [h1_2, h2_2],
                                        log = False,
                                        track_history = False)
    print(f'Augmented Lagrangian function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[1.], [-0.9]])
    # test lagrange-newton
    x, minimum = lagrange_newton(x0,
                                V_2, 
                                equality_constraints = [], 
                                inequality_constraints = [h1_2, h2_2],
                                log = False,
                                track_history = False)
    print(f'Lagrange-Newton function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________CONSTRAINED PROBLEM 3________________________

    print('\n---- TESTING CONSTRAINED PROBLEM 3 ----\n')
    x0 = np.array([[4.],[2.]])

    #test penalty function
    x, _ =  penalty_fn(x0,
                        V_3,
                        gradient_function=None,
                        ecp=h2_3,
                        icp=[h1_3],
                        sigma_max=1e3,
                        step_size = 1e-4,
                        threshold = 1e-3,
                        conv_threshold=1e-3, 
                        log = False, 
                        h = 1e-7, 
                        max_iter = 1e5, 
                        fd_method = 'forward', 
                        track_history = False)
    V = np.log(x[0]) - x[1]
    print(f'Penalty function:\n x={x.flatten()}, V(x)={V.item():.5f}\n')

    x0 = np.array([[2.],[2.]])

    # test barrier function
    x, _ =  barrier_fn(x0,
                        V_3,
                        gradient_function=None,
                        ecp=h2_3,
                        icp=[h1_3],
                        step_size = 1e-4,
                        threshold = 1e-3,
                        conv_threshold=1e-6,
                        log = False, 
                        h = 1e-7, 
                        max_iter = 1e5, 
                        fd_method = 'forward', 
                        track_history = False)
    V = np.log(x[0]) - x[1]
    print(f'Barrier function:\n x={x.flatten()}, V(x)={V.item():.5f}\n')

    x0 = np.array([[0.1],[0.7]])
    # test augmented lagrangian
    x, minimum = augmented_lagrangian(x0,
                                        V_3, 
                                        [h2_3], 
                                        [h1_3],
                                        log = False,
                                        track_history = False)
    print(f'Augmented Lagrangian function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[2.], [1.5]])
    # test lagrange-newton
    x, minimum = lagrange_newton(x0,
                                V_3, 
                                equality_constraints = [h2_3], 
                                inequality_constraints = [h1_3],
                                log = False,
                                track_history = False)
    print(f'Lagrange-Newton function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')