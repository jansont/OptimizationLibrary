from multiprocessing.sharedctypes import Value
from tkinter import NONE
import numpy as np
from numpy.linalg import norm, eig
from functools import partial



def steepest_descent(x0,
                     cost_function,
                     gradient_function = None,
                     step_size = 'armijo',
                     threshold = 1e-4, 
                     log = False, 
                     h = 1e-8, 
                     max_iter = 1e6, 
                     gamma = 1.5, 
                     r = 0.8, 
                     fd_method = 'central', 
                     track_history = False):
    '''
    Performs vanilla gradient descent. 
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used.
            Default is None. 
        step_size :: float or String
            Step size to use during gradient descent. 
            If 'armijo', Armijo step size selection is used. 
            Default: 'armijo'
        threshold :: float
            Threshold at which to stop minimization. Values 
            should be close to 0. Default: 1e-4
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        max_iter :: int
            Maximum optimization iterations. Default: 1e6
        gamma :: float
            Gamma parameter for armijo. Default is 1.5. 
        r :: float
            r parameter for armijo. Default is 0.8. 
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    -----------------------------------------------------------
    Args examples: 
    x0 = np.zeros((2,1))
    cost_function = lambda x: x[0]**2 + x[1]**2
    '''

    #if no gradient function available, use finite difference appx
    if gradient_function == None: 
        fd = Finite_Difference(cost_function, fd_method, h)
        gradient_function = fd.estimate_gradient

    #initialize iterator, x, and gradient
    i = 0
    x = x0
    gradient = gradient_function(x)
    minimum = cost_function(x)

    x_history, V_history = [x0],[minimum]
    #iterate until near zero gradient or max iterations reached
    while norm(gradient) >= threshold and i <= max_iter: 

        #update gradient
        gradient = gradient_function(x)
        search_dir = -1*gradient

    #determine step size
        if step_size == 'armijo':                      
            w = armijo(x, cost_function, gradient, search_dir=search_dir, gamma = gamma, r = r, log=False)
        elif isinstance(step_size, (int, float)):
            w = step_size
        else: 
            raise TypeError('step size should be float, int or "armijo"') 
        
        # move to a new x by moving from the original x in the negative
        # direction of the gradient according to a given step size
        x = x + w*search_dir
        minimum = cost_function(x).item()

        #result tracking
        i += 1
        if log and i % 1e4 == 0: 
            print(f'x = {x}, V(x) = {minimum:.5f}')
        if track_history: 
            x_history.append(x), V_history.append(minimum)

    if track_history:
        return x, minimum, x_history, V_history
    else: 
        return x, minimum





def conjugate_gradient(x0,
                     cost_function,
                     gradient_function = None,
                     step_size = 'armijo',
                     threshold = 1e-8, 
                     log = False, 
                     h = 1e-8, 
                     max_iter = 1e6, 
                     gamma = 1.5, 
                     r = 0.8, 
                     fd_method = 'central', 
                     track_history = False):
    '''
    Performs conjugate gradient descent. 
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used. 
        step_size :: float or String
            Step size to use during gradient descent. 
            If 'armijo', Armijo step size selection is used. 
            Default: 'armijo'
        threshold :: float
            Threshold at which to stop minimization. Values 
            should be close to 0. Default: 1e-8
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        gamma :: float
            Gamma parameter for armijo. Default is 1.5. 
        r :: float
            r parameter for armijo. Default is 0.8. 
        max_iter :: int
            Maximum optimization iterations. Default: 1e6
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    -----------------------------------------------------------
    Args examples: 
    x0 = np.zeros((2,1))
    cost_function = lambda x: x[0]**2 + x[1]**2

    '''
    #if no gradient function available, use finite difference appx
    if gradient_function == None: 
        fd = Finite_Difference(cost_function, fd_method, h)
        gradient_function = fd.estimate_gradient

    minimum = cost_function(x0)
    x_history, V_history = [x0],[minimum]
    i = 0
    prev_gradient = gradient_function(x0)
    search_direction = prev_gradient * -1
    while norm(prev_gradient) >= threshold and i <= max_iter: 

        #determine step size
        if step_size == 'armijo':
            w = armijo(x0,
                    cost_function,
                    prev_gradient,
                    search_dir=search_direction,
                    gamma = gamma,
                    r = r,
                    log=False)

        elif isinstance(step_size, (int, float)):
            w = step_size
        else: 
            raise TypeError('step size should be float, int or "armijo"')

        #conjugate_gradient_algorithm
        x1 = x0 + w * search_direction
        next_gradient = gradient_function(x1)
        beta = (next_gradient - prev_gradient).T @  next_gradient
        beta /= prev_gradient.T @ prev_gradient
        search_direction = -1*next_gradient + beta * search_direction
        prev_gradient = next_gradient
        x0 = x1
        minimum = cost_function(x0).item()
        i+=1
        #track results
        if log and i%1e4 == 0: 
            print(f'x = {x0}, V(x) = {minimum:.5f}')
        if track_history: 
            x_history.append(x0), V_history.append(minimum)

    if track_history:
        return x0, minimum, x_history, V_history
    else: 
        return x0, minimum




class Finite_Difference:
    def __init__(self, function, method = 'forward', h = 1e-8):
        '''
        Args: 
            function: cost function Rn -> R
            h: Default is 1e-8. 
        '''
        self.function = function
        self.h = h
        self.method = method

    def central_difference(self, x):
        '''
        Performs central difference estimate of the gradient. 
        Args: 
            x: np.array
                Point at which to estimate derivative. Shape (n,)
        Returns: 
            gradient: np.array
                Gradient estimate at x. Shape (n,)
        '''
        gradient = np.zeros_like(x)
        e = np.eye(len(x))
        for i in range(x.shape[0]):
            grad = self.function(x + e[i].reshape(-1,1)*self.h) - self.function(x - e[i].reshape(-1,1)*self.h)
            grad /= 2*self.h
            gradient[i] = grad
        return gradient

    def forward_difference(self, x):
        '''
        Performs forward difference estimate of the gradient. 
        Args: 
            x: np.array
                Point at which to estimate derivative. Shape (n,)
        Returns: 
            gradient: np.array
                Gradient estimate at x. Shape (n,)
        '''
        gradient = np.zeros_like(x)
        e = np.eye(len(x))
        for i in range(x.shape[0]):
            grad = self.function(x + e[i].reshape(-1,1)*self.h) - self.function(x)
            grad /= self.h
            gradient[i] = grad
        return gradient  

    def estimate_gradient(self, x):
        '''
        Select finite difference method
        '''
        if self.method == 'central':
            return self.central_difference(x)
        elif self.method == 'forward': 
            return self.forward_difference(x)
        else: 
            raise ValueError("Method must be 'central' or 'forward'")

    def hessian(self, x):
        '''
        Returns hessian matrix of self.f
        '''
        n = np.max(np.shape(x))
        I = np.eye(n)
        H = np.zeros_like(I)
        f = self.function
        h = self.h
        for i in range(n):
            for j in range(n):
                hess = f(x + h * I[:, [i]] + h * I[:, [j]]) \
                        -  f(x + h * I[:, [i]]) \
                        -  f(x + h * I[:, [j]]) \
                        + f(x)
                hess /=  h**2
                H[i, j] =  hess
        return 0.5 * (H + H.T)


def cone_condition(g, s, theta=89):
    '''
    Checks if a given search direction respects the cone condition.
    Args: 
        g :: np.array
            Gradient at a point.
        s :: np.array
            Search direction
    '''
    cos_phi = (-s.T @ g) / (np.linalg.norm(s) * np.linalg.norm(g))
    cos_theta = np.cos(theta * 2 * np.pi / 360)

    return cos_phi > cos_theta

def secant(x0,
            cost_function,
            gradient_function = None,
            H = None,
            step_size = 'armijo',
            threshold = 1e-8, 
            log = False, 
            h = 1e-8, 
            max_iter = 1e6, 
            gamma = 1.5, 
            r = 0.8,
            fd_method = 'central', 
            track_history = False):
    '''
    Performs minimization using secant algorithm with Davidson-Fletcher-Powell.  
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn. Default is None.
            If None, finite difference estimation of gradient is used. 
        H :: np.array (shape: len(x0) x len(x0))
            Estimate for C inverse. Default is None.
            if None, H = eye(len(x0)) is used. 
        step_size :: float or String
            Step size to use during gradient descent. 
            If 'armijo', Armijo step size selection is used. 
            Default: 'armijo
        threshold :: float
            Threshold at which to stop minimization. Values 
            should be close to 0. Default: 1e-8
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        gamma :: float
            Gamma parameter for armijo. Default is 1.5. 
        r :: float
            r parameter for armijo. Default is 0.8. 
        max_iter :: int
            Maximum optimization iterations. Default: 1e6
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    -----------------------------------------------------------
    Args examples: 
    x0 = np.zeros((2,1))
    cost_function = lambda x: x[0]**2 + x[1]**2
    '''

    if H == None:
        H = np.eye(len(x0))
    if H.shape[0] != H.shape[1] != len(x0):
        raise ValueError('H should be square numpy array with n = len(x0).')

    #if no gradient function available, use finite difference appx
    if gradient_function == None: 
        fd = Finite_Difference(cost_function, fd_method, h)
        gradient_function = fd.estimate_gradient

    j=0
    minimum = cost_function(x0)
    x_history, V_history = [x0],[minimum]
    while True:
        gradient_x0 = gradient_function(x0)
        s = -(H @ gradient_x0)

        if not cone_condition(gradient_x0, s):
            j = 0
            s = -gradient_function(x0)

        #determine step size
        if step_size == 'armijo':                      
            w = armijo(x0, cost_function, gradient_x0, search_dir=s, gamma = gamma, r = r, log=False)
        elif isinstance(step_size, (int, float)):
            w = step_size
        else: 
            raise TypeError('step size should be float, int or "armijo"') 
        x1 = x0 + w*s
        gradient_x1 = gradient_function(x1)
        if norm(gradient_x1) < threshold or j > max_iter:
            break

        dx = x1-x0
        dg = gradient_x1-gradient_x0
        H = H + np.matmul(dx,dx.T)/np.matmul(dx.T,dg) - np.matmul(np.matmul(H,dg),(np.matmul(H,dg)).T)/np.matmul(dg.T,np.matmul(H,dg))
        
        j += 1
        x0 = x1
        minimum = cost_function(x0).item()
        x_history.append(x0), V_history.append(minimum)

        #track results
        if log and j%1e4 == 0: 
            print(f'x = {x0}, V(x) = {minimum:.5f}')
        if track_history: 
            x_history.append(x0), V_history.append(minimum)

    if track_history:
        return x0, minimum, x_history, V_history
    else: 
        return x0, minimum




def armijo(x,
           cost_function,
           gradient,
           search_dir,
           gamma=1.5,
           r = 0.8,
           log=True):
    '''
    Determine step size using secant algorithm
    Args: 
        x :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used. 
        search_dir :: np.array (Shape (n,))
            Search direction vector. 
        gamma :: float
            Gamma parameter for armijo algorithm. Default: 1.5
        r :: 0.8
            r parameter for armijo algorithm. Default: 0.8
        log :: bool
            True to log armijo optimization progress. Default: False 
    Returns: 
        w :: float
            Optimal step size at x in direction search_dir. 
        -----------------------------------------------------------
        Args examples: 
        x0 = np.zeros((2,1))
        cost_function = lambda x: x[0]**2 + x[1]**2
        gradient_function = lambda x: [2*x[0]**2 ,  2*x[1]]
        search_dir = np.zeros((2,1))
    '''
    
    def v_bar(w):
        return cost_x + 0.5*w*grad_x_s

    w = 1
    cost_x = cost_function(x)
    grad_x_s = gradient.T @ search_dir
    # initialize p
    p = 0
    # propogate forward
    w = gamma**p
    while cost_function(x + w*search_dir) < v_bar(w): 
        w = gamma**p
        # increment p
        p += 1
    # initialize q
    q = 0
    # propogate backwards
    w = r**q * gamma**p
    while cost_function(x + w*search_dir) > v_bar(w): 
        # increment q
        q += 1
        # consider step size w
        w = r**q * gamma**p

    # return step size
    if log:
        print(f'p={p}, q={q}, w={w}')
    return w

def penalty_fn(x0,
               cost_function,
               gradient_function,
               step_size='armijo',
               ecp=None,
               icp=None,
               sigma_max=1e5,
               threshold=1e-6,
               conv_threshold=1e-3,
               log = False, 
               h = 1e-8, 
               max_iter = 1e12, 
               fd_method = 'central', 
               track_history = False):
    '''
    Performs the penalty function method for constrained optimization. 
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,1)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used.
            Default is None. 
        step_size :: float or String
            Step size to use during gradient descent. 
            If 'armijo', Armijo step size selection is used. 
            Default: 'armijo'
        ecp :: Python function
            Function of equality contraint.
        icp :: List
            List of inequality contraints as functions.
        sigma_max :: float
            Maximum value for sigma.
        threshold :: float
            Threshold at which to stop penalty function. Values 
            should be close to 0. Default: 1e-4
        conv_threshold :: float
            Threshold at which to stop steepest descent. Values 
            should be close to 0. Default: 1e-4
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        max_iter :: int
            Maximum optimization iterations. Default: 1e6
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,1)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    -----------------------------------------------------------
    Args examples: 
    x0 = np.zeros((2,1))
    cost_function = lambda x: x[0]**3-x[1]
    ecp = lambda x: x[0]**2 + x[1]**2 - 4
    icp = [lambda x: x[0]-1]
    '''
    x_hist, V_hist = [], []

    def phi(cost_function, sigma, ecp, icp, x):
        cost = cost_function(x)
        if ecp is not None:
            cost = cost + 0.5*sigma*norm(ecp(x))**2
        if icp is not None:
            for eq in icp:
                cost += 0.5*sigma*norm(np.minimum(eq(x),np.zeros_like(eq(x))))**2
        return cost
    
    def cost_norm(x):
        cost = 0
        if ecp is not None:
            cost = cost + norm(ecp(x))**2
        if icp is not None:
            for eq in icp:
                cost += norm(np.minimum(eq(x),np.zeros_like(eq(x))))**2
        return np.sqrt(cost)

    sigma = 1
    x = x0
    minimum = cost_function(x)
    x_hist.append(x)
    V_hist.append(minimum)

    while cost_norm(x) > threshold:
        # print(sigma)
        x, _ = steepest_descent(x0,
                             partial(phi, cost_function, sigma, ecp, icp),
                             gradient_function=gradient_function,
                             step_size=step_size,
                             threshold=conv_threshold,
                             log=log,
                             h=h,
                             max_iter = max_iter,
                             fd_method = fd_method,
                             track_history = track_history)
        sigma *= 10
        if sigma >= sigma_max:
            break
        x_hist.append(x)
        V_hist.append(minimum)
    if track_history:
        return x, cost_function(x).item(), x_hist, V_hist
    else: 
        return x, cost_function(x).item()

def barrier_fn(x0,
               cost_function,
               gradient_function,
               mode='inv',
               step_size='armijo',
               ecp=None,
               icp=None,
               threshold=1e-6,
               conv_threshold=1e-4,
               log = False, 
               h = 1e-8, 
               max_iter = 1e12, 
               fd_method = 'central', 
               track_history = False):
    '''
    Performs the barrier function method for constrained optimization. Equality
    contraints are handled like in the penalty function.
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,1)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used.
            Default is None.
        mode :: String
            Method to handle inequality contraints.
            If 'inv', a reciprocal operation is used. Othewise
            a log operation is used.
            Default: 'inv'
        step_size :: float or String
            Step size to use during gradient descent. 
            If 'armijo', Armijo step size selection is used. 
            Default: 'armijo'
        ecp :: Python function
            Function of equality contraint.
        icp :: List
            List of inequality contraints as functions.
        threshold :: float
            Threshold at which to stop penalty function. Values 
            should be close to 0. Default: 1e-4
        conv_threshold :: float
            Threshold at which to stop steepest descent. Values 
            should be close to 0. Default: 1e-4
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        max_iter :: int
            Maximum optimization iterations. Default: 1e6
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,1)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    -----------------------------------------------------------
    Args examples: 
    x0 = np.zeros((2,1))
    cost_function = lambda x: x[0]**3-x[1]
    ecp = lambda x: x[0]**2 + x[1]**2 - 4
    icp = [lambda x: x[0]-1]
    '''
    
    def phi(cost_function, sigma, r, mode, ecp, icp, x):
        cost = cost_function(x)
        if ecp is not None:
            cost = cost + 0.5*sigma*norm(ecp(x))**2
        if icp is not None:
            for eq in icp:
                if mode == 'inv':
                    cost += r*np.reciprocal(eq(x))
                else:
                    cost += r*np.log(eq(x))
        return cost
    
    def cost_norm(x):
        cost = 0
        if ecp is not None:
            cost = cost + norm(ecp(x))**2
        if icp is not None:
            for eq in icp:
                cost += norm(eq(x))**2
        return np.sqrt(cost)

    sigma = 1
    r = 1
    x = x0
    minimum = cost_function(x)
    x_hist, V_hist = [x] , [minimum]

    while cost_norm(x) > threshold:
        x, _ = steepest_descent(x0,
                             partial(phi, cost_function, sigma, r, mode, ecp, icp),
                             gradient_function = gradient_function,
                             step_size = step_size,
                             threshold = conv_threshold,
                             log = log,
                             h = h,
                             max_iter = max_iter,
                             fd_method = fd_method,
                             track_history = track_history)
                            
        sigma *= 10
        r /= 10
        if sigma >= 1e4 or r<= 1e-5:
            break
        minimum = cost_function(x)
        x_hist.append(x)
        V_hist.append(minimum)
    if track_history:
        return x, cost_function(x).item(), x_hist, V_hist
    else: 
        return x, cost_function(x).item() 



def augmented_lagrangian(x0, 
                         cost_function, 
                         equality_constraints = [], 
                         inequality_constraints = [],
                         threshold = 1e-3, 
                         log = False,
                         track_history = False,):
    '''
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        equality_constraints :: list of functions
            List of constraint equalities. 
        inequality constraints :: list of functions
            List of constraint inequalities. 
        threshold :: float
            Threshold at which to stop minimization. Values 
        log :: bool
            True to log optimization progress. Default: False
            Parameter for finite difference estimation. 
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    '''
    class P:
        def __init__(self, lambd, sigma):
            self.lambd = lambd
            self.sigma = sigma

        def phi(self, x):
            cost = cost_function(x)
            lambda_eq = lambd[:num_ec , :]
            lambda_ineq = lambd[num_ec:num_c , :]
            sigma_eq = sigma[:num_ec , :]
            sigma_ineq = sigma[num_ec:num_c , :]

            ecs = np.array([ec(x) for ec in equality_constraints])
            cost = cost - sum(lambda_eq * ecs) + 0.5 * sum(sigma_eq * ecs**2)
            
            for i, ineq in enumerate(inequality_constraints):
                ic = ineq(x)
                if ic <= lambda_ineq[i] / sigma_ineq[i]:
                    p_i = np.array([-lambda_ineq[i] * ic + 0.5 * sigma_ineq[i] * ic**2])
                else: 
                    p_i = np.array([-0.5 * lambda_ineq[i]**2 / sigma_ineq[i] ])
                cost = cost + p_i

            return cost
    minimum = cost_function(x0).item()
    x_history, V_history = [x0],[minimum]
    num_ec = len(equality_constraints)
    num_ic = len(inequality_constraints)
    num_c = num_ec + num_ic

    lambd = np.zeros((num_c,1))
    sigma = np.ones((num_c,1))

    c = 1e12 * sigma
    x = x0
    minimum = cost_function(x)
    x_history.append(x), V_history.append(minimum)
    j = 0
    while norm(c) > threshold and all(sigma < 1e12): 
        p = P(lambd, sigma)
        x,_ = steepest_descent(x,
                               p.phi,
                               None,
                               step_size = 'armijo',
                               threshold = 1e-6, 
                               max_iter = 1e4, 
                               fd_method = 'forward')

        previous_cost = c
        inequality_cost = [const(x) for const in inequality_constraints]
        equality_cost = [const(x) for const in equality_constraints]
        c = equality_cost + inequality_cost 

        if norm(c, np.inf) > 0.25 * norm(previous_cost, np.inf):
            for i in range(num_c):
                if np.abs(c[i]) > 0.25 * norm(previous_cost, np.inf):
                    sigma[i] *= 10
            continue
        lambd = lambd - sigma * c

        minimum = cost_function(x).item()
        x_history.append(x), V_history.append(minimum)
        j += 1
        if log:
            print(f'x = {x}, V(x) = {minimum:.5f}')
    if track_history:
        return x, cost_function(x).item(), x_history, V_history
    else:
        return x, cost_function(x).item()





def lagrange_newton(x0, 
                    cost_function, 
                    gradient_function = None,
                    hessian = None,
                    equality_constraints = [], 
                    inequality_constraints = [],
                    threshold=1e-8,
                    h = 1e-8,
                    fd_method = 'forward',
                    log = False,
                    track_history = False):
    '''
    Lagrange Newton Algorith for constrainted optimization
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used. 
        hessian :: np.array (shape: len(x0) x len(x0))
            Hessian of cost function. 
            if None, Finite difference hessian is used
        equality_constraints :: list of functions
            List of constraint equalities. 
        inequality constraints :: list of functions
            List of constraint inequalities. 
        threshold :: float
            Threshold at which to stop minimization. Values 
        h :: float
            should be close to 0. Default: 1e-8
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        log :: bool
            True to log optimization progress. Default: False
            Parameter for finite difference estimation. 
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    '''

    fd = Finite_Difference(cost_function, fd_method, h)
    if hessian is None: 
        hessian_ = fd.hessian
    else: 
        hessian_ = hessian
    if gradient_function is None:
        gradV = fd.estimate_gradient
    else: 
        gradV = gradient_function

    minimum = cost_function(x0).item()
    x_history, V_history = [x0], [minimum]

    num_ec = len(equality_constraints)
    num_ic = len(inequality_constraints)
    num_c = num_ec + num_ic
    lambd = np.zeros((num_c, 1))
    x = x0


    def W(x, lmb):
        lambda_eq = lmb[:num_ec, :]
        lambda_iq = lmb[num_ec:num_c, :]
        hess = hessian_(x)
        hess_eq = 0
        for i,ec in enumerate(equality_constraints):
            hess_eq -=  Finite_Difference(ec, fd_method).hessian(x) * lambda_eq[i] 
        hess_iq = 0
        for i,ic in enumerate(inequality_constraints):
            hess_iq -=  Finite_Difference(ic, fd_method).hessian(x) * lambda_iq[i] 
        return hess + hess_eq + hess_iq

    def A(x):
        equality_grads = [Finite_Difference(ec, fd_method).estimate_gradient(x) 
                                                for ec in equality_constraints]
        inequality_grads = [Finite_Difference(ic, fd_method).estimate_gradient(x) 
                                                for ic in inequality_constraints]
        grads = equality_grads + inequality_grads
        return np.array(grads).squeeze()

    dx = 1e12
    while True:

        if norm(dx) <= threshold: 
            break
        
        inequality_cost = [ic(x) for ic in inequality_constraints]
        equality_cost = [ec(x) for ec in equality_constraints]

        KKT = np.block([[W(x, lambd), -A(x).T],
                        [-A(x), np.zeros((num_c, num_c))]])
        if num_c == num_ic:
            funcs = np.block([[-gradV(x) + A(x) @lambd], 
                      [np.array(inequality_cost)]])
        elif num_c == num_ec:
            funcs = np.block([[-gradV(x) + A(x) @lambd], 
                    [np.array(equality_cost)]])
        else:
            funcs = np.block([[-gradV(x) + A(x) @lambd], 
                          [np.array(equality_cost)],
                          [np.array(inequality_cost)]])
        solution, _, _, _ = np.linalg.lstsq(KKT, funcs, rcond=1e-5)
        x0 = x
        x = x + solution[:x.shape[0], :]
        lambd = lambd + solution[x.shape[0]:, :]

        minimum = cost_function(x).item()
        x_history.append(x), V_history.append(minimum)
        dx = x - x0
        
        #track results
        if log: 
            print(f'x = {x}, V(x) = {minimum:.5f}')
    
    x_history.append(x), V_history.append(minimum)
    if track_history:
        return x, minimum, x_history, V_history
    else: 
        return x, minimum

    

    



