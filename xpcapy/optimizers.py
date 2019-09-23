"""
"""
import numpy as np
import logging
from collections import namedtuple
from scipy.optimize import minimize

logger = logging.getLogger('optimizers')
Decomp = namedtuple('decomp', ['U','V','sigma','theta', 'loss', 'numiters'])

### L-BFGS-B ###

def _solve_lbfgs(U0, V0, sigma0, inflater, deflater, loss, d1fun, tol, max_iters):
    """
    Solve using scipy's L-BFGS-B
    """
    logging.info("Solving with L-BFGS-B...")

    nrows = U0.shape[0]
    rank = U0.shape[1]
    ncols = V0.shape[0]

    # vectorize matrix
    uvs_vec = deflater(U0, V0, sigma0)

    # Set bounds for u,v, sigma
    u_bounds = [(-np.inf, np.inf)] * (nrows*rank)
    v_bounds = [(-np.inf,np.inf)] * (ncols*rank)
    s_bounds = [(0.001,1.0)]
    bounds = u_bounds + v_bounds + s_bounds

    if logger.getEffectiveLevel() == logging.DEBUG:
        # Call scipy's optimize function to check that the analytic and numeric dervs match
        import scipy.optimize
        deriv_numeric = scipy.optimize.approx_fprime(uvs_vec,loss, 0.001)
        deriv_analy = d1fun(uvs_vec)
        logging.debug("difference between anly and nmrc deriv: " )
        logging.debug(np.mean(deriv_numeric-deriv_analy))

    logger.debug("### tol:%.10f", tol)
    soln = minimize(loss, uvs_vec, jac=d1fun, method='l-bfgs-b', bounds=bounds, options={'maxiter':max_iters, 'gtol':tol, 'disp':True})
    logger.debug("### lbfgs soln")
    logger.debug(soln)
    u,v,s = inflater(soln.x)
    theta = u @ v.T
    return Decomp(u,v,s,theta,soln.fun,soln.nit)


### ALT-NEWTONS ###

def _solve_alt_newt(U0, V0, sigma0, left, right, loss, vectorloss,
                    d1_U_fun, d1_V_fun, d2_theta_fun,
                    d1_sigma_fun, d2_sigma_fun,
                    tol = 0.1, max_iters=100, updateSigma=True):
    """
    Solve via alternating newton's method

    :param U0:
    :param V0:
    :param sigma0:
    :param left:
    :param right:
    :param loss:
    :param d1fun:
    :param d2fun:
    :param tol:
    :param max_iters:
    :param updateSigma:
    :return:
    """
    logging.info("Solving with alternating newton...")

    # Initialize variables
    err = tol + 1
    loss_i = loss(U0,V0, sigma0)
    i = 0
    U = U0
    V = V0
    sigma = sigma0

    # Alternate updating U, V, and sigma
    while (i<max_iters and err > tol):
        i+=1
        logging.debug("Loss: %f", loss_i)
        _alt_newt_updateU(U, V, sigma, left, right, d1_U_fun, d2_theta_fun, vectorloss)
        _alt_newt_updateV(U, V, sigma, left, right, d1_V_fun, d2_theta_fun, vectorloss)
        if (updateSigma):
            _alt_newt_updateSigma(U, V, sigma, d1_sigma_fun, d2_sigma_fun, loss)

        prev_loss = loss_i
        loss_i = loss(U,V,sigma)
        logging.debug ("prev loss: %f\t loss_i: %f", prev_loss, loss_i)
        err = prev_loss - loss_i
        logging.info("Iteration %i out of maximum %i: Loss = %.4f" %(i, max_iters, loss_i))

    logging.debug("Final loss: %f \nTotal iterations: %i" % (loss_i,i))

    theta = U @ V.T
    # Return U,V, theta with loss and number of iterations
    return Decomp(U,V, sigma, theta,loss_i, i)


def _newtonshift(d1_i, d2_i, X):
    """
    Calculates the proposed shift for matrix. We expect 2nd derv to be only wrt to theta.
    This is because to calculate the entire hessian would be too expensive especially when we
    only need the diagonal of the hessian. So in this function we calculate the hessian
    wrt to X

    :param d1_i: 1st derv of matrix trying to shift
    :param d2_i: 2nd derv wrt to theta
    :param X: matrix being frozen (in alternating)
    :return:
    """
    # Check if 2nd derv is positive
    if all(d2_i > 0):
        # Multiply each *column* of X (which can be either U or V) by the second derv
        # if X=V (meaning updating U): (nxk) * nx1 --> nxk (dimensions remain the same)
        # if X=U (meaning updating V): (mxk) * mx1 = mxk (dimensions remain the same)
        X_adj = X * np.sqrt(d2_i)[:, None]
    else:
        # if X=V: V is nxk, so is V_adj
        # if X=U: U is mxk, so is U_adj
        X_adj = X

    # if X=V: t(nxk) @ nxk --> kxn @ nxk --> kxk
    # if X=U: t(mxk) @ mxk --> kxm @ mxk --> kxk
    hess = X_adj.T @ X_adj

    # Solve the equation (d2_i)? = d1_i
    # if X=V: (kxk) (?) = t(1xk) so soln dim: kx1
    # if X=U: (kxk) (?) = t(1xk) so soln dim: kx1
    logging.debug("Hessian:")
    logging.debug(hess)
    soln = np.linalg.solve(hess, -d1_i.T)

    if any(np.isinf(soln)) or any(np.isnan(soln)):
        return np.zeros(soln.shape)

    return soln

def _halfstep(L_i, R_i, theta_i, sigma, beta, X, delta_i, vectorlossfun):
    """
    Ensures that the loss decreases at each step. If it
    didn't decrease with the full step of delta_i, then
    continually half the step.
    L_i: left side vector
    R_i: right side vector
    theta_i: current theta (before step) 
    beta: vector that is changing
    X: other matrix of decomposition
    delta_i: proposed amount of change to beta
    lossfun: loss function to evaluate change
    """
    # Compute loss before step
    loss_i = vectorlossfun(L_i, R_i, theta_i, sigma)
    
    # Add step to beta, theta
    beta_updated = beta + delta_i
    theta_updated = X @ beta_updated
    
    # Compute current low after step
    curloss = vectorlossfun( L_i, R_i, theta_updated, sigma)

    # Check if loss improved and
    # only do a set number of halving iterations
    err = curloss - loss_i
    ihalved = 0
    while (err >= 0 and ihalved < 5):
        ihalved+=1

        # Halve delta and update beta and theta
        delta_i = delta_i / 2.0
        beta_updated = beta + delta_i
        theta_updated = X @ beta_updated

        # Calculate new loss
        curloss = vectorlossfun(L_i, R_i, theta_updated, sigma)
        err = curloss - loss_i

    # If we the loss was improved with delta_i or any series of 
    # delta_i halved, return the updated beta.
    if err < 0:
        return beta_updated
    else:
        return beta
    
def _alt_newt_updateU(U, V, sigma, L, R, d1_U_fun, d2_theta_fun, vectorlossfun):
    """
    Using 1st and 2nd derivatives, update U
    :param U:
    :param V:
    :param L:
    :param R:
    :param d1_U_fun:
    :param d2_theta_fun:
    :param vectorlossfun:
    :return:
    """

    # Calculate theta
    theta = U @ V.T

    # Get every row in theta so since theta is mxn and U is mxk, we get m
    m = U.shape[0]
    # Compute derivatives wrt U, for 2nd derivative wrt theta for speed
    d1_U = d1_U_fun(U,V,sigma)
    d2_theta = d2_theta_fun(U, V, sigma)
    # Update each row
    for i in range(m):
        # Get current row
        L_i = L[i,:]
        R_i = R[i,:]
        theta_i = theta[i,:]
        # Compute proposed shift from Newton's method
        # delta_i dim is 1xk
        delta_i = _newtonshift(d1_U[i,:], d2_theta[i,:], V)

        # Check that loss is decreasing and reassign if not
        updated_row_i = _halfstep(L_i, R_i, theta_i, sigma, U[i,:], V, delta_i, vectorlossfun)

        U[i,:] = updated_row_i

    return U

def _alt_newt_updateV(U, V, sigma, L, R, d1_V_fun, d2_theta_fun, vectorlossfun):
    """
    A little repetitive, but helps with clarity of code to
    have updateU and updateV
    :param theta:
    :param U:
    :param V:
    :param L:
    :param R:
    :param d1fun:
    :param d2fun:
    :param lossfun:
    :return:
    """

    # Calculate theta
    theta = U @ V.T

    # Get every column in theta so since theta is mxn and V is nxk, we get n
    n = V.shape[0]
    # Compute derivatives wrt V, for 2nd derivative wrt theta for speed
    d1_V = d1_V_fun(U,V,sigma)
    d2_theta = d2_theta_fun(U, V, sigma)
    logging.debug(d2_theta.shape)
    # Update each row
    for j in range(n):
        # Get current col
        L_j = L[:,j]
        R_j = R[:,j]
        theta_j = theta[:,j]
        # Compute proposed shift from Newton's method
        delta_j = _newtonshift(d1_V[j,:], d2_theta[:,j], U)

        # Check that loss is decreasing and reassign if not
        updated_row_j = _halfstep(L_j, R_j, theta_j, sigma, V[j,:], U, delta_j, vectorlossfun)

        V[j,:] = updated_row_j

    return V

def _alt_newt_updateSigma(U, V, sigma, d1_sigma_fun, d2_sigma_fun, lossfun):
    """
    Update sigma using 1st and 2nd derivatives
    :param U:
    :param V:
    :param sigma:
    :param d1_sigma_fun:
    :param d2_sigma_fun:
    :param lossfun:
    :return:
    """
    # Compute derivatives wrt to sigma
    d1 = d1_sigma_fun(U,V,sigma)
    d2 = d2_sigma_fun(U,V,sigma)

    # if locally convex, use Newton
    if (d2 > 0).all():
        delta = -d1/d2
    else:
        delta = -np.sign(d1) /10

    # Force sigma to be within [0.001,1]
    delta = max(0.001-sigma, delta) # don't go past 0
    delta = min(1-sigma, delta) # don't go past 1

    # Update sigma and save previous val for line search
    sigma0 = sigma
    sigma_updated = sigma0 + delta
    loss0 = lossfun(U,V,sigma0)
    curloss = lossfun(U,V,sigma_updated)


    # Iteratively make delta change smaller and smaller so that our loss is decreasing
    iterations = 0
    while (curloss > loss0 and iterations < 4):
        delta = delta / 4
        sigma_updated = sigma0 + delta
        curloss = lossfun(U,V,sigma_updated)

        iterations = iterations + 1

    # If we failed to find a delta that decreases loss
    if (curloss > loss0):
        logging.warning("Newton's method for finding an optimal sigma failed. (delta: %f, d1: %f, d2: %f)"
                        % (delta, d1,d2))
        sigma_updated = sigma0
        curloss = loss0

    return sigma_updated



