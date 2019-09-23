"""
Convenience functions to simulate data for testing.
"""
from scipy.stats import norm as norm
import numpy as np
import logging
from collections import namedtuple

SimulatedData = namedtuple ("SimulatedData", ['data_matrix', 'mean_matrix', 'mask'])

def simulate_data(mrows=100, ncols=10, sigma=0.5, prop_binary=0, rank=3, cont_error="normal", prob_missing=0.0):
    if (sigma > 1 or sigma < 0):
        logging.error("Sigma cannot be greater than 1 or less than 0")
        return None

    # Simulate V to standard normal (mean=0, variance=1). V dim is n x k
    V_raw = np.random.normal(size=(ncols,rank))
    VV_t_raw = V_raw @ V_raw.T
    VV_t = np.corrcoef(VV_t_raw) * (1-sigma**2)

    # Simulate theta
    theta = np.random.multivariate_normal(np.zeros(ncols),cov=VV_t,size=mrows)
    logging.debug("shape of theta (%i, %i)" % theta.shape)

    # Set binary columns
    ncols_binary = int(np.round(prop_binary*ncols))
    is_binary = [True] * ncols_binary + [False] * (ncols-ncols_binary)

    # Compute mean for each column
    mean_matrix = np.empty(shape=(mrows,ncols))
    for j in range(ncols):
        if is_binary[j]:
            mean_matrix[:,j] = norm.cdf(theta[:,j], scale=sigma)
        else:
            if cont_error == "normal":
                mean_matrix[:,j] = theta[:,j]
            elif cont_error == "exp":
                raise NotImplementedError
            else:
                logging.error("%s not implemented as an error method" % cont_error)
                raise NotImplementedError

    # Add noise, create actual simulations
    copula_matrix = theta + np.random.normal(scale=sigma, size=(mrows,ncols))
    data_matrix = np.zeros((mrows,ncols))
    for j in range(ncols):
        if is_binary[j]:
            data_matrix[:,j] = [1 if copula_matrix[i,j] > 0 else 0 for i in range(mrows)]
        else:
            if cont_error == "normal":
                data_matrix[:,j] = copula_matrix[:,j]
            elif cont_error == "exp":
                logging.error("%s not implemented as an error method" % cont_error)
                raise NotImplementedError

    # Add percentage missing
    if prob_missing > 0:
        mask = np.random.binomial(1, prob_missing, size=theta.shape)
        masked_data = np.ma.array(data_matrix, mask=mask, fill_value=np.nan)
        return SimulatedData(masked_data.filled(), mean_matrix, mask)


    return SimulatedData(data_matrix,mean_matrix, None)

