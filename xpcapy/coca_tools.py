## Tools for COCA

# Pre and post processing data tools for COCA

import numpy as np
from scipy.stats import rankdata
from scipy.stats import norm

def col2cop(vals, denom_method = "m+1"):
    """
    Takes in one column of raw data 
    Returns the copula value for COCA
    """
    # m = number of observed values
    isObs = ~np.isnan(vals)
    m = np.sum(isObs)
    if denom_method == "m+1" :
        denom = m+1
    elif denom_method == "m" : 
        denom = m
    else :
        raise ValueError("denom_method should be 'm' or 'm+1'")
        
        
    # Computing ranks for observed data
    obs_vals = rankdata(vals[isObs], method = "average") / denom
    # Converting to z-value
    obs_vals = norm.ppf(obs_vals)
    # Filling out the vector of data to return
    ans = np.zeros(vals.size)
    ans[~isObs] = np.nan
    ans[isObs] = obs_vals
    
    return(ans)

def raw2cop(mat, denom_method = "m+1"):
    """
    Takes in a matrix of raw data
    Returns a matrix of copula values
    """
    ans = mat.copy()
    nCols = mat.shape[1]
    for j in range(nCols):
        ans[:,j] = col2cop(mat[:,j], denom_method=denom_method)
    return(ans)    
    
def quantile(x, p):
    """
    Quantile function used for COCA
    """
    # numpy.percentile expects values in [0,100] 
    p_use = p * 100.0
    x_use = x[~np.isnan(x)]
    ans = np.percentile(x_use, p_use, interpolation="lower")
    return(ans)
    
def theta2median_vec(theta_vec, col_vec):
    """
    Takes in a vector of theta values and the column of raw data
    Returns the corresponding median estimates
    """
    latent_u = norm.cdf(theta_vec)
    ans = quantile(col_vec, latent_u)
    return(ans)
    
def theta2median(theta_mat, raw_data):
    """
    Takes in a theta matrix and raw data matrix
    Returns median COCA estimates for entire matrix
    note that raw_data should be original data, **not** copula-ized data!!
    """
    # copying to get dimensions correct
    ans = theta_mat.copy()
    # computing median estimate column by column
    nCols = theta_mat.shape[1]
    for j in range(nCols):
        ans[:,j] = theta2median_vec(theta_mat[:,j], raw_data[:,j])
        
    return(ans)
