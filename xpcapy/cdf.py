"""
Normal(0,1) (Gaussian) distribution approximator functions for use in XPCA loss function computation.
We found that varying the cdf from scipy.stats.norm.cdf to scipy.special.ndtr improves the run time
performance without any hit to the accuracy (since norm.cdf is just a wrapper around ndtr). We've also
included other choices for the cdf computation that are estimations of cdf; these will improve speed
with a cost to accuracy.

Choices:

1. scipy.special.ndtr (default)
This is a high precision estimator for the CDF which is integral to scipy.  The documentation for this library is here:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html We just include the library here and do
not put another wrapper around it for speed's sake

2. sigmoid approximator

3. tanh and arctan approximator

authors: Luke Diaz
         Kina Kincher-Winoto

"""
import numpy as np

##################
# OPTION 1
from scipy.special import ndtr

##################
# OPTION 2

def sigmoidCDF(x):
    """
    This takes the form CDF(x) = 1/(1+exp(-beta*x^2)) where beta = 2*sqrt(2/pi).  Beta is chosen so that the first derivative
    of the CDF aligns with the first derivative of this approximator at the origin.
    This is an extremely fast approximator, with an average clocking demonstrating this computation taking approximately
    3.5 to 4 clock-cycles worth of C-array addition in numpy.

    :param x:
    :return:
    """
    beta       = -2*np.sqrt(2/np.pi)
    return np.reciprocal(1+np.exp(beta*x))

##################
# OPTION 3
def tanhCDF(x):
    """
    Based off paper:
    The accuracy and speed are between ndtr and sigmoid approximator.

    :param x:
    :return:
    """
    A = 39 / (2 * np.sqrt(2 * np.pi))
    B = 111 / 2
    C = 35 / (111 * np.sqrt(2 * np.pi))
    D = 1 / 2
    return(D*np.tanh(np.multiply(A,x)-np.multiply(B,np.arctan(np.multiply(C,x))))+D)
