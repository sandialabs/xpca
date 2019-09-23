"""XPCA, Python-pure version of copula-based rank decomposition

authors: Kina Kincher-Winoto
         Cliff Anderson-Bergman
         Luke Diaz
"""
import xpcapy
from ._decomposer import _Decomposer
import numpy as np
from . import xpcaoptimizer
from .optimizers import Decomp
from collections import namedtuple
from scipy.stats import rankdata
from scipy.stats import norm
from scipy import linalg
from scipy.stats.mstats import mquantiles
from . import cdf # normal CDF approximators
import logging


logger=logging.getLogger()
logger.setLevel(logging.INFO)

class XPCA(_Decomposer):
    """
    Class for copula-based PCA 
    
    """
    def __init__(self, rank=1, tol=0.01):
        super(XPCA, self).__init__(rank,tol)

    @property
    def left(self):
        return self.__left

    @left.setter
    def left(self, lef):
        self.__left = lef

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, rig):
        self.__right = rig

    @property
    def decomp(self):
        return self.__decomp
    @decomp.setter
    def decomp(self, d):
        self.__decomp = d
    
    @property
    def edflookup(self):
        return self.__edflookup

    @edflookup.setter
    def edflookup(self, value):
        self.__edflookup = value

    def fit(self, X, U0=None, V0=None, sigma0=1.0, tol=0.001, post_svd=True,
            method="alt-newt", max_iters=100,imputation="median", regularizer_value=0.5,
            cdf=cdf.__getattribute__('ndtr'), grid_size=None):
        """
        Options for cdf: ndtr (exact), sigmoidCDF (turbo), tanhCDF (in between)

        :param X:
        :param U0:
        :param V0:
        :param sigma0:
        :param tol:
        :param post_svd:
        :param method:
        :param max_iters:
        :param imputation:
        :param regularizer_value:
        :param cdf:
        :param grid_size: default is actually number of rows / 10.
        :return:
        """
        return self._fit(X, U0=U0,V0=V0, sigma0=sigma0,tol=tol, post_svd=post_svd,
                         method=method, max_iters=max_iters,
                         imputation=imputation, regularizer_value=regularizer_value,
                         cdf=cdf,grid_size=int(X.shape[0]/ 10))
    
    def _fit(self, X, U0, V0, sigma0, tol, post_svd, method, max_iters, imputation, regularizer_value,cdf, grid_size):
        """

        :param X:
        :param U0:
        :param V0:
        :param sigma0:
        :param tol:
        :param post_svd:
        :param method:
        :param max_iters:
        :param imputation:
        :param regularizer_value:
        :param cdf:
        :param grid_size:
        :return:
        """
        self.data = X
        # Step 1: run EDF per column to get Left and Right intervals
        intervals = self._raw_to_intervals(X, regularizer_value)
        nrows = X.shape[0]
        ncols = X.shape[1]


        # Step 2: Transform the EDF into Gaussian space using inverse normal cdf
        self.left = norm.ppf(intervals.left)
        self.right = norm.ppf(intervals.right)

        # Intermediate Step: initial values/guesses
        if U0 is None and V0 is None:
            # Use COCA as starting point
            coca_decomp = xpcapy.coca.COCA(rank=self.rank, tol=self.tol)
            coca_decomp.fit(X)
            U0 = coca_decomp.U
            V0 = coca_decomp.V
        else:
            # scale is standard deviation (notation of U,V to follow paper)
            if U0 is None or U0.shape != (nrows,self.rank):
                U0 = np.random.normal(scale=0.1, size=(nrows,self.rank))
            if V0 is None or V0.shape != (ncols, self.rank):
                V0 = np.random.normal(scale=0.1, size=(ncols,self.rank))


        # Step 3: Compute low-rank PCA
        solver = xpcaoptimizer.XPCAOptimizer(self.left, self.right, self.rank,cdf)
        # Options for cdf: ndtr (exact), sigmoidCDF (turbo), tanhCDF (in between)

        self.decomp = solver.solve(U0,V0,sigma0,tol=tol,method=method, max_iters=max_iters)
        # Step 3b: Run post SVD so U,V are orthogonal
        if (post_svd):
            _U, _s, _Vh = linalg.svd(self.decomp.theta, full_matrices=True, compute_uv=True)
            _s = np.concatenate((_s, np.zeros(_U.shape[1]-len(_s)))) # reshape to allow for matrix mult
            U_orth = (_U @ np.diag(_s))[:, :self.rank]
            V_orth = _Vh[:, :self.rank]

            # Have to create a whole new decomposition bc can't set U, V attributes
            decomp_orth = Decomp(U_orth, V_orth, self.decomp.sigma, self.decomp.theta,
                                 self.decomp.loss, self.decomp.numiters)

            self.decomp = decomp_orth

        # Step 4: Transform decomposition back to original data space
        self.edflookup = self._build_edf_lookup(X)
        if imputation == "median":
            imputations = self._calc_median_imps()
        elif imputation == "mean":
            imputations = self._calc_mean_imps(grid_size=grid_size)
        else:
            raise NotImplementedError("imputation method not yet implemented")

        # Set the ABC's attributes
        self.U = self.decomp.U
        self.V = self.decomp.V
        self.theta = self.decomp.theta
        self.fitted = imputations

        return self

    def _raw_to_intervals(self, X, regularizer_value, denominator_method="m", ):
        """
        Transforms X into intervals via EDF

        Parameters
        -------
        X : matrix-like, shape (m, n) 
            , where
            m is the number of rows or samples and
            n is the number of columns or features

        regularizer_value : value used to pull in maximum and minimum values.
                            interval value will be pulled in by
                            regularizer_value/denominator_method

        denominator_method : string, "m" or "m+1"
                      the denominator in the calculation of EDF

        Returns
        -------
        intervals : namedtuple of (Left, Right)

        """
        nrows = X.shape[0]
        ncols = X.shape[1]
        
        # create the left and right side of the intervals for data
        # matrix of -1
        left = np.zeros((nrows, ncols)) - 1
        right = np.zeros((nrows, ncols)) - 1
        
        # calc the intervals per column
        for j in range(ncols):
            col_interval = self._col_to_intervals(X[:,j], regularizer_value, denominator_method)
            left[:,j] = col_interval[0]
            right[:,j] = col_interval[1]

        Intervals = namedtuple('Intervals', ['left', 'right'])
        return Intervals(left=left, right=right)
        

    def _col_to_intervals(self, column, regularizer_value, denominator_method):
        """
        Takes a column of values and returns the interval that 
        it resides in

        """
        # m is the number of non-nan entries in the column
        m = sum(~np.isnan(column))
        
        # If denominator is "m", keep as is. Used for XPCA
        # else, if denominator is "m+1", add 1. Used for COCA
        if (denominator_method == "m"):
            denominator = m
        elif (denominator_method == "m+1"):
            denominator = m+1
        else:
            raise ValueError('Valid denominator methods are "m" '
                    'and "m+1". Given "%s"' % denominator_method)

        # Compute the intervals that the entry belongs in
        left = (rankdata(column, method='min')-1)/denominator
        right = (rankdata(column, method='max'))/denominator


        # Pull in the outermost intervals, if denominator was "m"
        if (denominator_method == "m"):
            # Alter the smallest interval
            left[left==0] = regularizer_value/denominator
            # Alter the largest interval which is 1
            right[right==1] = 1 - (regularizer_value/denominator)

        # If the data was missing (np.nan), then change the interval
        # to the min and max of the column
        _min = np.nanmin(left[~np.isnan(column)])
        _max = np.nanmax(right[~np.isnan(column)])
        left[np.isnan(column)] = _min
        right[np.isnan(column)] = _max

        return (left,right)

    def _build_edf_lookup(self, X):
        """
        Build edf lookup table
        """
        ncols = X.shape[1]
        lookup_per_col = dict()
        for col in range(ncols):
            lookup_per_col[col] = self._build_col_edf_lookup(X[:,col])

        # save the lookup table
        self.edflookup = lookup_per_col
        return self.edflookup

    def _build_col_edf_lookup(self, coldata):
        """
        Build the edf lookup table for this column of data (vector)
        """
        # Sort and unique data (numpy's unique sorts and uniques)
        col_nona = np.unique(coldata[~np.isnan(coldata)])
        # Get probabilities from values using edf
        probs = rankdata(col_nona,method='min') / len(col_nona)
        # Build the lookup table for this column in form of a dictionary
        return list(zip([float(p) for p in probs],col_nona))

    def _inverseEDF(self, edfs_to_invert):
        """
        Transform the data that comes in as probabilities (edfs) to values in original data space
        aka inverse EDF of the edfs

        :param edfs_to_invert: the data that is being transformed
        :return:
        """

        if self.edflookup is None:
            logging.warn("Inverse edf lookup table was not built. Imputations will not be completed.")
            return None

        vals = np.empty(edfs_to_invert.shape)

        # Iterate through edf, column by column (order='F') and look up prob
        col = 0
        for datacol in np.nditer(edfs_to_invert, flags=['external_loop'], order="F"):
            # working within a single column now
            table = self.edflookup[col] # original edf function for this col

            # For each element in the datacol, retrieve the value associated with the prob
            for i,x in enumerate(datacol):
                prevval = None
                # Since table is sorted, look for the key that's just bigger than x
                # special case is if x is >= to the largest probability value in the table
                if x >= table[-1][0]:
                    vals[i,col] = table[-1][1]
                    continue;
                for (p,v) in table:
                    if x < p:
                        # special case for the first prob
                        if prevval is None:
                            prevval = v
                        vals[i,col] = prevval
                        break;
                    prevval = v
            col+=1

        return vals

    def _calc_median_imps(self):
        """
        Calculates the median imputation of theta by inverting each step
        of transformation before decompostion:

        x_imputed = F^(-1)(\Phi(theta_ij))
    
        """
        if self.decomp is None:
            return None

        # Don't worry this doesn't do a deep copy. Python is a pointer passer
        theta = self.decomp.theta

        # Invert inverse-CDF so apply CDF to get back edfs:
        edfs = norm.cdf(theta)
        
        # Invert edf so have to rely on a lookup table to get back values from probabilities:
        # imps = self._inverseEDF(edfs)

        imps = np.zeros(theta.shape)
        for j in range(theta.shape[1]):
            # use linear interpolation of the emperical cdf
            imps[:,j] = mquantiles(self.data[:,j],prob=edfs[:,j], alphap=0, betap=1)

        return imps

    def _conditional_prob_P(self, tau_ij, sigma, f_hat):
        """
        This calculates the conditional probability that is
        in line 5 in Algorithm 9

        The conditional probability is given by equation 19 under section 4.2

        :param tau_ij:
        :param sigma:
        :param f_hat:
        :return:
        """
        # Recenter and rescale
        centered_z = (norm.ppf(f_hat) - tau_ij) / sigma

        # Compute cdf
        cdf_p = norm.cdf(centered_z)

        # If last entry of cdf isn't 1, force it by dividing everything
        # by that last entry.
        if (cdf_p[-1] != 1):
            cdf_p = cdf_p / cdf_p[-1]

        # Compute probability of each entry by subtracting each entry by the previous
        prob = cdf_p - np.concatenate(([0], cdf_p[0:-1]))

        return prob

    def _calc_mean_imps(self, grid_size=100):
        """
        Calculate the mean estimates according to algorithm 9, which estimates the
        full distribution by evaluating the expected value at configurable (via grid_size)
        number of points and then interpolating the rest. Assumes distribution is smooth.

        :param grid_size: number of points to compute before interpolation
        :return:
        """
        if self.decomp is None:
            return None

        # Don't worry this doesn't do a deep copy. Python is a pointer passer
        theta = self.decomp.theta
        nrows,ncols = theta.shape

        cdfs = self.edflookup

        sigma = self.decomp.sigma

        imps = np.zeros((nrows,ncols))

        interpolation_points = np.array(list(range(grid_size))) / (grid_size-1)
        for j in range(ncols):
            # tau is calculated per column
            tau_l = mquantiles(theta[:,j], prob=interpolation_points, alphap=1, betap=1)
            rho_l = np.empty(grid_size)
            # seperate out the probabilities and values in cdf (aka edf)
            f_hat_j, eta = zip(*cdfs[j])
            # calculate expected value of each point in tau_l
            for i in range(grid_size):
                # calculate the conditional probability of x = eta, given tau, sigma, f_hat
                conditional_prob = self._conditional_prob_P(tau_l[i], sigma, f_hat_j)
                # calculate the expected value of x
                rho_l[i] = np.sum(conditional_prob * eta)

            # remove the NAs if any present and issue a warning
            col_na = np.isnan(rho_l)
            if any(col_na):
                logging.warning("Column %i fitted NAs during mean approximation." % j)
                rho_l = rho_l[~col_na]
                tau_l = tau_l[~col_na]

            # if there's nothing to interpolate, return NA across column
            if len(rho_l) == 0:
                imps[:,j] = [np.nan] * nrows
                continue

            # if all of the values are the same, it's fitted as constant
            if (all(rho_l == rho_l[0])):
                imps[:,j] = rho_l[0] * nrows
                continue

            # interpolate
            imps[:,j] = np.interp(theta[:,j], xp=tau_l, fp=rho_l)


        return imps












