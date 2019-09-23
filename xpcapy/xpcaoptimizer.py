"""
Functions specific to XPCA's optimization. The loss functions, derivatives
and other functions are within this file.

"""

from scipy.stats import norm as norm
import numpy as np
from . import optimizers
import logging

logger = logging.getLogger("XPCAOptimizer")

class XPCAOptimizer():
    ALT_NEWT = ['alt-newt', 'alt_newt', 'altnewt', 'alternating-newton']
    LBFGS = ['lbfgsb', 'lbfgs', 'lbfgs-b', 'l-bfgs-b', 'l-bfgsb']
    AVAILABLE_METHODS = ALT_NEWT + LBFGS

    def __init__(self, left, right, rank, cdf):
        # Options for cdf: ndtr (exact), sigmoidCDF (turbo), tanhCDF (in between)
        self.cdf               = cdf 
        self.L                 = left
        self.R                 = right
        self.nrows, self.ncols = left.shape
        self.rank              = rank
        self.d1 = None
        self.d2 = None

    def solve(self, U0, V0, sigma0, tol, method, max_iters):
        logger.debug("Solving with %s", method)
        if U0 is None:
            U0 = np.random.normal(scale=0.1, size=(nrows,self.rank))
        if V0 is None:
            V0 = np.random.normal(scale=0.1, size=(ncols,self.rank))

        if method not in self.AVAILABLE_METHODS:
            raise NotImplementedError("Not yet implemented.")
    
        if method in self.ALT_NEWT:
            return optimizers._solve_alt_newt(U0, V0, sigma0, self.L, self.R, loss=self.loss_w_sigma,
                                              vectorloss=self.loss_vector,
                                              d1_U_fun=self._d1_u, d1_V_fun=self._d1_v, d2_theta_fun=self._d2_theta,
                                              d1_sigma_fun=self._d1_sigma, d2_sigma_fun=self._d2_sigma,
                                              tol=tol, max_iters=max_iters)

        if method in self.LBFGS:
            return optimizers._solve_lbfgs(U0, V0, sigma0, self.inflate, self.deflate,
                                           loss=self.full_loss, d1fun=self.xpca_d1, tol=tol, max_iters=max_iters)

    
    #### XPCA Helper functions ####
    def deflate(self, u, v, sigma):
        """
        Helper function that takes u,v,sigma and concatenates them all into
        one vector.
        Opposite of inflate.

        :param u:
        :param v:
        :param sigma:
        :return:
        """
        nrows = self.L.shape[0]
        ncols = self.L.shape[1]
        rank = self.rank

        u_v_sigma = np.concatenate([u.reshape((1,nrows*rank), order='F'),
            v.reshape((1,ncols*rank), order='F'),
            np.array([[sigma]])], axis=1)[0]
        return u_v_sigma

    def inflate(self,u_v_sigma):
        """
        Helper function that takes a vactor and outputs u,v,sigma.
        Opposite of deflate.
        :param u_v_sigma:
        :return:
        """

        nrows = self.L.shape[0]
        ncols = self.L.shape[1]
        rank = self.rank

        # reshape theta and sigma from vector
        u = u_v_sigma[:(nrows*rank)].reshape(nrows,rank, order='F')
        v = u_v_sigma[(nrows*rank):(nrows*rank)+(ncols*rank)].reshape(ncols,rank,order='F')
        sigma = u_v_sigma[-1]


        return (u,v,sigma)

    #### XPCA functions for lbfgs and optimization of sigma ####
    def full_loss(self, u_v_sigma):
        """
        Calculates the loss of flattened u,v,sigma. The loss is
        -log( \Phi( (r_{ij} - theta_{ij}) / sigma ) - \Phi( (l_{ij} - theta_{ij}) / sigma ) )
        for i,j in theta.

        :param u_v_sigma:
        :return:
        """
        # reshape u_v_sigma
        u,v,sigma = self.inflate(u_v_sigma)
        return self.loss_w_sigma(u,v,sigma)

    def loss_w_sigma(self, u,v,sigma, theta = None):
        """
        This is the loss with sigma. There are times when we actually can set sigma to 1
        and ignore it during calculation of loss, but this function does not take that
        shortcut.  

        :param u:
        :param v:
        :param sigma:
        :param theta:
        :return:
        """
        if theta == None:
            theta = u @ v.T
        L_adj = (self.L - theta) / sigma
        R_adj = (self.R - theta) / sigma
        logger.debug((L_adj == R_adj).any())
        probs = self.cdf(R_adj) - self.cdf(L_adj)
    
        logger.debug("probs in loss:%s", np.array2string(probs))
        log_probs = np.log(probs)
        log_probs[np.isinf(log_probs)] = 0.0
        loss = -np.sum(log_probs)
        logger.debug("loss: %.10f", loss)
        return loss

    def loss_vector(self,L_i, R_i, theta_i, sigma):
        """
        Compute the loss for a specific vector. Needed because sometimes
        we don't want to compute the loss for the entire matrix
        :param theta_i:
        :param L_i:
        :param R_i:
        :param sigma:
        :return:
        """
    
        L_adj = (L_i - theta_i) / sigma
        R_adj = (R_i - theta_i) / sigma
        probs = self.cdf(R_adj) - self.cdf(L_adj)

        log_probs = np.log(probs)
        log_probs[np.isinf(log_probs)] = 0.0
        return -np.nansum(log_probs)

    #### XPCA D1 and D2 functions ####
    def xpca_d1(self, u_v_sigma):
        """
        1st derivative of U, V, Sigma
        :param u_v_sigma: matrix of U,V,Sigma all in one
        :return:
        """
        # reshape theta and sigma from vector
        u,v,sigma = self.inflate(u_v_sigma)

        # calculation of helper values to dervs
        helpers = self._derv1_helpers(u, v, sigma)

        # calculation of 1st derivative wrt to theta, U, V
        d1_U = self._d1_u(u,v,sigma, derv_helpers=helpers)
        d1_V = self._d1_v(u,v,sigma, derv_helpers=helpers)

        # calc of 1st derv wrt sigma
        d1_S = self._d1_sigma(u, v, sigma, derv_helpers=helpers, ignore_na=True)

        # flatten back out
        self.d1 = self.deflate(d1_U, d1_V, d1_S)
        return self.d1

    def xpca_d2(self, u_v_sigma):
        """
        2nd derivative of U, V, Sigma
        :param u_v_sigma: matrix of U,V,Sigma all in one
        :return:
        """
        # reshape theta and sigma from vector
        u,v,sigma = self.inflate(u_v_sigma)

        # calculation of helper values to dervs
        helpers = self._derv2_helpers(u, v, sigma)

        # calc of 2nd derv wrt sigma
        d2_sigma = self._d2_sigma(u, v, sigma, derv2_helpers=helpers)

        # calc of 2nd derv wrt theta
        d2_theta = self._d2_theta(u,v,sigma, derv2_helpers=helpers)

        # calc of 2nd derv wrt u, v
        d2_u = self._d2_u(u,v,sigma,d2_theta=d2_theta)
        d2_v = self._d2_v(u,v,sigma,d2_theta=d2_theta)

        return self.deflate(d2_u, d2_v, d2_sigma)

    def _derv1_helpers(self, u, v, sigma, theta=None):
        """
        Helper function as to not recalculate these helpers over and over
        """
        if theta == None:
            theta = u @ v.T

        L_adj = (self.L - theta) / sigma
        R_adj = (self.R - theta) / sigma

        # phi of (L - theta) (pdf)
        pdf_l = norm.pdf(L_adj)
        # phi of (R - theta) (pdf)
        pdf_r = norm.pdf(R_adj)
        # Phi of (L - theta) (cdf)
        cdf_l = self.cdf(L_adj)
        # Phi of (R - theta) (cdf)
        cdf_r = self.cdf(R_adj)
        probs = cdf_r - cdf_l

        return (L_adj, R_adj, pdf_l, pdf_r, cdf_l, cdf_r)

    def _d1_u(self, u, v, sigma, derv_helpers=None):

        if derv_helpers == None:
            derv_helpers = self._derv1_helpers(u, v, sigma)

        L_adj, R_adj, pdf_l, pdf_r, cdf_l, cdf_r = derv_helpers

        d1_theta = (pdf_r - pdf_l) / ((cdf_r - cdf_l) * sigma)

        # mxn @ nxk --> mxk
        d1_u = d1_theta @ v

        return d1_u

    def _d1_v(self, u, v, sigma, derv_helpers=None):
        if derv_helpers == None:
            derv_helpers = self._derv1_helpers(u, v, sigma)

        L_adj, R_adj, pdf_l, pdf_r, cdf_l, cdf_r = derv_helpers

        d1_theta = (pdf_r - pdf_l) / ((cdf_r - cdf_l) * sigma)


        # (mxn).T @ mxk --> nxm @ mxk
        d1_v = d1_theta.T @ u

        return d1_v

    def _d1_sigma(self, u, v, sigma, derv_helpers=None, ignore_na=False):
        if derv_helpers == None:
            derv_helpers = self._derv1_helpers(u, v, sigma)

        L_adj, R_adj, pdf_l, pdf_r, cdf_l, cdf_r = derv_helpers

        # calc of 1st derv wrt sigma
        L_adj[np.isinf(L_adj)] = 0.0
        R_adj[np.isinf(R_adj)] = 0.0
        d1_S = (np.sum( ((pdf_r * R_adj) - (pdf_l * L_adj)) / (cdf_r - cdf_l))) / sigma
        if np.isinf(d1_S):
            d1_S = 0.0

        return d1_S

    def _derv2_helpers(self, u, v, sigma, theta=None):
        L_adj, R_adj, pdf_l, pdf_r, cdf_l, cdf_r = self._derv1_helpers(u,v,sigma, theta=theta)

        # The limit of R_adj * pdf(R_adj) approaches 0 as R_adj goes to infinity (same for L_adj)
        # This should be captured in pdf(Inf) = 0, but we need to change R_adj, L_adj
        L_adj[np.isinf(L_adj)] = 0
        R_adj[np.isinf(R_adj)] = 0

        # what is referred to as P
        probs = cdf_r - cdf_l
        pdf_L_L = pdf_l * L_adj
        pdf_R_R = pdf_r * R_adj

        return (L_adj, R_adj, pdf_l, pdf_r, probs, pdf_L_L, pdf_R_R)

    def _d2_theta(self, u,v,sigma, derv2_helpers=None):
        if derv2_helpers == None:
            derv2_helpers = self._derv2_helpers(u, v, sigma)

        L_adj, R_adj, pdf_l, pdf_r, probs, pdf_L_L, pdf_R_R = derv2_helpers

        # calc of 2nd derv wrt theta
        part1 = (pdf_R_R-pdf_L_L) / probs
        part2 = ( (pdf_r - pdf_l) / probs )**2
        d2_theta = part1 + part2

        return d2_theta

    def _d2_row_theta(self, u, v, sigma, derv2_helpers=None, d2_theta=None):
        if d2_theta == None:
            # calc wrt to theta first
            d2_theta = self._d2_theta(u,v,sigma, derv2_helpers)

        d2_row = d2_theta

        return d2_row

    def _d2_sigma(self, u, v, sigma, derv2_helpers=None):
        if derv2_helpers == None:
            derv2_helpers = self._derv2_helpers(u, v, sigma)

        L_adj, R_adj, pdf_l, pdf_r, probs, pdf_L_L, pdf_R_R = derv2_helpers

        # calc of 2nd derv wrt sigma
        part1 = (pdf_R_R - pdf_L_L) ** 2
        part2a = (R_adj ** 3 * pdf_r)
        part2b = (L_adj ** 3 * pdf_l)
        part2c = 2 * (pdf_L_L - pdf_R_R)
        numerator = np.sum( (part1 + probs * (part2a - part2b + part2c)) / (probs ** 2))

        d2_sigma = numerator / (sigma ** 2)

        return d2_sigma