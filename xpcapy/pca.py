"""
PCA with missing data

authors: Cliff Anderson-Bergman
         Kina Kincher-Winoto
"""
import numpy as np
from ._decomposer import _Decomposer
from . import pca_tools as pt

class PCA(_Decomposer):
    """
    Class for PCA 
    
    """
    def __init__(self, rank=1, tol=0.01):
        _Decomposer.__init__(self, rank, tol)
        self.__colMeans = None
        self.__colSds = None

    @property
    def colMeans(self):
        return(self.__colMeans)
    @colMeans.setter
    def colMeans(self, colMeans):
        self.__colMeans = colMeans
    
    @property
    def colSds(self):
        return(self.__colSds)
    @colSds.setter
    def colSds(self, colSds):
        self.__colSds = colSds
    
    def fit(self, X, U0=None, V0=None, tol=0.001, post_svd=True,
        method="alt_newt", max_iters=100, center = True, scale = True, 
        verbose = False):
        """
        Fits PCA with scaling
        """
        return self._fit(X, U0, V0, tol, post_svd, method, 
                         max_iters, center, scale, verbose = verbose)
    
    def _fit(self, X, U0, V0, tol, 
             post_svd, method, max_iters, 
             center, scale, verbose = False):
        self.data = X     
        nRows = X.shape[0]
        nCols = X.shape[1]
        mask = np.logical_not( np.isnan(X) )
        # Step 1: Standardize data
        stdzInfo = pt.stdzData(X, center = center, scale = scale)
        X_use = stdzInfo["data"]
        self.colMeans = stdzInfo["colMeans"]
        self.colSds = stdzInfo["colSds"]
        
        # Step 2: Initialize Factors
        if U0 is None or U0.shape != (nRows,self.rank):
            U0 = np.random.normal(scale=0.1, size=(nRows, self.rank))
        if V0 is None or V0.shape != (nCols, self.rank):
            V0 = np.random.normal(scale=0.1, size=(nCols, self.rank))

        U = U0
        V = V0

        # Step 3: Alternating Least Squares
        iter = 0
        err = np.infty
        old_loss = pt.pcaLoss(U, V, X_use)
        while( (iter < max_iters) & (err > tol)):
            iter = iter + 1
            U = pt.updateU(U, V, X_use, mask)
            V = pt.updateV(U, V, X_use, mask)
            new_loss = pt.pcaLoss(U, V, X_use)
            err = old_loss - new_loss
            old_loss = new_loss
            if(verbose):
                print("Current Loss = ", new_loss)
        
        self.U = U
        self.V = V
        
        theta = U @ V.transpose()
        fitted = pt.unStdzData(stdzInfo, theta)
        
        self.fitted = fitted



