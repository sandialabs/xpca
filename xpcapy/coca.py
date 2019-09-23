from . import coca_tools as ct
from . import pca
from ._decomposer import _Decomposer

class COCA(_Decomposer):
    def __init__(self, rank=1, tol=0.01):
        _Decomposer.__init__(self, rank, tol)
        self.__copData = None

    @property
    def copData(self):
        return(self.__copData)
    @copData.setter
    def copData(self, copData):
        self.__copData = copData
    
    def fit(self, data):
        """
        Core code for fitting COCA model
    
        Preprocesses data, fits with PCA then post-processes data
        """
        self._fit(data)
        
    def _fit(self, data):
        # Data preprocessing
        self.copData = ct.raw2cop(data, denom_method = "m+1")

        # PCA
        PCA_fitter = pca.PCA(self.rank, tol = self.tol)
        PCA_fitter.fit(self.copData)
    
        # Post processing
        theta = PCA_fitter.fitted
        dataEst = ct.theta2median(theta, data)
    
        self.fitted = dataEst
        self.U = PCA_fitter.U
        self.V = PCA_fitter.V
        self.theta = theta
    
    
    