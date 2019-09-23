from abc import ABC, abstractmethod

class _Decomposer(ABC):
    """
    Virtual Class for MXD Classes
    """
    def __init__(self, rank=1, tol=0.01):
        self.rank = rank
        self.tol = tol
        self.data = None
        self.U = None
        self.V = None
        self.theta = None
        self.fitted = None

    @property
    def rank(self):
        return(self.__rank)
    @rank.setter
    def rank(self, rank):
        self.__rank = rank

    @property
    def tol(self):
        return(self.__tol)
    @tol.setter
    def tol(self, tol):
        self.__tol = tol

    @property
    def U(self):
        return(self.__U)
    @U.setter
    def U(self, U):
        self.__U = U
    
    @property
    def V(self):
        return(self.__V)
    @V.setter
    def V(self, V):
        self.__V = V

    @property
    def theta(self):
        return(self.__theta)
    @theta.setter
    def theta(self, theta):
        self.__theta = theta

    @property
    def fitted(self):
        return(self.__fitted)
    @fitted.setter
    def fitted(self, fitted):
        self.__fitted = fitted
        
    @property
    def data(self):
        return(self.__data)
    @data.setter
    def data(self, data):
        self.__data = data

    @abstractmethod
    def fit(self):
        """
        Decomposes the matrix into
        Theta = AB^T

        All functions should implement a fit function.

        Returns
        -------

        """
        pass
