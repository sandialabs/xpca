import numpy as np

## FUNCTIONS DEFINED:
# stdzData: Standarizing data
# lsqr: least squares
# singular_lsqr: least squares with singular XtX

def stdzData(data, center = True, scale = True, warn = True):
    """ 
    Standardize data
    Pre-processing step for PCA. 
    Returns a tuple of standardized data, 
    column means and column standard deviations
    """
    
    nCols = data.shape[1]
    colMeans = np.zeros(nCols)
    colSds = np.zeros(nCols)
    # We will print a warning if sd == 0 in any column
    warningFlag = False
    zeroSdInds = np.zeros(0)

    stdz_data = data.copy()

    for j in range(0, nCols):
        if(center):
            this_mean = np.nanmean(data[:,j])
            colMeans[j] = this_mean
            stdz_data[:,j] = data[:,j] - this_mean
        if(scale):
            this_var = np.nanvar(data[:,j])
            this_sd = np.sqrt(this_var)
            colSds[j] = this_sd
            if(this_sd == 0):
                warningFlag = True
                zeroSdInds = np.append(zeroSdInds, j)
            else:
                stdz_data[:,j] = (data[:,j] - this_mean) / this_sd
    # Print warning if columns had zero sd
    if(warn):
        if(warningFlag):
            print("Warning: the following columns had sd = 0")
            print(zeroSdInds, "\n")
    return({'data':stdz_data, 'colMeans':colMeans, 'colSds':colSds})



def unStdzData(stdzInfo, theta):
    """
    Convert standardized estimate to orginal scale
    
    @param stdzInfo    Object returned from stdzData
    @param theta       Fitted estimate
    """
    
    ans = stdzInfo["data"].copy()
    nCol = ans.shape[1]
    means = stdzInfo["colMeans"]
    sds = stdzInfo["colSds"]
    for j in range(0, nCol):
        this_mean = means[j]
        this_sd = sds[j]
        thisCol = theta[:,j] * this_sd + this_mean
        ans[:,j] = thisCol
    return(ans)


def lsqr(X, y, use):
    """
    Least Squares for PCA
    Barebones least squares. If system is under determined, 
    will set coefficients to 0 until system is determined
    
    @param X    U or V matrix
    @param y    Slice of data matrix
    @param use  Vector indicating value of y is observed
    
    
    For PCA, y will be a slice of the data matrix **with missing values
    already dropped**. Similarly, X will be U or V with 
    rows corresponding to dropped values of y already dropped
    Thus, the solution can be undertermined.
    """
    nRow = X.shape[0]
    if(nRow == 0):
        return( np.zeros(nCol) )

    y_use = y[use]
    X_use = X[use,:]
    
    nRow = X_use.shape[0]
    nCol = X_use.shape[1]

    if(nRow < nCol):
        ans = singular_lsqr(X_use, y_use)
        return(ans)

    Xt = X_use.transpose()
    Xt_X = Xt @ X_use
    ans = np.linalg.solve(Xt_X, Xt @ y_use)
    return(ans)
    
    
    
def singular_lsqr(X, y):
    """
    Underdetermined least squares
    
    This provides *a* least squares solution
    when the system is underdetermined, i.e. nrow(X) < ncol(X)
    
    This is done by setting the excess trailing 
    coefficients to zero, i.e. if nrow(X) = 3 and ncol(X) = 5, 
    then the 4th and 5th coefficients will be set to 0
    """
    nRow = X.shape[0]
    nCol = X.shape[1]
    
    # In case a degenerate matrix is passed in
    if(nRow == 0):
        return(np.zeros(nCol))
    # Chops out extra columns
    X_skinny = X[:,range(nRow)]
    # Standard least squares at this point
    ans = lsqr(X_skinny, y, range(nRow))
    # Add zeros at the end
    ans = np.append(ans, np.zeros(nCol - nRow))
    return(ans)
    
    
def updateU(U, V, data, mask):
    nRow = U.shape[0]
    for i in range(0,nRow):
        this_y = data[i,:]
        this_mask = mask[i,:]
        new_row = lsqr(V, this_y, this_mask)
        U[i,:] = new_row
    return(U)
    
    
def updateV(U, V, data, mask):
    nRow = V.shape[0]
    for j in range(0, nRow):
        this_y = data[:,j]
        this_mask = mask[:,j]
        new_row = lsqr(U, this_y, this_mask)
        V[j,:] = new_row
    return(V)
    
def pcaLoss(U, V, data):
    theta = U @ V.transpose()
    err = data - theta
    err = err.flatten()
    ans = np.nanmean( np.square(err) )
    return(ans)