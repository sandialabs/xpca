from scipy.special import ndtr
import xpcapy
import time
import numpy as np

start = time.time()

import logging
logging.getLogger().setLevel(logging.INFO)

m=5000
n=40
r=5
print ("Generating %i x %i matrix of rank %i" % (m,n,r))
# Read in data
data = xpcapy.simulate.simulate_data(mrows=m, ncols=n, rank=r, prop_binary=0.2)
decomper = xpcapy.xpca.XPCA(rank=r)
solver = 'alt-newt'
imputation = 'median'
print ("Using %s solve and %s imputation" % (solver, imputation))

d = decomper.fit(data.data_matrix, method=solver, imputation=imputation, cdf=ndtr)

np.savetxt("theta.csv",d.theta, delimiter=',')
np.savetxt("fitted.csv", d.fitted, delimiter=',')

end = time.time()

print ("elapsed time: %f sec" % ((end-start)))
