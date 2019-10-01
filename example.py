from scipy.special import ndtr
import xpcapy
import time
import numpy as np
import argparse
import logging

if __name__ == "__main__":
    start_main = time.time()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", action="store", type=int, default=2,
            help="Desired rank of decomposition")
    parser.add_argument("--filename", action="store", 
            help="Full path of data in csv format")

    args = parser.parse_args()
    r = args.rank

    # If data was provided
    if args.filename:
        data = np.genfromtxt(args.filename, delimiter=',')
    # Simulate data since no filename was given
    else:
        # Pick arbitrary dimensions and rank
        m=5000
        n=40
        print ("Generating %i x %i matrix of rank %i" % (m,n,r))
        simulation = xpcapy.simulate.simulate_data(mrows=m, ncols=n, rank=r, prop_binary=0.2)
        data = simulation.data_matrix

    # Decompose data
    decomper = xpcapy.xpca.XPCA(rank=r)
    solver = 'alt-newt'
    imputation = 'median'
    print ("Using %s solve and %s imputation" % (solver, imputation))

    start = time.time()
    d = decomper.fit(data, method=solver, imputation=imputation, cdf=ndtr)
    end = time.time()
    print ("Elapsed time for decomposition: %f sec" % (end-start))

    print ("Saving off solved theta and final fitted matrices...")
    np.savetxt("theta.csv",d.theta, delimiter=',')
    np.savetxt("fitted.csv", d.fitted, delimiter=',')
    
    end_main = time.time()
    print ("Complete. Elapsed time %f sec" % (end_main-start_main))
