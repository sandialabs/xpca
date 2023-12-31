## Table of Contents

* [Background](#background) 
* [Install](#install) 
* [Example](#example)

## Background 
XPCA factors an observed data matrix X (m rows x n columns) into
two factors, A (m x k) and B (n x k) where rank = k. Each row in X corresponds
to an observed element, and each column corresponds to a feature of the observed
element. These columns can be of mixed variable types - continuous, count,
binary, etc. 

`xpcapy` is the python implementation of xpca. There is also an R version of
this code available [here](https://gitlab.com/xpca/xpcar)

XPCA implements 3 capabilities: xpca, pca, and coca. 

### coca 
`coca` is from work done by:

1. Fang Han and Han Liu. Semiparametric principal component analysis. In
NIPS’12: Proceedings of the 26th Annual Conference on Neural Information
Pro- cessing Systems, pages 171–179, 2012.  URL
http://papers.nips.cc/paper/4809-semiparametric-principal-component-analysis.

2. Bernhard Egger, Dinu Kaufmann, Sandro Sch ̈onborn, Volker Roth, and Thomas
Vetter. Copula eigenfaces — semiparametric principal component analysis for
facial appearance modeling.  In VISIGRAPP’16: Proceedings of the 11th Joint
Conference on Computer Vi- sion, Imaging and Computer Graphics Theory and
Applications, pages 50–58. SciTePress, 2016. 10.5220/0005718800480056.


### xpca 
`xpca` is Sandia-built work done by Cliff Anderson-Bergman, Tamara
Kolda, and Kina Kincher-Winoto.  Paper is available on arXiv:

[C. Anderson-Bergman, T. G. Kolda, K. Kincher-Winoto. XPCA: Extending PCA for a
Combination of Discrete and Continuous Variables. arXiv:1808.07510,
2018.](https://arxiv.org/abs/1808.07510)

### pca 
`pca` is the well-known algorithm. For reference: 

Michael E. Tipping and Christopher M. Bishop. Probabilistic principal component
analysis. Journal of the Royal Statistical Society: Series B (Statistical
Methodology), 61(3):611– 622, August 1999. 10.1111/1467-9868.00196.

## Install

`xpcapy` requires:
* Python 3 (tested on 3.6) 
* [NumPy](http://scipy.org/index.html) 1.11.0
* [SciPy](http://scipy.org/index.html) 0.17.1

These can be installed by running the following from the command line: 
```bash 
$ pip install numpy 
$ pip install scipy 
```

To install `xpcapy`:

Download this repo and run the following command from the terminal: 

```bash 
$ cd /path/to/xpcapy
$ python setup.py install 
``` 

## Example

Refer to `example.py` for an example as to how to simulate data and read in a
file. This script will run xpca and save off theta and final fitted matrix to a
file. 

Usage: 
```
# Just run it to simulate data
$ python3 example.py
# Or to read in a file of your own
$ python3 example.py --filename example_data.csv --rank 3 
``` 

Both variations of
the run (simulation or reading in your own data) will output a `theta.csv` and
`fitted.csv`

The included `example_data.csv` is an exemplar data set since it includes
continuous, binary, and count data.

## Questions?
Email xpca@sandia.gov if you have any questions.

