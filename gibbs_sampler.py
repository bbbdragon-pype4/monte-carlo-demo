'''
python3 watch_file.py -p1 python3 gibbs_sampler.py -d .
'''
import numpy as np
from itertools import product
from typing import Dict,Tuple

MEAN=np.array([0.1,2.0,-1.0])
SIGMA=np.array([[2.0, 0.8, 0.5],
                [0.8, 1.5, 0.9],
                [0.5, 0.9, 3.0]])    
N_SAMPLES=10
BURN=3

def compute_std_for_i(sigmaII: float,
                      sigmaINotI: np.array,
                      sigmaNotINotIInv: np.array,
                      sigmaNotII: np.array,
                     ) -> float:
    '''
    Takes several values and computes the standard deviation for the index i.  These values are:

    1) sigmaII - the on-diagonal element for the variance.
    2) sigmaINotI - ith row of the covariance matrix, with the i index deleted.
    3) sigmaNotINotIInv - inverse of the covariance matrix, with columns and rows for the i 
    index deleted.
    4) sigmaNotII - ith column of the covariance matrix, with the i index deleted.

    Here, we are estimating the variance from the covariance matrix in terms of the on-diagonal
    element, taking out contributions from other correlations.
    '''
    variance=sigmaII - sigmaINotI @ sigmaNotINotIInv @ sigmaNotII
    std=np.sqrt(variance)

    return std


def compute_covariance_vals(sigma: np.array=SIGMA) -> Dict:
    '''
    This function computes a dictionary of matrices and vectors computed from the covariance
    matrix.  In the notation:

    1) I in the first position indicates a row vector, I in the second position indicates a
    column vector.
    2) NotI in the first position indicates a column vector with the i-th element deleted.  
    NotI in the second position indicates a row vector with the i-th element deleted.
    3) II indicates the i-th on-diagonal element.
    4) Inv indicates the inverse of the matrix.
    5) std is the updated standard deviation.
    '''
    dim=sigma.shape[0]
    covarianceVals={}

    for i in range(sigma.shape[0]):

        sigmaNotINotI=np.array([[sigma[j,k] for k in range(dim) if k != i] 
                                 for j in range(dim) if j != i])
        covarianceValsForI={'sigmaII':sigma[i,i],
                            'sigmaINotI':np.array([sigma[i,j] for j in range(dim) if j != i]),
                            'sigmaNotII':np.array([sigma[j,i] for j in range(dim) if j != i]),
                            'sigmaNotINotIInv':np.linalg.inv(sigmaNotINotI),
                           }
        covarianceValsForI['std']=compute_std_for_i(**covarianceValsForI)
        covarianceVals[i]=covarianceValsForI

    return covarianceVals
    

def compute_mean_for_i(covarianceVals: Dict,
                       i: int,
                       x: np.array,
                       mean: np.array = MEAN,
                      ) -> float: 
    '''
    This updates the i-th element in the mean vector using the mean vector, a sampled vector x,
    and covariance matrix sigma.  The equation implemented is:

    m_i' = m_i + sigmaINotI^T sigmaNotINotIInv (xNotI - mNotI)

    In essence, we are updating the mean with contributions from non-i elements of x and sigma.
    We can see this by the following:

    1) ||xNotI - mNotI|| is proportional to the magnitude of the change to m_i.
    2) sigmaNotI^T (xNotI - mNotI) weights the changes by covariances, and sums the result.
    3) sigmaNotINotIInv normalizes the update.
    '''
    d=mean.shape[0]
    # mean and x not including the i-th element.
    meanNotI=np.array([mean[j] for j in range(d) if j != i])
    xNotI=np.array([x[j] for j in range(d) if j != i])
    # difference vector
    diff=xNotI-meanNotI
    # unpacking necessary vectors and matrices associated with sigma.
    covarianceValsForI=covarianceVals[i]
    sigmaINotI=covarianceValsForI['sigmaINotI']
    sigmaNotINotIInv=covarianceValsForI['sigmaNotINotIInv']
    
    # The updated mean.
    return mean[i] + sigmaINotI @ sigmaNotINotIInv @ diff


def sample_x(x: np.array,
             covarianceVals: Dict,
             mean: np.array = MEAN,
            ) -> Tuple[np.array,np.array]:
    '''
    Runs a single iteration of the Gibbs sampler:
    
    1) Computes a new mean vector.
    2) Grabs the standard deviation from covarianceVals.
    3) Samples a new vector from the Normal distribution parametrized by the mean and 
    standard deviation.  Because we are sampling from a Normal distribution, we sample
    each element of the new vector from a scalar Normal distribution.
    '''
    d=mean.shape[0] 
    conditionalMean=np.array([compute_mean_for_i(i=i,
                                                 covarianceVals=covarianceVals,
                                                 x=x,
                                                 mean=mean) for i in range(d)]) # (1)
    conditionalStds=np.array([covarianceVals[i]['std'] for i in range(d)]) # (2)
    conditionalX=np.random.normal(conditionalMean,conditionalStds) # (3)

    return conditionalX,conditionalMean



def run_gibbs(mean: np.array = MEAN,
              sigma: np.array = SIGMA,
              numSamples: int = N_SAMPLES,
              burn: int = BURN,
             ) -> np.array:
    '''
    Main iteration:

    1) Set up matrices containing the burns and the samples.
    2) Compute the scalars, vectors, and matrices associated with sigma.
    3) Iterate through burn and number of samples:
      a) Compute x and mean.
      b) Set it as a column of samples.
    4) Throw out the burn samples.
    '''
    d=mean.shape[0]
    x=np.zeros(d)
    total=burn+numSamples
    samples=np.zeros((d,total)) # (1)
    covarianceVals=compute_covariance_vals(sigma) # (2)

    for l in range(total): # (3)

        x,mean=sample_x(x=x,
                        covarianceVals=covarianceVals,
                        mean=mean)
        samples[:,l]=x

    return samples[:,burn:] # (4)


if __name__=='__main__':

    samples=run_gibbs()

    print(samples)
