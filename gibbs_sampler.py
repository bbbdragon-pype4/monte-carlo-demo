'''
python3 watch_file.py -p1 python3 gibbs_sampler.py -d .
'''
import numpy as np
from itertools import product
from typing import Dict

MEAN=np.array([0.1,2.0,-1.0])
SIGMA=np.array([[2.0, 0.8, 0.5],
                [0.8, 1.5, 0.9],
                [0.5, 0.9, 3.0]])    
N_SAMPLES=10
BURN=3

def compute_variance_for_i(sigmaII: float,
                           sigmaINotI: np.array,
                           sigmaNotINotIInv: np.array,
                           sigmaNotII: np.array,
                          ):

    variance=sigmaII - sigmaINotI @ sigmaNotINotIInv @ sigmaNotII
    std=np.sqrt(variance)

    return std


def compute_covariance_vals(sigma: np.array=SIGMA):

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
        covarianceValsForI['std']=compute_variance_for_i(**covarianceValsForI)
        covarianceVals[i]=covarianceValsForI

    return covarianceVals
    

def compute_mean_for_i(covarianceVals: Dict,
                       i: int,
                       x: np.array,
                       mean: np.array = MEAN,
                      ): 

    d=mean.shape[0]
    meanNotI=np.array([mean[j] for j in range(d) if j != i])
    xNotI=np.array([x[j] for j in range(d) if j != i])
    diff=xNotI-meanNotI
    covarianceValsForI=covarianceVals[i]
    sigmaINotI=covarianceValsForI['sigmaINotI']
    sigmaNotINotIInv=covarianceValsForI['sigmaNotINotIInv']
    
    return mean[i] + sigmaINotI @ sigmaNotINotIInv @ diff


def sample_x(x: np.array,
             covarianceVals: Dict,
             mean: np.array = MEAN,
            ):

    d=mean.shape[0]
    conditionalMean=np.array([compute_mean_for_i(i=i,
                                                 covarianceVals=covarianceVals,
                                                 x=x,
                                                 mean=mean) for i in range(d)])
    conditionalStds=np.array([covarianceVals[i]['std'] for i in range(d)])
    conditionalX=np.random.normal(conditionalMean,conditionalStds)

    return conditionalX,conditionalMean



def run_gibbs(mean: np.array = MEAN,
              sigma: np.array = SIGMA,
              numSamples: int = N_SAMPLES,
              burn: int = BURN,
             ):

    d=mean.shape[0]
    x=np.zeros(d)
    total=burn+numSamples
    samples=np.zeros((d,total))
    covarianceVals=compute_covariance_vals(sigma)

    for l in range(total):

        x,mean=sample_x(x=x,
                        covarianceVals=covarianceVals,
                        mean=mean)
        samples[:,l]=x

    return samples[:,burn:]


if __name__=='__main__':

    samples=run_gibbs()

    print(samples)
