'''
python3 watch_file.py -p1 python3 acceptance_rejection.py -d .
'''
import numpy as np

MIXTURE_WEIGHTS=[0.2, 0.7, 0.1]
MIXTURE_MEANS=[-2.0, 3.0, -1.0]
MIXTURE_STDS=[0.5, 1.5, 0.1]  
N_SAMPLES=1000
RNG=np.random.default_rng()

PROPOSAL_MEAN=0.0
PROPOSAL_STD=2.0
PROPOSAL_M=2.0

def gaussian_sample(proposalMean: float = PROPOSAL_MEAN,
                    proposalStd: float = PROPOSAL_STD,
                    rng: np.random._generator.Generator = RNG,
                    nSamples: int = N_SAMPLES,
                   ) -> np.ndarray:
    '''
    Sample nSamples from a 1D Gaussian distribution defined by proposalMean and proposalStd.
    '''
    return rng.normal(loc=proposalMean,scale=proposalStd,size=nSamples)


def gaussian_pdf(X: np.array,
                 mean: float = PROPOSAL_MEAN,
                 std: float = PROPOSAL_STD,
                ) -> np.array:
    '''
    Pdf for a 1D gaussian with mean and std.  Works on nd.arrays.
    '''
    denom=(1 / (std * np.sqrt(2 * np.pi)))
    diff=(X - mean)
    
    return denom*np.exp(-0.5*(diff/std)**2)


def gaussian_mixture_1d_pdf(X: np.array,
                            means: np.array = MIXTURE_MEANS,
                            stds: np.array = MIXTURE_STDS,
                            weights: np.array = MIXTURE_WEIGHTS,
                           ) -> np.array:
    '''
    Pdf for a 1D Gaussian mixture with means, stds, and weights.  Works on np.arrays.
    '''
    pdfs=np.array([gaussian_pdf(X,mean=mean,std=std) 
                   for (mean,std) in zip(means,stds)])
    mixturePdfs=pdfs.T*weights
    sums=np.sum(mixturePdfs,axis=1)

    return sums
    

def uniform(nSamples: int = N_SAMPLES,
            rng: np.random._generator.Generator = RNG,
           ) -> np.array:
    '''
    Sample nSamples from uniform distribution.
    '''
    return rng.uniform(size=nSamples)


def accept_reject(proposal_sample_f: callable = gaussian_sample,
                  proposal_pdf_f: callable = gaussian_pdf,
                  mixture_1d_pdf_f: callable = gaussian_mixture_1d_pdf,
                  uniform_f: callable = uniform,
                  m: float = PROPOSAL_M,
                  nSamples: int = N_SAMPLES,
                 ) -> np.array:
    '''
    Implements the accept/reject algorithm for sampling from a 1D distribution, in this case
    a Gaussian mixture.

    * proposal_sample_f samples the proposal distrubtion, a 1D Gaussian function.
    * proposal_pdf_f is the PDF of the proposal distribution, g(X).
    * mixture_1d_pdf_f is the PDF of the target distribution, f(X).
    * uniform_f samples a uniform distribution.
    * m is a scaling constant
    * nSamples is the number of samples we take.
    '''
    # proposalSample ~ N(mu,sigma)
    proposalSample=proposal_sample_f(nSamples=nSamples)
    # uniformSample ~ Uniform(0,1)
    uniformSample=uniform_f(nSamples=nSamples)
    # g(proposalSample)
    proposalPdf=proposal_pdf_f(proposalSample)
    # f(proposalSample)
    mixturePdf=mixture_1d_pdf_f(proposalSample)
    # g(proposalSample)/(m*f(proposalSample))
    likelihoods=mixturePdf/(m*proposalPdf) 
    # for x in uniformSample: hit is true if x < g(proposalSample)/(m*f(proposalSample)), else
    # false
    hits=uniformSample<likelihoods
    # take every proposalSample that satisfies the hits criterion
    filteredSample=np.array([sample for (sample,hit) in zip(proposalSample,hits) if hit])

    return filteredSample
    

if __name__=='__main__':

    print(accept_reject())

