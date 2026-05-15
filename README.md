# Monte Carlo Demos

This project is an implementation of two widely-used Monte Carlo methods, one for scalars and one for vectors, the acceptance-rejection method and the Gibbs sampler.  It is done to demonstrate that I have some knowledge of these methods on the code level.  This is not intended to be used, as I am sure there are more efficient implementations.

## Acceptance rejection

This is kind of the "gateway" into MCS, but should only be used for scalar values.  It is good for complex distributions, referred to as f(X), where we do not have an easily computable functional inverse to sample the X values.  We choose a proposal distribution g, which we can sample from, and a constant M, which ensures that f(X) < M * g(X).  Here is how it works:

