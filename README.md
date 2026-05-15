# Monte Carlo Demos

This project is an implementation of two widely-used Monte Carlo methods, one for scalars and one for vectors, the acceptance-rejection method and the Gibbs sampler.  It is done to demonstrate that I have some knowledge of these methods on the code level.  This is not intended to be used, as I am sure there are more efficient implementations.

## Acceptance rejection

This is kind of the "gateway" into MCS, but should only be used for scalar values.  It is good for complex distributions, referred to as f(X), where we do not have an easily computable functional inverse to sample the X values.  We choose a proposal distribution g, which we can sample from, and a constant M, which ensures that $f(X) < M g(X)$.  Here is how it works:

1) Sample $x \sim g$.
2) Sample $y \sim \textbf{Uniform}(0,1)$
3) If $y < f(x)/(M g(x))$, accept $x$, else reject.  Repeat (1)

Code is available in `acceptance_rejection.py`.

## Gibbs sampling

Gibbs sampling is more suitable for sampling vector values.  We are assuming a multinomial Gaussian.  The math for this can be verified online.  We start with a mean $\mu$, a random vector $x$, and a covariance matrix $\Sigma$.  We begin by computing variances ${\sigma_i}^2$ from $\Sigma$.  Then, at each iteration, we compute values $\mu_i$ from elements of $\Sigma$, $x$ and $\mu$ which are not indexed by $i$.  We can then sample a new value of $x_i$ from a Normal distribution parametrized by $\mu_i$ and $\sigma_i$.  In this way, we build a new vector $x$.

## Running

All you need to do to run the code is to type:
```
python acceptance_rejection.py
```
or:
```
python gibbs_sampler.py
```

