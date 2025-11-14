# Predict unconditional extreme quantiles using peaks over threshold

Predict unconditional extreme quantiles using peaks over threshold

## Usage

``` r
predict_unconditional_quantiles(interm_lvl, quantiles = c(0.99), Y, ntest = 1)
```

## Arguments

- interm_lvl:

  Probability level at which the empirical quantile should be used as
  the intermediate threshold.

- quantiles:

  Probability levels at which to predict the extreme quantiles.

- Y:

  Vector of ("training") observations.

- ntest:

  Number of "test" observations.

## Value

Named list containing:

- predictions:

  matrix of dimension `ntest` times `length(quantiles)` containing the
  estimated extreme quantile at levels `quantile`, repeated `ntest`
  times,

- pars:

  matrix of dimension `ntest` times `2` containing the two GPD parameter
  MLEs, repeated `ntest` times.

- threshold:

  The threshold for the peaks-over-threshold GPD model. It is the
  empirical quantile of `Y` at level `interm_lvl`, i.e.
  `stats::quantile(Y, interm_lvl)`.
