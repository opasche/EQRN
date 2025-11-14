# Generalized Pareto likelihood loss

Generalized Pareto likelihood loss

## Usage

``` r
loss_GPD(
  sigma,
  xi,
  y,
  rescaled = TRUE,
  interm_lvl = NULL,
  return_vector = FALSE
)
```

## Arguments

- sigma:

  Value(s) for the GPD scale parameter.

- xi:

  Value(s) for the GPD shape parameter.

- y:

  Vector of observations

- rescaled:

  Whether y already is a vector of excesses (TRUE) or needs rescaling
  (FALSE).

- interm_lvl:

  Probability level at which the empirical quantile should be used as
  the intermediate threshold to compute the excesses, if
  `rescaled==FALSE`.

- return_vector:

  Whether to return the the vector of GPD losses for each observation
  instead of the negative log-likelihood (average loss).

## Value

GPD negative log-likelihood of the GPD parameters over the sample of
observations.
