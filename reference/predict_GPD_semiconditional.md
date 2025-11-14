# Predict semi-conditional extreme quantiles using peaks over threshold

Predict semi-conditional extreme quantiles using peaks over threshold

## Usage

``` r
predict_GPD_semiconditional(
  Y,
  interm_lvl,
  thresh_quantiles,
  interm_quantiles_test = thresh_quantiles,
  prob_lvls_predict = c(0.99)
)
```

## Arguments

- Y:

  Vector of ("training") observations.

- interm_lvl:

  Probability level at which the empirical quantile should be used as
  the intermediate threshold.

- thresh_quantiles:

  Numerical vector of the same length as `Y` representing the varying
  intermediate threshold on the train set.

- interm_quantiles_test:

  Numerical vector of the same length as `Y` representing the varying
  intermediate threshold used for prediction on the test set.

- prob_lvls_predict:

  Probability levels at which to predict the extreme semi-conditional
  quantiles.

## Value

Named list containing:

- predictions:

  matrix of dimension `length(interm_quantiles_test)` times
  `length(prob_lvls_predict)` containing the estimated extreme quantile
  at levels `quantile`, for each `interm_quantiles_test`,

- pars:

  matrix of dimension `ntest` times `2` containing the two GPD parameter
  MLEs, repeated `length(interm_quantiles_test)` times.
