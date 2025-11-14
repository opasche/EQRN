# Multilevel 'quantile_prediction_error'

Multilevel version of
[`quantile_prediction_error()`](https://opasche.github.io/EQRN/reference/quantile_prediction_error.md).

## Usage

``` r
multilevel_q_pred_error(
  y,
  Pred_Q,
  proba_levels,
  prefix = "",
  na.rm = FALSE,
  give_names = TRUE
)
```

## Arguments

- y:

  Vector of observations.

- Pred_Q:

  Matrix of of size `length(y)` times `proba_levels`, whose columns are
  the quantile predictions at each `proba_levels` and each row
  corresponds to an observation or realisation.

- proba_levels:

  Vector of probability levels at which the predictions were made. Must
  be of length `ncol(Pred_Q)`.

- prefix:

  A string prefix to add to the output's names (if `give_names` is
  `TRUE`).

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

- give_names:

  Whether to name the output errors (bool).

## Value

A vector of length `length(proba_levels)` giving the quantile prediction
error calibration metrics between each column of `Pred_Q` and the
observations. If `give_names` is `TRUE`, the output vector is named
`paste0(prefix, "qPredErr_q", proba_levels)`.
