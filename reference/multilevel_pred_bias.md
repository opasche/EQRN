# Multilevel prediction bias

Multilevel version of
[`prediction_bias()`](https://opasche.github.io/EQRN/reference/prediction_bias.md).

## Usage

``` r
multilevel_pred_bias(
  True_Q,
  Pred_Q,
  proba_levels,
  square_bias = FALSE,
  prefix = "",
  na.rm = FALSE,
  give_names = TRUE
)
```

## Arguments

- True_Q:

  Matrix of size `n_obs` times `proba_levels`, whose columns are the
  vectors of ground-truths at each `proba_levels` and each row
  corresponds to an observation or realisation.

- Pred_Q:

  Matrix of the same size as `True_Q`, whose columns are the predictions
  at each `proba_levels` and each row corresponds to an observation or
  realisation.

- proba_levels:

  Vector of probability levels at which the predictions were made. Must
  be of length `ncol(Pred_Q)`.

- square_bias:

  Whether to return the square bias (bool); defaults to `FALSE`.

- prefix:

  A string prefix to add to the output's names (if `give_names` is
  `TRUE`).

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

- give_names:

  Whether to name the output MSEs (bool).

## Value

A vector of length `length(proba_levels)` giving the (square) bias of
each columns of predictions in `Pred_Q` for the respective `True_Q`. If
`give_names` is `TRUE`, the output vector is named
`paste0(prefix, "MSE_q", proba_levels)`.
