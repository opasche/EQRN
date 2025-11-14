# Multilevel quantile MSEs

Multilevel version of
[`mean_squared_error()`](https://opasche.github.io/EQRN/reference/mean_squared_error.md).

## Usage

``` r
multilevel_MSE(
  True_Q,
  Pred_Q,
  proba_levels,
  prefix = "",
  na.rm = FALSE,
  give_names = TRUE,
  sd = FALSE
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

- prefix:

  A string prefix to add to the output's names (if `give_names` is
  `TRUE`).

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

- give_names:

  Whether to name the output MSEs (bool).

- sd:

  Whether to return the squared error standard deviation (bool).

## Value

A vector of length `length(proba_levels)` giving the mean square errors
between each respective columns of `True_Q` and `Pred_Q`. If
`give_names` is `TRUE`, the output vector is named
`paste0(prefix, "MSE_q", proba_levels)`. If `sd==TRUE` a named list is
instead returned, containing the `"MSEs"` described above and `"SDs"`,
their standard deviations.
