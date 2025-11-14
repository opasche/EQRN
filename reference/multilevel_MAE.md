# Multilevel quantile MAEs

Multilevel version of
[`mean_absolute_error()`](https://opasche.github.io/EQRN/reference/mean_absolute_error.md).

## Usage

``` r
multilevel_MAE(
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

  Whether to name the output MAEs (bool).

- sd:

  Whether to return the absolute error standard deviation (bool).

## Value

A vector of length `length(proba_levels)` giving the mean absolute
errors between each respective columns of `True_Q` and `Pred_Q`. If
`give_names` is `TRUE`, the output vector is named
`paste0(prefix, "MAE_q", proba_levels)`. If `sd==TRUE` a named list is
instead returned, containing the `"MAEs"` described above and `"SDs"`,
their standard deviations.
