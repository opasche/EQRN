# Multilevel 'quantile_exceedance_proba_error'

Multilevel version of
[`quantile_exceedance_proba_error()`](https://opasche.github.io/EQRN/reference/quantile_exceedance_proba_error.md).

## Usage

``` r
multilevel_exceedance_proba_error(
  Probs,
  proba_levels = NULL,
  return_years = NULL,
  type_probs = c("cdf", "exceedance"),
  prefix = "",
  na.rm = FALSE,
  give_names = TRUE
)
```

## Arguments

- Probs:

  Matrix, whose columns give, for each `proba_levels`, the predicted
  probabilities to exceed or be smaller than a fixed quantile.

- proba_levels:

  Vector of probability levels of the quantiles.

- return_years:

  The probability levels can be given in term or return years instead.
  Only used if `proba_levels` is not given.

- type_probs:

  Whether the predictions are the `"cdf"` (default) or `"exceedance"`
  probabilities.

- prefix:

  A string prefix to add to the output's names (if `give_names` is
  `TRUE`).

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

- give_names:

  Whether to name the output errors (bool).

## Value

A vector of length `length(proba_levels)` giving the
[`quantile_exceedance_proba_error()`](https://opasche.github.io/EQRN/reference/quantile_exceedance_proba_error.md)
calibration metric of each column of `Probs` at the corresponding
`proba_levels`. If `give_names` is `TRUE`, the output vector is named
`paste0(prefix, "exPrErr_q", proba_levels)` (or
`paste0(prefix, "exPrErr_", return_years,"y")` if `return_years` are
given instead of `proba_levels`).
