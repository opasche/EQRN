# Multilevel 'proportion_below'

Multilevel version of
[`proportion_below()`](https://opasche.github.io/EQRN/reference/proportion_below.md).

## Usage

``` r
multilevel_prop_below(
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

  Whether to name the output proportions (bool).

## Value

A vector of length `length(proba_levels)` giving the proportion of
observations below the predictions (`Pred_Q`) at each probability level.
If `give_names` is `TRUE`, the output vector is named
`paste0(prefix, "propBelow_q", proba_levels)`.
