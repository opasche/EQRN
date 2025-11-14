# Quantile exceedance probability prediction calibration error

Quantile exceedance probability prediction calibration error

## Usage

``` r
quantile_exceedance_proba_error(
  Probs,
  prob_level = NULL,
  return_years = NULL,
  type_probs = c("cdf", "exceedance"),
  na.rm = FALSE
)
```

## Arguments

- Probs:

  Predicted probabilities to exceed or be smaller than a fixed quantile.

- prob_level:

  Probability level of the quantile.

- return_years:

  The probability level can be given in term or return years instead.
  Only used if `prob_level` is not given.

- type_probs:

  Whether the predictions are the `"cdf"` (default) or `"exceedance"`
  probabilities.

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

## Value

The calibration metric for the predicted probabilities.

## Examples

``` r
quantile_exceedance_proba_error(c(0.1, 0.3, 0.2), prob_level=0.8)
#> [1] -0.6
```
