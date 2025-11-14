# Quantile prediction calibration error

Quantile prediction calibration error

## Usage

``` r
quantile_prediction_error(y, Q_hat, prob_level, na.rm = FALSE)
```

## Arguments

- y:

  Vector of observations.

- Q_hat:

  Vector of predicted quantiles at probability level `prob_level`.

- prob_level:

  Probability level of the predicted quantile.

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

## Value

The quantile prediction error calibration metric.

## Examples

``` r
quantile_prediction_error(c(2.3, 4.2, 1.8), c(2.9, 5.6, 2.7), prob_level=0.8)
#> [1] 0.8660254
```
