# Prediction residual variance

Prediction residual variance

## Usage

``` r
prediction_residual_variance(y, y_hat, na.rm = FALSE)
```

## Arguments

- y:

  Vector of observations or ground-truths.

- y_hat:

  Vector of predictions.

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

## Value

The residual variance of the predictions `y_hat` for `y`.

## Examples

``` r
prediction_residual_variance(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
#> [1] 0.08333333
```
