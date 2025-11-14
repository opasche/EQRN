# Prediction bias

Prediction bias

## Usage

``` r
prediction_bias(y, y_hat, square_bias = FALSE, na.rm = FALSE)
```

## Arguments

- y:

  Vector of observations or ground-truths.

- y_hat:

  Vector of predictions.

- square_bias:

  Whether to return the square bias (bool); defaults to `FALSE`.

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

## Value

The (square) bias of the predictions `y_hat` for `y`.

## Examples

``` r
prediction_bias(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
#> [1] 0.06666667
```
