# R squared

The coefficient of determination, often called R squared, is the
proportion of data variance explained by the predictions.

## Usage

``` r
R_squared(y, y_hat, na.rm = FALSE)
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

The R squared of the predictions `y_hat` for `y`.

## Examples

``` r
R_squared(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
#> [1] 0.9438669
```
