# Mean absolute error

Mean absolute error

## Usage

``` r
mean_absolute_error(
  y,
  y_hat,
  return_agg = c("mean", "sum", "vector"),
  na.rm = FALSE
)
```

## Arguments

- y:

  Vector of observations or ground-truths.

- y_hat:

  Vector of predictions.

- return_agg:

  Whether to return the `"mean"` (default), `"sum"`, or `"vector"` of
  errors.

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

## Value

The mean (or total or vectorial) absolute error between `y` and `y_hat`.

## Examples

``` r
mean_absolute_error(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
#> [1] 0.2
```
