# Quantile loss

Quantile loss

## Usage

``` r
quantile_loss(
  y,
  y_hat,
  q,
  return_agg = c("mean", "sum", "vector"),
  na.rm = FALSE
)
```

## Arguments

- y:

  Vector of observations.

- y_hat:

  Vector of predicted quantiles at probability level `q`.

- q:

  Probability level of the predicted quantile.

- return_agg:

  Whether to return the `"mean"` (default), `"sum"`, or `"vector"` of
  losses.

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

## Value

The mean (or total or vectorial) quantile loss between `y` and `y_hat`
at level `q`.

## Examples

``` r
quantile_loss(c(2.3, 4.2, 1.8), c(2.9, 5.6, 2.7), q=0.8)
#> [1] 0.1933333
```
