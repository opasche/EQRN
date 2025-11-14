# Proportion of observations below conditional quantile vector

Proportion of observations below conditional quantile vector

## Usage

``` r
proportion_below(y, Q_hat, na.rm = FALSE)
```

## Arguments

- y:

  Vector of observations.

- Q_hat:

  Vector of predicted quantiles.

- na.rm:

  A logical value indicating whether `NA` values should be stripped
  before the computation proceeds.

## Value

The proportion of observation below the predictions.

## Examples

``` r
proportion_below(c(2.3, 4.2, 1.8), c(2.9, 5.6, 1.7))
#> [1] 0.6666667
```
