# Covariate lagged replication for temporal dependence

Covariate lagged replication for temporal dependence

## Usage

``` r
lagged_features(X, max_lag, drop_present = TRUE)
```

## Arguments

- X:

  Covariate matrix.

- max_lag:

  Integer giving the maximum lag (i.e. the number of temporal dependence
  steps).

- drop_present:

  Whether to drop the "present" features (bool).

## Value

Matrix with the original columns replicated, and shifted by `1:max_lag`
if `drop_present==TRUE` (default) or by `0:max_lag` if
`drop_present==FALSE`.

## Examples

``` r
lagged_features(matrix(seq(20), ncol=2), max_lag=3, drop_present=TRUE)
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    3   13    2   12    1   11
#> [2,]    4   14    3   13    2   12
#> [3,]    5   15    4   14    3   13
#> [4,]    6   16    5   15    4   14
#> [5,]    7   17    6   16    5   15
#> [6,]    8   18    7   17    6   16
#> [7,]    9   19    8   18    7   17
```
