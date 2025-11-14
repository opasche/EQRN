# Square loss

Square loss

## Usage

``` r
square_loss(y, y_hat)
```

## Arguments

- y:

  Vector of observations or ground-truths.

- y_hat:

  Vector of predictions.

## Value

The vector of square errors between `y` and `y_hat`.

## Examples

``` r
square_loss(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
#> [1] 0.01 0.16 0.01
```
