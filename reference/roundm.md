# Mathematical number rounding

This function rounds numbers in the mathematical sense, as opposed to
the base `R` function [`round()`](https://rdrr.io/r/base/Round.html)
that rounds 'to the even digit'.

## Usage

``` r
roundm(x, decimals = 0)
```

## Arguments

- x:

  Vector of numerical values to round.

- decimals:

  Integer indicating the number of decimal places to be used.

## Value

A vector containing the entries of `x`, rounded to `decimals` decimals.

## Examples

``` r
roundm(2.25, 1)
#> [1] 2.3
```
