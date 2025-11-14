# Last element of a vector

Returns the last element of the given vector in the most efficient way.

## Usage

``` r
last_elem(x)
```

## Arguments

- x:

  Vector.

## Value

The last element in the vector `x`.

## Details

The last element is obtained using `x[length(x)]`, which is done in
`O(1)` and faster than, for example, any of `Rcpp::mylast(x)`,
`tail(x, n=1)`, `dplyr::last(x)`, `x[end(x)[1]]]`, and `rev(x)[1]`.

## Examples

``` r
last_elem(c(2, 6, 1, 4))
#> [1] 4
```
