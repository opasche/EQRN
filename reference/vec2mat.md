# Convert a vector to a matrix

Convert a vector to a matrix

## Usage

``` r
vec2mat(v, axis = c("col", "row"))
```

## Arguments

- v:

  Vector.

- axis:

  One of `"col"` (default) or `"row"`.

## Value

The vector `v` as a matrix. If `axis=="col"` (default) the column vector
`v` is returned as a `length(v)` times `1` matrix. If `axis=="row"`, the
vector `v` is returned as a transposed `1` times `length(v)` matrix.

## Examples

``` r
vec2mat(c(2, 7, 3, 8), "col")
#>      [,1] [,2] [,3] [,4]
#> [1,]    2    7    3    8
```
