# Insert value in vector

Insert value in vector

## Usage

``` r
vector_insert(vect, val, ind)
```

## Arguments

- vect:

  A 1-D vector.

- val:

  A value to insert in the vector.

- ind:

  The index at which to insert the value in the vector, must be an
  integer between `1` and `length(vect) + 1`.

## Value

A 1-D vector of length `length(vect) + 1`, with `val` inserted at
position `ind` in the original `vect`.

## Examples

``` r
vector_insert(c(2, 7, 3, 8), val=5, ind=3)
#> [1] 2 7 5 3 8
```
