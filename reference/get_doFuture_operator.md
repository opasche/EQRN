# Get doFuture operator

Get doFuture operator

## Usage

``` r
get_doFuture_operator(
  strategy = c("sequential", "multisession", "multicore", "mixed")
)
```

## Arguments

- strategy:

  One of `"sequential"` (default), `"multisession"`, `"multicore"`, or
  `"mixed"`.

## Value

Returns the appropriate operator to use in a
[`foreach::foreach()`](https://rdrr.io/pkg/foreach/man/foreach.html)
loop. The [`%do%`](https://rdrr.io/pkg/foreach/man/foreach.html)
operator is returned if `strategy=="sequential"`. Otherwise, the
[`%dopar%`](https://rdrr.io/pkg/foreach/man/foreach.html) operator is
returned.

## Examples

``` r
`%fun%` <- get_doFuture_operator("sequential")
```
