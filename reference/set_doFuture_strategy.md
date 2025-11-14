# Set a doFuture execution strategy

Set a doFuture execution strategy

## Usage

``` r
set_doFuture_strategy(
  strategy = c("sequential", "multisession", "multicore", "mixed"),
  n_workers = NULL
)
```

## Arguments

- strategy:

  One of `"sequential"` (default), `"multisession"`, `"multicore"`, or
  `"mixed"`.

- n_workers:

  A positive numeric scalar or a function specifying the maximum number
  of parallel futures that can be active at the same time before
  blocking. If a function, it is called without arguments when the
  future is created and its value is used to configure the workers. The
  function should return a numeric scalar. Defaults to
  [`future::availableCores()`](https://future.futureverse.org/reference/re-exports.html)`-1`
  if `NULL` (default), with `"multicore"` constraint in the relevant
  case. Ignored if `strategy=="sequential"`.

## Value

The appropriate
[`get_doFuture_operator()`](https://opasche.github.io/EQRN/reference/get_doFuture_operator.md)
operator to use in a
[`foreach::foreach()`](https://rdrr.io/pkg/foreach/man/foreach.html)
loop. The [`%do%`](https://rdrr.io/pkg/foreach/man/foreach.html)
operator is returned if `strategy=="sequential"`. Otherwise, the
[`%dopar%`](https://rdrr.io/pkg/foreach/man/foreach.html) operator is
returned.

## Examples

``` r
# \donttest{
`%fun%` <- set_doFuture_strategy("multisession", n_workers=3)
# perform foreach::foreach loop using the %fun% operator
end_doFuture_strategy()
# }
```
