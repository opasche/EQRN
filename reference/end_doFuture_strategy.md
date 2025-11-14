# End the currently set doFuture strategy

Resets the default strategy using `future::plan("default")`.

## Usage

``` r
end_doFuture_strategy()
```

## Value

No return value.

## Examples

``` r
# \donttest{
`%fun%` <- set_doFuture_strategy("multisession", n_workers=3)
# perform foreach::foreach loop using the %fun% operator
end_doFuture_strategy()
# }
```
