# Tail excess probability prediction based on conditional GPD parameters

Tail excess probability prediction based on conditional GPD parameters

## Usage

``` r
GPD_excess_probability(
  val,
  sigma,
  xi,
  interm_threshold,
  threshold_p,
  body_proba = "default",
  proba_type = c("excess", "cdf")
)
```

## Arguments

- val:

  Quantile value(s) used to estimate the conditional excess probability
  or cdf.

- sigma:

  Value(s) for the GPD scale parameter.

- xi:

  Value(s) for the GPD shape parameter.

- interm_threshold:

  Intermediate (conditional) quantile(s) at level `threshold_p` used as
  a (varying) threshold.

- threshold_p:

  Probability level of the intermediate conditional quantiles
  `interm_threshold`.

- body_proba:

  Value to use when the predicted conditional probability is below
  `threshold_p` (in which case it cannot be precisely assessed by the
  model). If `"default"` is given (the default),
  `paste0(">",1-threshold_p)` is used if `proba_type=="excess"`, and
  `paste0("<",threshold_p)` is used if `proba_type=="cdf"`.

- proba_type:

  Whether to return the `"excess"` probability over `val` (default) or
  the `"cdf"` at `val`.

## Value

Vector of probabilities (and possibly a few `body_proba` values if `val`
is not large enough) of the same length as the longest vector between
`val`, `sigma`, `xi` and `interm_threshold`.
