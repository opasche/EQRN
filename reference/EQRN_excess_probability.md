# Tail excess probability prediction using an EQRN_iid object

Tail excess probability prediction using an EQRN_iid object

## Usage

``` r
EQRN_excess_probability(
  val,
  fit_eqrn,
  X,
  intermediate_quantiles,
  interm_lvl = fit_eqrn$interm_lvl,
  body_proba = "default",
  proba_type = c("excess", "cdf"),
  device = default_device()
)
```

## Arguments

- val:

  Quantile value(s) used to estimate the conditional excess probability
  or cdf.

- fit_eqrn:

  Fitted `"EQRN_iid"` object.

- X:

  Matrix of covariates to predict the corresponding response's
  conditional excess probabilities.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level
  `fit_eqrn$interm_lvl`.

- interm_lvl:

  Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.

- body_proba:

  Value to use when the predicted conditional probability is below
  `interm_lvl` (in which case it cannot be precisely assessed by the
  model). If `"default"` is given (the default),
  `paste0(">",1-interm_lvl)` is used if `proba_type=="excess"`, and
  `paste0("<",interm_lvl)` is used if `proba_type=="cdf"`.

- proba_type:

  Whether to return the `"excess"` probability over `val` (default) or
  the `"cdf"` at `val`.

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

Vector of probabilities (and possibly a few `body_proba` values if `val`
is not large enough) of length `nrow(X)`.
