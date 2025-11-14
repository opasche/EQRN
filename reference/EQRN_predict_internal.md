# Internal predict function for an EQRN_iid

Internal predict function for an EQRN_iid

## Usage

``` r
EQRN_predict_internal(
  fit_eqrn,
  X,
  prob_lvl_predict,
  intermediate_quantiles,
  interm_lvl,
  device = default_device()
)
```

## Arguments

- fit_eqrn:

  Fitted `"EQRN_iid"` object.

- X:

  Matrix of covariates to predict the corresponding response's
  conditional quantiles.

- prob_lvl_predict:

  Probability level at which to predict the conditional quantiles.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level
  `fit_eqrn$interm_lvl`.

- interm_lvl:

  Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

Vector of length `nrow(X)` containing the conditional quantile estimates
of the response associated to each covariate observation at each
probability level `prob_lvl_predict`.
