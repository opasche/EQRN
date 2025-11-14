# GPD parameters prediction function for an EQRN_iid fitted object

GPD parameters prediction function for an EQRN_iid fitted object

## Usage

``` r
EQRN_predict_params(
  fit_eqrn,
  X,
  intermediate_quantiles = NULL,
  return_parametrization = c("classical", "orthogonal"),
  interm_lvl = fit_eqrn$interm_lvl,
  device = default_device()
)
```

## Arguments

- fit_eqrn:

  Fitted `"EQRN_iid"` object.

- X:

  Matrix of covariates to predict conditional GPD parameters.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level
  `fit_eqrn$interm_lvl`.

- return_parametrization:

  Which parametrization to return the parameters in, either
  `"classical"` or `"orthogonal"`.

- interm_lvl:

  Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

Named list containing: `"scales"` and `"shapes"` as numerical vectors of
length `nrow(X)`.
