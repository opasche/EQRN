# GPD parameters prediction function for an EQRN_seq fitted object

GPD parameters prediction function for an EQRN_seq fitted object

## Usage

``` r
EQRN_predict_params_seq(
  fit_eqrn,
  X,
  Y,
  intermediate_quantiles = NULL,
  return_parametrization = c("classical", "orthogonal"),
  interm_lvl = fit_eqrn$interm_lvl,
  seq_len = fit_eqrn$seq_len,
  device = default_device()
)
```

## Arguments

- fit_eqrn:

  Fitted `"EQRN_seq"` object.

- X:

  Matrix of covariates to predict conditional GPD parameters.

- Y:

  Response variable vector corresponding to the rows of `X`.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level
  `fit_eqrn$interm_lvl`.

- return_parametrization:

  Which parametrization to return the parameters in, either
  `"classical"` or `"orthogonal"`.

- interm_lvl:

  Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.

- seq_len:

  Data sequence length (i.e. number of past observations) used to
  predict each response quantile. By default, the training
  `fit_eqrn$seq_len` is used.

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

Named list containing: `"scales"` and `"shapes"` as numerical vectors of
length `nrow(X)`, and the `seq_len` used.
