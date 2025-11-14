# Generalized Pareto likelihood loss of a EQRN_seq predictor

Generalized Pareto likelihood loss of a EQRN_seq predictor

## Usage

``` r
compute_EQRN_seq_GPDLoss(
  fit_eqrn,
  X,
  Y,
  intermediate_quantiles = NULL,
  interm_lvl = fit_eqrn$interm_lvl,
  seq_len = fit_eqrn$seq_len,
  device = default_device()
)
```

## Arguments

- fit_eqrn:

  Fitted `"EQRN_seq"` object.

- X:

  Matrix of covariates.

- Y:

  Response variable vector corresponding to the rows of `X`.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level
  `fit_eqrn$interm_lvl`.

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

Negative GPD log likelihood of the conditional EQRN predicted parameters
over the response exceedances over the intermediate quantiles.
