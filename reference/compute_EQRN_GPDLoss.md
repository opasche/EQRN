# Generalized Pareto likelihood loss of a EQRN_iid predictor

Generalized Pareto likelihood loss of a EQRN_iid predictor

## Usage

``` r
compute_EQRN_GPDLoss(
  fit_eqrn,
  X,
  y,
  intermediate_quantiles = NULL,
  interm_lvl = fit_eqrn$interm_lvl,
  device = default_device()
)
```

## Arguments

- fit_eqrn:

  Fitted `"EQRN_iid"` object.

- X:

  Matrix of covariates.

- y:

  Response variable vector.

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

Negative GPD log likelihood of the conditional EQRN predicted parameters
over the response exceedances over the intermediate quantiles.
