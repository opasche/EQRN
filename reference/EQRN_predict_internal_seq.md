# Internal predict function for an EQRN_seq fitted object

Internal predict function for an EQRN_seq fitted object

## Usage

``` r
EQRN_predict_internal_seq(
  fit_eqrn,
  X,
  Y,
  prob_lvl_predict,
  intermediate_quantiles,
  interm_lvl,
  crop_predictions = FALSE,
  seq_len = fit_eqrn$seq_len,
  device = default_device()
)
```

## Arguments

- fit_eqrn:

  Fitted `"EQRN_seq"` object.

- X:

  Matrix of covariates to predict the corresponding response's
  conditional quantiles.

- Y:

  Response variable vector corresponding to the rows of `X`.

- prob_lvl_predict:

  Probability level at which to predict the conditional quantile.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level
  `fit_eqrn$interm_lvl`.

- interm_lvl:

  Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.

- crop_predictions:

  Whether to crop out the fist `seq_len` observations (which are `NA`)
  from the returned vector

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

Vector of length `nrow(X)` (or `nrow(X)-seq_len` if `crop_predictions`)
containing the conditional quantile estimates of the response associated
to each covariate observation at each probability level.
