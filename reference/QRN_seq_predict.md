# Predict function for a QRN_seq fitted object

Predict function for a QRN_seq fitted object

## Usage

``` r
QRN_seq_predict(
  fit_qrn_ts,
  X,
  Y,
  q_level = fit_qrn_ts$interm_lvl,
  crop_predictions = FALSE,
  device = default_device()
)
```

## Arguments

- fit_qrn_ts:

  Fitted `"QRN_seq"` object.

- X:

  Matrix of covariates to predict the corresponding response's
  conditional quantiles.

- Y:

  Response variable vector corresponding to the rows of `X`.

- q_level:

  Optional, checks that `q_level == fit_qrn_ts$interm_lvl`.

- crop_predictions:

  Whether to crop out the fist `seq_len` observations (which are `NA`)
  from the returned matrix.

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

Matrix of size `nrow(X)` times `1` (or `nrow(X)-seq_len` times `1` if
`crop_predictions`) containing the conditional quantile estimates of the
corresponding response observations.
