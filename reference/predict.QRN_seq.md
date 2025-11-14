# Predict method for a QRN_seq fitted object

Predict method for a QRN_seq fitted object

## Usage

``` r
# S3 method for class 'QRN_seq'
predict(object, ...)
```

## Arguments

- object:

  Fitted `"QRN_seq"` object.

- ...:

  Arguments passed on to
  [`QRN_seq_predict`](https://opasche.github.io/EQRN/reference/QRN_seq_predict.md)

  `X`

  :   Matrix of covariates to predict the corresponding response's
      conditional quantiles.

  `Y`

  :   Response variable vector corresponding to the rows of `X`.

  `q_level`

  :   Optional, checks that `q_level == fit_qrn_ts$interm_lvl`.

  `crop_predictions`

  :   Whether to crop out the fist `seq_len` observations (which are
      `NA`) from the returned matrix.

  `device`

  :   (optional) A
      [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
      Defaults to
      [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

Matrix of size `nrow(X)` times `1` (or `nrow(X)-seq_len` times `1` if
`crop_predictions`) containing the conditional quantile estimates of the
corresponding response observations.

## Details

See
[`QRN_seq_predict()`](https://opasche.github.io/EQRN/reference/QRN_seq_predict.md)
for more details.
