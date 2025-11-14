# Predict method for an EQRN_seq fitted object

Predict method for an EQRN_seq fitted object

## Usage

``` r
# S3 method for class 'EQRN_seq'
predict(object, ...)
```

## Arguments

- object:

  Fitted `"EQRN_seq"` object.

- ...:

  Arguments passed on to
  [`EQRN_predict_seq`](https://opasche.github.io/EQRN/reference/EQRN_predict_seq.md)

  `X`

  :   Matrix of covariates to predict the corresponding response's
      conditional quantiles.

  `Y`

  :   Response variable vector corresponding to the rows of `X`.

  `prob_lvls_predict`

  :   Vector of probability levels at which to predict the conditional
      quantiles.

  `intermediate_quantiles`

  :   Vector of intermediate conditional quantiles at level
      `fit_eqrn$interm_lvl`.

  `interm_lvl`

  :   Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.

  `crop_predictions`

  :   Whether to crop out the fist `seq_len` observations (which are
      `NA`) from the returned matrix.

  `seq_len`

  :   Data sequence length (i.e. number of past observations) used to
      predict each response quantile. By default, the training
      `fit_eqrn$seq_len` is used.

  `device`

  :   (optional) A
      [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
      Defaults to
      [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

Matrix of size `nrow(X)` times `prob_lvls_predict` (or `nrow(X)-seq_len`
times `prob_lvls_predict` if `crop_predictions`) containing the
conditional quantile estimates of the corresponding response
observations at each probability level. Simplifies to a vector if
`length(prob_lvls_predict)==1`.

## Details

See
[`EQRN_predict_seq()`](https://opasche.github.io/EQRN/reference/EQRN_predict_seq.md)
for more details.
