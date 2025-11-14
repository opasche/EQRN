# Predict method for an EQRN_iid fitted object

Predict method for an EQRN_iid fitted object

## Usage

``` r
# S3 method for class 'EQRN_iid'
predict(object, ...)
```

## Arguments

- object:

  Fitted `"EQRN_iid"` object.

- ...:

  Arguments passed on to
  [`EQRN_predict`](https://opasche.github.io/EQRN/reference/EQRN_predict.md)

  `X`

  :   Matrix of covariates to predict the corresponding response's
      conditional quantiles.

  `prob_lvls_predict`

  :   Vector of probability levels at which to predict the conditional
      quantiles.

  `intermediate_quantiles`

  :   Vector of intermediate conditional quantiles at level
      `fit_eqrn$interm_lvl`.

  `interm_lvl`

  :   Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.

  `device`

  :   (optional) A
      [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
      Defaults to
      [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

Matrix of size `nrow(X)` times `prob_lvls_predict` containing the
conditional quantile estimates of the response associated to each
covariate observation at each probability level. Simplifies to a vector
if `length(prob_lvls_predict)==1`.

## Details

See
[`EQRN_predict()`](https://opasche.github.io/EQRN/reference/EQRN_predict.md)
for more details.
