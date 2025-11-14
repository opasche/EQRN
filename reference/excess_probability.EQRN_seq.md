# Tail excess probability prediction method using an EQRN_iid object

Tail excess probability prediction method using an EQRN_iid object

## Usage

``` r
# S3 method for class 'EQRN_seq'
excess_probability(object, ...)
```

## Arguments

- object:

  Fitted `"EQRN_seq"` object.

- ...:

  Arguments passed on to
  [`EQRN_excess_probability_seq`](https://opasche.github.io/EQRN/reference/EQRN_excess_probability_seq.md)

  `val`

  :   Quantile value(s) used to estimate the conditional excess
      probability or cdf.

  `X`

  :   Matrix of covariates to predict the response's conditional excess
      probabilities.

  `Y`

  :   Response variable vector corresponding to the rows of `X`.

  `intermediate_quantiles`

  :   Vector of intermediate conditional quantiles at level
      `fit_eqrn$interm_lvl`.

  `interm_lvl`

  :   Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.

  `crop_predictions`

  :   Whether to crop out the fist `seq_len` observations (which are
      `NA`) from the returned vector

  `body_proba`

  :   Value to use when the predicted conditional probability is below
      `interm_lvl` (in which case it cannot be precisely assessed by the
      model). If `"default"` is given (the default),
      `paste0(">",1-interm_lvl)` is used if `proba_type=="excess"`, and
      `paste0("<",interm_lvl)` is used if `proba_type=="cdf"`.

  `proba_type`

  :   Whether to return the `"excess"` probability over `val` (default)
      or the `"cdf"` at `val`.

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

Vector of probabilities (and possibly a few `body_proba` values if `val`
is not large enough) of length `nrow(X)` (or `nrow(X)-seq_len` if
`crop_predictions`).

## Details

See
[`EQRN_excess_probability_seq()`](https://opasche.github.io/EQRN/reference/EQRN_excess_probability_seq.md)
for more details.
