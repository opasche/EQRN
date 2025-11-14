# Wrapper for fitting a recurrent QRN with restart for stability

Wrapper for fitting a recurrent QRN with restart for stability

## Usage

``` r
QRN_fit_multiple(
  X,
  y,
  q_level,
  number_fits = 3,
  ...,
  seed = NULL,
  data_type = c("seq", "iid")
)
```

## Arguments

- X:

  Matrix of covariates, for training.

- y:

  Response variable vector to model the conditional quantile of, for
  training.

- q_level:

  Probability level of the desired conditional quantiles to predict.

- number_fits:

  Number of restarts.

- ...:

  Other parameters given to
  [`QRN_seq_fit()`](https://opasche.github.io/EQRN/reference/QRN_seq_fit.md).

- seed:

  Integer random seed for reproducibility in network weight
  initialization.

- data_type:

  Type of data dependence, must be one of `"iid"` (for iid observations)
  or `"seq"` (for sequentially dependent observations). For the moment,
  only `"seq"` is accepted.

## Value

An QRN object of classes `c("QRN_seq", "QRN")`, containing the fitted
network, as well as all the relevant information for its usage in other
functions.
