# Dataset creator for sequential data

A
[`torch::dataset`](https://torch.mlverse.org/docs/reference/dataset.html)
object that can be initialized with sequential data, used to feed a
recurrent network during training or prediction. It is used in
[`EQRN_fit_seq()`](https://opasche.github.io/EQRN/reference/EQRN_fit_seq.md)
and corresponding predict functions, as well as in other recurrent
methods such as
[`QRN_seq_fit()`](https://opasche.github.io/EQRN/reference/QRN_seq_fit.md)
and its predict functions. It can perform scaling of the response's past
as a covariate, and compute excesses as a response when used in
[`EQRN_fit_seq()`](https://opasche.github.io/EQRN/reference/EQRN_fit_seq.md).
It also allows for fold separation or sequential discontinuity in the
data.

## Usage

``` r
mts_dataset(
  Y,
  X,
  seq_len,
  intermediate_quantiles = NULL,
  scale_Y = TRUE,
  fold_separation = NULL,
  sample_frac = 1,
  device = EQRN::default_device()
)
```

## Arguments

- Y:

  Response variable vector to model the extreme conditional quantile of,
  for training. Entries must be in sequential order.

- X:

  Matrix of covariates, for training. Entries must be in sequential
  order.

- seq_len:

  Data sequence length (i.e. number of past observations) used during
  training to predict each response quantile.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level `interm_lvl`.

- scale_Y:

  Whether to rescale the response past, when considered as an input
  covariate, to zero mean and unit covariance before applying the
  network (recommended).

- fold_separation:

  Fold separation index, when using concatenated folds as data.

- sample_frac:

  Value between `0` and `1`. If `sample_frac < 1`, a subsample of the
  data is used. Defaults to `1`.

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

The
[`torch::dataset`](https://torch.mlverse.org/docs/reference/dataset.html)
containing the given data, to be used with a recurrent neural network.
