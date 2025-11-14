# EQRN fit function for sequential and time series data

Use the
[`EQRN_fit_restart()`](https://opasche.github.io/EQRN/reference/EQRN_fit_restart.md)
wrapper instead, with `data_type="seq"`, for better stability using
fitting restart.

## Usage

``` r
EQRN_fit_seq(
  X,
  y,
  intermediate_quantiles,
  interm_lvl,
  shape_fixed = FALSE,
  hidden_size = 10,
  num_layers = 1,
  rnn_type = c("lstm", "gru"),
  p_drop = 0,
  intermediate_q_feature = TRUE,
  learning_rate = 1e-04,
  L2_pen = 0,
  seq_len = 10,
  shape_penalty = 0,
  scale_features = TRUE,
  n_epochs = 500,
  batch_size = 256,
  X_valid = NULL,
  y_valid = NULL,
  quant_valid = NULL,
  lr_decay = 1,
  patience_decay = n_epochs,
  min_lr = 0,
  patience_stop = n_epochs,
  tol = 1e-05,
  orthogonal_gpd = TRUE,
  patience_lag = 1,
  fold_separation = NULL,
  optim_met = "adam",
  seed = NULL,
  verbose = 2,
  device = default_device()
)
```

## Arguments

- X:

  Matrix of covariates, for training. Entries must be in sequential
  order.

- y:

  Response variable vector to model the extreme conditional quantile of,
  for training. Entries must be in sequential order.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level `interm_lvl`.

- interm_lvl:

  Probability level for the intermediate quantiles
  `intermediate_quantiles`.

- shape_fixed:

  Whether the shape estimate depends on the covariates or not (bool).

- hidden_size:

  Dimension of the hidden latent state variables in the recurrent
  network.

- num_layers:

  Number of recurrent layers.

- rnn_type:

  Type of recurrent architecture, can be one of `"lstm"` (default) or
  `"gru"`.

- p_drop:

  Probability parameter for dropout before each hidden layer for
  regularization during training.

- intermediate_q_feature:

  Whether to use the `intermediate_quantiles` as an additional
  covariate, by appending it to the `X` matrix (bool).

- learning_rate:

  Initial learning rate for the optimizer during training of the neural
  network.

- L2_pen:

  L2 weight penalty parameter for regularization during training.

- seq_len:

  Data sequence length (i.e. number of past observations) used during
  training to predict each response quantile.

- shape_penalty:

  Penalty parameter for the shape estimate, to potentially regularize
  its variation from the fixed prior estimate.

- scale_features:

  Whether to rescale each input covariates to zero mean and unit
  covariance before applying the network (recommended).

- n_epochs:

  Number of training epochs.

- batch_size:

  Batch size used during training.

- X_valid:

  Covariates in a validation set, or `NULL`. Entries must be in
  sequential order. Used for monitoring validation loss during training,
  enabling learning-rate decay and early stopping.

- y_valid:

  Response variable in a validation set, or `NULL`. Entries must be in
  sequential order. Used for monitoring validation loss during training,
  enabling learning-rate decay and early stopping.

- quant_valid:

  Intermediate conditional quantiles at level `interm_lvl` in a
  validation set, or `NULL`. Used for monitoring validation loss during
  training, enabling learning-rate decay and early stopping.

- lr_decay:

  Learning rate decay factor.

- patience_decay:

  Number of epochs of non-improving validation loss before a
  learning-rate decay is performed.

- min_lr:

  Minimum learning rate, under which no more decay is performed.

- patience_stop:

  Number of epochs of non-improving validation loss before early
  stopping is performed.

- tol:

  Tolerance for stopping training, in case of no significant training
  loss improvements.

- orthogonal_gpd:

  Whether to use the orthogonal reparametrization of the estimated GPD
  parameters (recommended).

- patience_lag:

  The validation loss is considered to be non-improving if it is larger
  than on any of the previous `patience_lag` epochs.

- fold_separation:

  Index of fold separation or sequential discontinuity in the data.

- optim_met:

  DEPRECATED. Optimization algorithm to use during training. `"adam"` is
  the default.

- seed:

  Integer random seed for reproducibility in network weight
  initialization.

- verbose:

  Amount of information printed during training (0:nothing, 1:most
  important, 2:everything).

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

An EQRN object of classes `c("EQRN_seq", "EQRN")`, containing the fitted
network, as well as all the relevant information for its usage in other
functions.
