# Recurrent QRN fitting function

Used to fit a recurrent quantile regression neural network on a data
sample. Use the
[`QRN_fit_multiple()`](https://opasche.github.io/EQRN/reference/QRN_fit_multiple.md)
wrapper instead, with `data_type="seq"`, for better stability using
fitting restart.

## Usage

``` r
QRN_seq_fit(
  X,
  Y,
  q_level,
  hidden_size = 10,
  num_layers = 1,
  rnn_type = c("lstm", "gru"),
  p_drop = 0,
  learning_rate = 1e-04,
  L2_pen = 0,
  seq_len = 10,
  scale_features = TRUE,
  n_epochs = 10000,
  batch_size = 256,
  X_valid = NULL,
  Y_valid = NULL,
  lr_decay = 1,
  patience_decay = n_epochs,
  min_lr = 0,
  patience_stop = n_epochs,
  tol = 1e-04,
  fold_separation = NULL,
  warm_start_path = NULL,
  patience_lag = 5,
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

- Y:

  Response variable vector to model the conditional quantile of, for
  training. Entries must be in sequential order.

- q_level:

  Probability level of the desired conditional quantiles to predict.

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

- learning_rate:

  Initial learning rate for the optimizer during training of the neural
  network.

- L2_pen:

  L2 weight penalty parameter for regularization during training.

- seq_len:

  Data sequence length (i.e. number of past observations) used during
  training to predict each response quantile.

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

- Y_valid:

  Response variable in a validation set, or `NULL`. Entries must be in
  sequential order. Used for monitoring validation loss during training,
  enabling learning-rate decay and early stopping.

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

- fold_separation:

  Index of fold separation or sequential discontinuity in the data.

- warm_start_path:

  Path of a saved network using
  [`torch::torch_save()`](https://torch.mlverse.org/docs/reference/torch_save.html),
  to load back for a warm start.

- patience_lag:

  The validation loss is considered to be non-improving if it is larger
  than on any of the previous `patience_lag` epochs.

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

An QRN object of classes `c("QRN_seq", "QRN")`, containing the fitted
network, as well as all the relevant information for its usage in other
functions.
