# Recurrent network module for GPD parameter prediction

A recurrent neural network as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html),
designed for generalized Pareto distribution parameter prediction, with
sequential dependence.

## Usage

``` r
Recurrent_GPD_net(
  type = c("lstm", "gru"),
  nb_input_features,
  hidden_size,
  num_layers = 1,
  dropout = 0,
  shape_fixed = FALSE,
  device = EQRN::default_device()
)
```

## Arguments

- type:

  the type of recurrent architecture, can be one of `"lstm"` (default)
  or `"gru"`,

- nb_input_features:

  the input size (i.e. the number of features),

- hidden_size:

  the dimension of the hidden latent state variables in the recurrent
  network,

- num_layers:

  the number of recurrent layers,

- dropout:

  probability parameter for dropout before each hidden layer for
  regularization during training,

- shape_fixed:

  whether the shape estimate depends on the covariates or not (bool),

- device:

  a
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html)
  for an internal constant vector. Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

The specified recurrent GPD network as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html).

## Details

The constructor allows specifying:

- type:

  the type of recurrent architecture, can be one of `"lstm"` (default)
  or `"gru"`,

- nb_input_features:

  the input size (i.e. the number of features),

- hidden_size:

  the dimension of the hidden latent state variables in the recurrent
  network,

- num_layers:

  the number of recurrent layers,

- dropout:

  probability parameter for dropout before each hidden layer for
  regularization during training,

- shape_fixed:

  whether the shape estimate depends on the covariates or not (bool),

- device:

  a
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html)
  for an internal constant vector. Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).
