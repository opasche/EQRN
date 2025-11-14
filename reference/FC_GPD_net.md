# MLP module for GPD parameter prediction

A fully-connected network (or multi-layer perception) as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html),
designed for generalized Pareto distribution parameter prediction.

## Usage

``` r
FC_GPD_net(
  D_in,
  Hidden_vect = c(5, 5, 5),
  activation = torch::nnf_sigmoid,
  p_drop = 0,
  shape_fixed = FALSE,
  device = EQRN::default_device()
)
```

## Arguments

- D_in:

  the input size (i.e. the number of features),

- Hidden_vect:

  a vector of integers whose length determines the number of layers in
  the neural network and entries the number of neurons in each
  corresponding successive layer,

- activation:

  the activation function for the hidden layers (should be either a
  callable function, preferably from the `torch` library),

- p_drop:

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

The specified MLP GPD network as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html).

## Details

The constructor allows specifying:

- D_in:

  the input size (i.e. the number of features),

- Hidden_vect:

  a vector of integers whose length determines the number of layers in
  the neural network and entries the number of neurons in each
  corresponding successive layer,

- activation:

  the activation function for the hidden layers (should be either a
  callable function, preferably from the `torch` library),

- p_drop:

  probability parameter for dropout before each hidden layer for
  regularization during training,

- shape_fixed:

  whether the shape estimate depends on the covariates or not (bool),

- device:

  a
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html)
  for an internal constant vector. Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).
