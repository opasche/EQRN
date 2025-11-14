# Instantiate an optimizer for training an EQRN_iid network

Instantiate an optimizer for training an EQRN_iid network

## Usage

``` r
setup_optimizer(network, learning_rate, L2_pen, hidden_fct, optim_met = "adam")
```

## Arguments

- network:

  A
  [`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
  network to be trained in
  [`EQRN_fit()`](https://opasche.github.io/EQRN/reference/EQRN_fit.md).

- learning_rate:

  Initial learning rate for the optimizer during training of the neural
  network.

- L2_pen:

  L2 weight penalty parameter for regularization during training.

- hidden_fct:

  Activation function for the hidden layers. Can be either a callable
  function (preferably from the `torch` library), or one of the the
  strings `"SNN"`, `"SSNN"` for self normalizing networks (with common
  or separated networks for the scale and shape estimates,
  respectively). This will affect the default choice of optimizer.

- optim_met:

  DEPRECATED. Optimization algorithm to use during training. `"adam"` is
  the default.

## Value

A
[`torch::optimizer`](https://torch.mlverse.org/docs/reference/optimizer.html)
object used in
[`EQRN_fit()`](https://opasche.github.io/EQRN/reference/EQRN_fit.md) for
training.
