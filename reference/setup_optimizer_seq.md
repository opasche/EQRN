# Instantiate an optimizer for training an EQRN_seq network

Instantiate an optimizer for training an EQRN_seq network

## Usage

``` r
setup_optimizer_seq(network, learning_rate, L2_pen, optim_met = "adam")
```

## Arguments

- network:

  A
  [`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
  network to be trained in
  [`EQRN_fit_seq()`](https://opasche.github.io/EQRN/reference/EQRN_fit_seq.md).

- learning_rate:

  Initial learning rate for the optimizer during training of the neural
  network.

- L2_pen:

  L2 weight penalty parameter for regularization during training.

- optim_met:

  DEPRECATED. Optimization algorithm to use during training. `"adam"` is
  the default.

## Value

A
[`torch::optimizer`](https://torch.mlverse.org/docs/reference/optimizer.html)
object used in
[`EQRN_fit_seq()`](https://opasche.github.io/EQRN/reference/EQRN_fit_seq.md)
for training.
