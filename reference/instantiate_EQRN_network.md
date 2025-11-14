# Instantiates the default networks for training a EQRN_iid model

Instantiates the default networks for training a EQRN_iid model

## Usage

``` r
instantiate_EQRN_network(
  net_structure,
  shape_fixed,
  D_in,
  hidden_fct,
  p_drop = 0,
  orthogonal_gpd = TRUE,
  device = default_device()
)
```

## Arguments

- net_structure:

  Vector of integers whose length determines the number of layers in the
  neural network and entries the number of neurons in each corresponding
  successive layer.

- shape_fixed:

  Whether the shape estimate depends on the covariates or not (bool).

- D_in:

  Number of covariates (including the intermediate quantile feature if
  used).

- hidden_fct:

  Activation function for the hidden layers. Can be either a callable
  function (preferably from the `torch` library), or one of the the
  strings `"SNN"`, `"SSNN"` for self normalizing networks (with common
  or separated networks for the scale and shape estimates,
  respectively). In the latter cases, `shape_fixed` has no effect.

- p_drop:

  Probability parameter for dropout before each hidden layer for
  regularization during training. `alpha-dropout` is used with SNNs.

- orthogonal_gpd:

  Whether to use the orthogonal reparametrization of the estimated GPD
  parameters (recommended).

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

## Value

A
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
network used to regress the GPD parameters in
[`EQRN_fit()`](https://opasche.github.io/EQRN/reference/EQRN_fit.md).
