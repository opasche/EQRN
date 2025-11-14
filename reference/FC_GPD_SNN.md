# Self-normalized fully-connected network module for GPD parameter prediction

A fully-connected self-normalizing network as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html),
designed for generalized Pareto distribution parameter prediction.

## Usage

``` r
FC_GPD_SNN(D_in, Hidden_vect = c(64, 64, 64), p_drop = 0.01)
```

## Arguments

- D_in:

  the input size (i.e. the number of features),

- Hidden_vect:

  a vector of integers whose length determines the number of layers in
  the neural network and entries the number of neurons in each
  corresponding successive layer,

- p_drop:

  probability parameter for the `alpha-dropout` before each hidden layer
  for regularization during training.

## Value

The specified SNN MLP GPD network as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html).

## Details

The constructor allows specifying:

- D_in:

  the input size (i.e. the number of features),

- Hidden_vect:

  a vector of integers whose length determines the number of layers in
  the neural network and entries the number of neurons in each
  corresponding successive layer,

- p_drop:

  probability parameter for the `alpha-dropout` before each hidden layer
  for regularization during training.

## References

Gunter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter.
Self-Normalizing Neural Networks. Advances in Neural Information
Processing Systems 30 (NIPS 2017), 2017.
