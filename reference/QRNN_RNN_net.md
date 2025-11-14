# Recurrent quantile regression neural network module

A recurrent neural network as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html),
designed for quantile regression.

## Usage

``` r
QRNN_RNN_net(
  type = c("lstm", "gru"),
  nb_input_features,
  hidden_size,
  num_layers = 1,
  dropout = 0
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
  regularization during training.

## Value

The specified recurrent QRN as a
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
  regularization during training.
