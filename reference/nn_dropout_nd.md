# Dropout module

A dropout layer as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html).

## Usage

``` r
nn_dropout_nd(p = 0.5, inplace = FALSE)
```

## Arguments

- p:

  probability for dropout.

- inplace:

  whether the dropout in performed inplace.

## Details

The constructor allows specifying:

- p:

  probability of an element to be zeroed (default is 0.5),

- inplace:

  if set to TRUE, will do the operation in-place (default is FALSE).
