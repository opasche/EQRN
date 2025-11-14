# Alpha-dropout module

An alpha-dropout layer as a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html),
used in self-normalizing networks.

## Usage

``` r
nn_alpha_dropout(p = 0.5, inplace = FALSE)
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

## References

Gunter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter.
Self-Normalizing Neural Networks. Advances in Neural Information
Processing Systems 30 (NIPS 2017), 2017.
