# Performs a learning rate decay step on an optimizer

Performs a learning rate decay step on an optimizer

## Usage

``` r
decay_learning_rate(optimizer, decay_rate)
```

## Arguments

- optimizer:

  A
  [`torch::optimizer`](https://torch.mlverse.org/docs/reference/optimizer.html)
  object.

- decay_rate:

  Learning rate decay factor.

## Value

The `optimizer` with a decayed learning rate.
