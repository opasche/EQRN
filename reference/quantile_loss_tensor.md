# Tensor quantile loss function for training a QRN network

Tensor quantile loss function for training a QRN network

## Usage

``` r
quantile_loss_tensor(
  out,
  y,
  q = 0.5,
  return_agg = c("mean", "sum", "vector", "nanmean", "nansum")
)
```

## Arguments

- out:

  Batch tensor of the quantile output by the network.

- y:

  Batch tensor of corresponding response variable.

- q:

  Probability level of the predicted quantile

- return_agg:

  The return aggregation of the computed loss over the batch. Must be
  one of `"mean", "sum", "vector", "nanmean", "nansum"`.

## Value

The quantile loss over the batch between the network output ans the
observed responses as a `torch::Tensor`, whose dimensions depend on
`return_agg`.
