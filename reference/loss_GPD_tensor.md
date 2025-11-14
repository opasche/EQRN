# GPD tensor loss function for training a EQRN network

GPD tensor loss function for training a EQRN network

## Usage

``` r
loss_GPD_tensor(
  out,
  y,
  orthogonal_gpd = TRUE,
  shape_penalty = 0,
  prior_shape = NULL,
  return_agg = c("mean", "sum", "vector", "nanmean", "nansum")
)
```

## Arguments

- out:

  Batch tensor of GPD parameters output by the network.

- y:

  Batch tensor of corresponding response variable.

- orthogonal_gpd:

  Whether the network is supposed to regress in the orthogonal
  reparametrization of the GPD parameters (recommended).

- shape_penalty:

  Penalty parameter for the shape estimate, to potentially regularize
  its variation from the fixed prior estimate.

- prior_shape:

  Prior estimate for the shape, used only if `shape_penalty>0`.

- return_agg:

  The return aggregation of the computed loss over the batch. Must be
  one of `"mean", "sum", "vector", "nanmean", "nansum"`.

## Value

The GPD loss over the batch between the network output and the observed
responses as a `torch::Tensor`, whose dimensions depend on `return_agg`.
