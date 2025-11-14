# Feature processor for EQRN

Feature processor for EQRN

## Usage

``` r
process_features(
  X,
  intermediate_q_feature,
  intermediate_quantiles = NULL,
  X_scaling = NULL,
  scale_features = TRUE
)
```

## Arguments

- X:

  A covariate matrix.

- intermediate_q_feature:

  Whether to use the intermediate `quantiles` as an additional
  covariate, by appending it to the `X` matrix (bool).

- intermediate_quantiles:

  The intermediate conditional quantiles.

- X_scaling:

  Existing `"X_scaling"` object containing the precomputed mean and
  variance for each covariate. This enables reusing the scaling choice
  and parameters from the train set, if computing the excesses on a
  validation or test set, in order to avoid overfitting. This is
  performed automatically in the `"EQRN"` objects.

- scale_features:

  Whether to rescale each input covariates to zero mean and unit
  variance before applying the network (recommended). If `X_scaling` is
  given, `X_scaling$scaling` overrides `scale_features`.

## Value

Named list containing:

- X_excesses:

  the (possibly rescaled and q_feat transformed) covariate matrix,

- X_scaling:

  object of class `"X_scaling"` to use for consistent scaling on future
  datasets.
