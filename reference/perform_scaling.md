# Performs feature scaling without overfitting

Performs feature scaling without overfitting

## Usage

``` r
perform_scaling(X, X_scaling = NULL, scale_features = TRUE, stat_attr = FALSE)
```

## Arguments

- X:

  A covariate matrix.

- X_scaling:

  Existing `"X_scaling"` object containing the precomputed mean and
  variance for each covariate. This enables reusing the scaling choice
  and parameters from the train set, if computing the excesses on a
  validation or test set, in order to avoid overfitting. This is
  performed automatically in the `"EQRN"` objects.

- scale_features:

  Whether to rescale each input covariates to zero mean and unit
  variance before applying the model (recommended). If `X_scaling` is
  given, `X_scaling$scaling` overrides `scale_features`.

- stat_attr:

  DEPRECATED. Whether to keep attributes in the returned covariate
  matrix itself.

## Value

Named list containing:

- X_excesses:

  the (possibly rescaled and q_feat transformed) covariate matrix,

- X_scaling:

  object of class `"X_scaling"` to use for consistent scaling on future
  datasets.
