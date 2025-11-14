# Maximum likelihood estimates for the GPD distribution using peaks over threshold

Maximum likelihood estimates for the GPD distribution using peaks over
threshold

## Usage

``` r
fit_GPD_unconditional(Y, interm_lvl = NULL, thresh_quantiles = NULL)
```

## Arguments

- Y:

  Vector of observations

- interm_lvl:

  Probability level at which the empirical quantile should be used as
  the threshold, if `thresh_quantiles` is not given.

- thresh_quantiles:

  Numerical value or numerical vector of the same length as `Y`
  representing either a fixed or a varying threshold, respectively.

## Value

Named list containing:

- scale:

  the GPD scale MLE,

- shape:

  the GPD shape MLE,

- fit:

  the fitted
  [`ismev::gpd.fit()`](https://rdrr.io/pkg/ismev/man/gpd.fit.html)
  object.
