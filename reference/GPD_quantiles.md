# Compute extreme quantile from GPD parameters

Compute extreme quantile from GPD parameters

## Usage

``` r
GPD_quantiles(p, p0, t_x0, sigma, xi)
```

## Arguments

- p:

  Probability level of the desired extreme quantile.

- p0:

  Probability level of the (possibly varying) intermediate
  threshold/quantile.

- t_x0:

  Value(s) of the (possibly varying) intermediate threshold/quantile.

- sigma:

  Value(s) for the GPD scale parameter.

- xi:

  Value(s) for the GPD shape parameter.

## Value

The quantile value at probability level `p`.
