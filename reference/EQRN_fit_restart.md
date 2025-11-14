# Wrapper for fitting EQRN with restart for stability

Wrapper for fitting EQRN with restart for stability

## Usage

``` r
EQRN_fit_restart(
  X,
  y,
  intermediate_quantiles,
  interm_lvl,
  number_fits = 3,
  ...,
  seed = NULL,
  data_type = c("iid", "seq")
)
```

## Arguments

- X:

  Matrix of covariates, for training.

- y:

  Response variable vector to model the extreme conditional quantile of,
  for training.

- intermediate_quantiles:

  Vector of intermediate conditional quantiles at level `interm_lvl`.

- interm_lvl:

  Probability level for the intermediate quantiles
  `intermediate_quantiles`.

- number_fits:

  Number of restarts.

- ...:

  Other parameters given to either
  [`EQRN_fit()`](https://opasche.github.io/EQRN/reference/EQRN_fit.md)
  or
  [`EQRN_fit_seq()`](https://opasche.github.io/EQRN/reference/EQRN_fit_seq.md),
  depending on the `data_type`.

- seed:

  Integer random seed for reproducibility in network weight
  initialization.

- data_type:

  Type of data dependence, must be one of `"iid"` (for iid observations)
  or `"seq"` (for sequentially dependent observations).

## Value

An EQRN object of classes `c("EQRN_iid", "EQRN")`, if
`data_type=="iid",` or `c("EQRN_seq", "EQRN")`, if \`data_type=="seq",
containing the fitted network, as well as all the relevant information
for its usage in other functions.
