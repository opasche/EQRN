# (INTERNAL) Corrects a dimension simplification bug from the torch package

(INTERNAL) Issue was raised to the `torch` maintainers and should be
fixed, deprecating this function.

## Usage

``` r
fix_dimsimplif(dl_i, ..., responses = TRUE)
```

## Arguments

- dl_i:

  batch object from an itteration over a
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html).

- ...:

  dimension(s) of the covariate object (excluding the first "batch"
  dimension)

- responses:

  Bolean indicating whether the batch object `dl_i` is a
  covariates-response pair.

## Value

The fixed dl_i object
