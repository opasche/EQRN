# Install Torch Backend Libraries

This function can be called just after installing the EQRN package.
Calling `EQRN::install_backend()` installs the necessary LibTorch and
LibLantern backend libraries of the
[`torch`](https://torch.mlverse.org/) dependency by calling
[`torch::install_torch()`](https://torch.mlverse.org/docs/reference/install_torch.html).
See <https://torch.mlverse.org/docs/articles/installation> for more
details and troubleshooting. Calling this function shouldn't be
necessary in interactive environments, as loading EQRN (e.g. with
[`library(EQRN)`](https://github.com/opasche/EQRN) or with any
`EQRN::fct()`) should prompt to do it automatically (via `.onLoad()`).
This behaviour is inherited from the
[`torch`](https://torch.mlverse.org/) package.

## Usage

``` r
install_backend(...)
```

## Arguments

- ...:

  Arguments passed to
  [`torch::install_torch()`](https://torch.mlverse.org/docs/reference/install_torch.html).

## Value

No return value.
