# Ensure Torch Backend Libraries are Installed

Ensure Torch Backend Libraries are Installed

## Usage

``` r
ensure_backend_installed(
  behaviour = c("error", "warn", "message", "bool"),
  ...
)
```

## Arguments

- behaviour:

  one of "error" or "bool".

- ...:

  Optional parameters passed to
  [`torch::torch_is_installed()`](https://torch.mlverse.org/docs/reference/torch_is_installed.html).

## Value

Boolean indicating whether the LibTorch and LibLantern backend libraries
are installed. If `behaviour = "error"`, an error is raised if the
backend libraries are not installed.
