# Check if Torch Backend Libraries are Installed

Check if Torch Backend Libraries are Installed

## Usage

``` r
backend_is_installed(...)
```

## Arguments

- ...:

  Optional parameters passed to
  [`torch::torch_is_installed()`](https://torch.mlverse.org/docs/reference/torch_is_installed.html).

## Value

Boolean indicating whether the LibTorch and LibLantern backend libraries
are installed.

## Examples

``` r
if(backend_is_installed()){
  cat("torch's LibTorch and LibLantern backend libraries are installed!\n")
}
#> torch's LibTorch and LibLantern backend libraries are installed!
```
