# Default torch device

Default torch device

## Usage

``` r
default_device()
```

## Value

Returns `torch::torch_device("cuda")` if
[`torch::cuda_is_available()`](https://torch.mlverse.org/docs/reference/cuda_is_available.html),
or `torch::torch_device("cpu")` otherwise.

## Examples

``` r
if(backend_is_installed()){
  device <- default_device()
  print(device)
}
#> torch_device(type='cpu') 
```
