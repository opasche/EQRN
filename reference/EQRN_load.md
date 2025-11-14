# Load an EQRN object from disc

Loads in memory an `"EQRN"` object that has previously been saved on
disc using
[`EQRN_save()`](https://opasche.github.io/EQRN/reference/EQRN_save.md).

## Usage

``` r
EQRN_load(path, name = NULL, device = default_device(), ...)
```

## Arguments

- path:

  Path to the save location as a string.

- name:

  String name of the save. If `NULL` (default), assumes the save name
  has been given implicitly in the `path`.

- device:

  (optional) A
  [`torch::torch_device()`](https://torch.mlverse.org/docs/reference/torch_device.html).
  Defaults to
  [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md).

- ...:

  DEPRECATED. Used for back-compatibility.

## Value

The loaded `"EQRN"` model.
