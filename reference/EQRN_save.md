# Save an EQRN object on disc

Creates a folder named `name` and located in `path`, containing binary
save files, so that the given `"EQRN"` object `fit_eqrn` can be loaded
back in memory from disc using
[`EQRN_load()`](https://opasche.github.io/EQRN/reference/EQRN_load.md).

## Usage

``` r
EQRN_save(fit_eqrn, path, name = NULL, no_warning = TRUE)
```

## Arguments

- fit_eqrn:

  An `"EQRN"` object

- path:

  Path to save folder as a string.

- name:

  String name of the save.

- no_warning:

  Whether to silence the warning raised if a save folder needed beeing
  created (bool).

## Value

No return value.
