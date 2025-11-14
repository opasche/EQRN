# Safe RDS save

Safe version of [`saveRDS()`](https://rdrr.io/r/base/readRDS.html). If
the given save path (i.e. `dirname(file_path)`) does not exist, it is
created instead of raising an error.

## Usage

``` r
safe_save_rds(object, file_path, recursive = TRUE, no_warning = FALSE)
```

## Arguments

- object:

  R variable or object to save on disk.

- file_path:

  Path and name of the save file, as a string.

- recursive:

  Should elements of the path other than the last be created? If `TRUE`,
  behaves like the Unix command `mkdir -p`.

- no_warning:

  Whether to cancel the warning issued if a directory is created (bool).

## Value

No return value.

## Examples

``` r
safe_save_rds(c(1, 2, 8), "./some_folder/my_new_folder/my_vector.rds")
```
