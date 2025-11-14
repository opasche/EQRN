# Check directory existence

Checks if the desired directory exists. If not, the desired directory is
created.

## Usage

``` r
check_directory(dir_name, recursive = TRUE, no_warning = FALSE)
```

## Arguments

- dir_name:

  Path to the desired directory, as a string.

- recursive:

  Should elements of the path other than the last be created? If `TRUE`,
  behaves like the Unix command `mkdir -p`.

- no_warning:

  Whether to cancel the warning issued if a directory is created (bool).

## Value

No return value.

## Examples

``` r
check_directory("./some_folder/my_new_folder")
#> Warning: The following given directory did not exist and was created by 'check_directory': ./some_folder/my_new_folder
```
