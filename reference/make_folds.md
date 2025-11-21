# Create cross-validation folds

Utility function to create folds of data, used in cross-validation
proceidures. The implementation is originally from the `gbex` `R`
package

## Usage

``` r
make_folds(y, num_folds, stratified = FALSE)
```

## Arguments

- y:

  Numerical vector of observations

- num_folds:

  Number of folds to create.

- stratified:

  Logical value. If `TRUE`, the folds are stratified along `rank(y)`.

## Value

Vector of indices of the assigned folds for each observation.

## Examples

``` r
make_folds(rnorm(30), 5)
#>  [1] 1 2 1 1 5 1 2 2 3 1 4 3 5 4 5 4 5 3 2 4 5 2 3 3 3 4 4 2 5 1
```
