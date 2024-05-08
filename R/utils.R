
#' Check directory existence
#'
#' @description Checks if the desired directory exists. If not, the desired directory is created.
#'
#' @param dir_name Path to the desired directory, as a string.
#' @param recursive Should elements of the path other than the last be created?
#' If `TRUE`, behaves like the Unix command `mkdir -p`.
#' @param no_warning Whether to cancel the warning issued if a directory is created (bool).
#'
#' @export
#'
#' @examples \dontrun{check_directory("./results/my_new_folder")}
check_directory <- function(dir_name, recursive=TRUE, no_warning=FALSE){
  if (!dir.exists(dir_name)){
    dir.create(dir_name, recursive=recursive)
    if(!no_warning){
      warning(paste0("The following given directory did not exist and was created by 'check_directory': ", dir_name))
    }
  }
}


#' Safe RDS save
#'
#' @description Safe version of [saveRDS()].
#' If the given save path (i.e. `dirname(file_path)`) does not exist, it is created instead of raising an error.
#'
#' @param object R variable or object to save on disk.
#' @param file_path Path and name of the save file, as a string.
#' @param recursive Should elements of the path other than the last be created?
#' If `TRUE`, behaves like the Unix command `mkdir -p`.
#' @param no_warning Whether to cancel the warning issued if a directory is created (bool).
#'
#' @export
#'
#' @examples \dontrun{safe_save_rds(c(1, 2, 8), "./results/my_new_folder/my_vector.rds")}
safe_save_rds <- function(object, file_path, recursive=TRUE, no_warning=FALSE){
  dir_name <- dirname(file_path)
  check_directory(dir_name, recursive=recursive, no_warning=no_warning)
  
  saveRDS(object, file = file_path)
  
}


#' Last element of a vector
#'
#' @param x Vector.
#'
#' @description Returns the last element of the given vector in the most efficient way.
#'
#' @return The last element in the vector `x`.
#'
#' @details The last element is obtained using `x[length(x)]`, which is done in `O(1)` and faster than, for example, any of
#' `Rcpp::mylast(x)`, `tail(x, n=1)`, `dplyr::last(x)`, `x[end(x)[1]]]`, and `rev(x)[1]`.
#' @export
#'
#' @examples last_elem(c(2, 6, 1, 4))
last_elem <- function(x){
  x[length(x)]
}


#' Mathematical number rounding
#'
#' @description This function rounds numbers in the mathematical sense,
#' as opposed to the base `R` function [round()] that rounds 'to the even digit'.
#'
#' @param x Vector of numerical values to round.
#' @param decimals Integer indicating the number of decimal places to be used.
#'
#' @return A vector containing the entries of `x`, rounded to `decimals` decimals.
#' @export
#'
#' @examples roundm(2.25, 1)
roundm = function(x, decimals=0){
  posneg <- sign(x)
  z <- abs(x)*10^decimals
  z <- z + 0.5 + sqrt(.Machine$double.eps)
  z <- trunc(z)
  z <- z/10^decimals
  z*posneg
}


#' Convert a vector to a matrix
#'
#' @param v Vector.
#' @param axis One of `"col"` (default) or `"row"`.
#'
#' @return The vector `v` as a matrix.
#' If `axis=="col"` (default) the column vector `v` is returned as a `length(v)` times `1` matrix.
#' If `axis=="row"`, the vector `v` is returned as a transposed `1` times `length(v)` matrix.
#' @export
#'
#' @examples vec2mat(c(2, 7, 3, 8), "col")
vec2mat <- function(v, axis=c("col","row")){
  axis <- match.arg(axis)
  if (is.null(dim(v))) {
    v <- if(axis=="col"){matrix(v, nrow=1)}else{matrix(v, ncol=1)}
  }
  return(v)
}


#' Tibble replicatior
#'
#' @param tbl A [tibble::tibble()].
#' @param m An integer.
#'
#' @return The tibble is replicated `m` times and colums names appended with `rep_id = 1:m`.
#' @importFrom magrittr %>%
#' @importFrom tibble rownames_to_column
#' @importFrom tidyr expand_grid
#' @importFrom dplyr left_join select
#'
#' @examples #rep_tibble(tibble::tibble(a=c(2,3), b=c(5,6)), 3)
#' @keywords internal
rep_tibble <- function(tbl, m){
  tbl <-  tbl %>% tibble::rownames_to_column()
  
  tidyr::expand_grid(rep_id = 1:m, rowname = tbl$rowname) %>%
    dplyr::left_join(tbl, by = "rowname") %>%
    dplyr::select(-rowname)
}


#' Replicated vector to matrix
#'
#' @param vec Vector.
#' @param nrep Number of repetitions.
#' @param dim One of `"row"` (default) or `"col"`.
#'
#' @return Matrix of replicated vector.
#'
#' @examples #rep_vector2matrix(c(2, 7, 3, 8), 3, "row")
#' @keywords internal
rep_vector2matrix <- function(vec, nrep, dim = c("row", "col")){
  ## stack nrep of vec in (row|col) of a matrix
  
  dim <- match.arg(dim)
  l <- length(vec)
  
  if (l == 0){
    stop("vec must contain at least one element.")
  }
  
  if (dim == "col"){
    matrix(vec, nrow = l, ncol = nrep)
  } else {
    matrix(vec, nrow = nrep, ncol = l, byrow = TRUE)
  }
}


#' Convert a list to a matrix
#'
#' @param lst A list.
#' @param dim One of `"row"` (default) or `"col"`.
#'
#' @return The list converted to a matrix, by stacking the elements of `lst` in the rows or columns of a matrix.
#'
#' @examples #list2matrix(list(2, 7, 3, 8), "row")
#' @keywords internal
list2matrix <- function(lst, dim = c("row", "col")){
  dim <- match.arg(dim)
  l <- length(lst)
  
  if (l == 0){
    stop("'lst' must contain at least one element.")
  }
  
  if (dim == "col"){
    matrix(unlist(lst), ncol = l)
  } else {
    matrix(unlist(lst), nrow = l, byrow = TRUE)
  }
}


#' Convert a matrix to a list
#'
#' @param mat A matrix.
#'
#' @return A list with elements corresponding to rows of `mat`.
#'
#' @examples #matrix2list(matrix(c(1, 2, 3, 4, 5, 6), ncol=2))
#' @keywords internal
matrix2list <- function(mat){
  split(mat, rep(1:nrow(mat), times = ncol(mat)))
}


#' Check the simulation X matrix
#'
#' @param X Covariate matrix.
#' @param n Number of observations.
#' @param p Number of covariates.
#'
#' @return Returns TRUE if X is a matrix with dimension n * p. Otherwise an error is raised.
#'
#' @examples #check_X_matrix(matrix(c(1, 2, 3, 4, 5, 6), ncol=2), n=3, p=2)
#' @keywords internal
check_X_matrix <- function(X, n, p){
  cond_1 <- is.matrix(X)
  
  if (cond_1){
    cond_2 <- all.equal(dim(X), c(n, p))
  } else {
    cond_2 <- FALSE
  }
  
  if (cond_1 & cond_2){
    return(TRUE)
  } else {
    stop(paste0("X must be a matrix with ", deparse(substitute(n)),
                " rows and ", deparse(substitute(p)), " columns."))
  }
}


#' Create cross-validation folds
#'
#' @description Utility function to create folds of data, used in cross-validation proceidures.
#' The implementation is from the `gbex` `R` package
#'
#' @param y Numerical vector of observations
#' @param num_folds Number of folds to create.
#' @param stratified Logical value. If `TRUE`, the folds are stratified along `rank(y)`.
#'
#' @return Vector of indices of the assigned folds for each observation.
#' @export
#'
#' @examples make_folds(rnorm(30), 5)
make_folds <- function(y, num_folds, stratified=FALSE){
  n = length(y)
  if(stratified) {
    folds_matrix <- sapply(1:ceiling(n/num_folds), function(i) {
      sample(1:num_folds)
    })
    folds_vector <- folds_matrix[1:n]
    folds <- folds_vector[rank(-y)]
  } else {
    index_shuffled = sample(1:n)
    folds = cut(seq(1, length(index_shuffled)), breaks = num_folds,
                labels = F)[order(index_shuffled)]
  }
  return(folds)
}


#' Covariate lagged replication for temporal dependence
#'
#' @param X Covariate matrix.
#' @param max_lag Integer giving the maximum lag (i.e. the number of temporal dependence steps).
#' @param drop_present Whether to drop the "present" features (bool).
#'
#' @return Matrix with the original columns replicated, and shifted by `1:max_lag` if `drop_present==TRUE` (default)
#' or by `0:max_lag` if `drop_present==FALSE`.
#' @export
#'
#' @examples lagged_features(matrix(seq(20), ncol=2), max_lag=3, drop_present=TRUE)
lagged_features <- function(X, max_lag, drop_present=TRUE){
  if(max_lag>=nrow(X)){
    stop("The 'max_lag' should be smaller than 'nrow(X)' in 'lagged_features'.")
  }
  n <- nrow(X)
  p <- ncol(X)
  Xl <- matrix(as.double(NA), nrow=n-max_lag, ncol=p*(max_lag+1))
  for(i in 0:max_lag){
    Xl[, (p*i+(1:p))] <- X[(max_lag+1-i):(n-i), , drop=F]
  }
  if(drop_present){
    Xl <- Xl[, (p+1):(p*(max_lag+1)), drop=F]
  }
  return(Xl)
}


#' Insert value in vector
#'
#' @param vect A 1-D vector.
#' @param val A value to insert in the vector.
#' @param ind The index at which to insert the value in the vector,
#' must be an integer between `1` and `length(vect) + 1`.
#'
#' @return A 1-D vector of length `length(vect) + 1`,
#' with `val` inserted at position `ind` in the original `vect`.
#' @export
#'
#' @examples vector_insert(c(2, 7, 3, 8), val=5, ind=3)
vector_insert <- function(vect, val, ind){
  n <- length(vect)
  if(ind<1 | ind>(n+1)){
    stop("In 'vector_insert': 'ind' must be an integer between 1 and (length(vect) + 1).")
  }
  if(ind == 1){
    return(c(val, vect))
  }
  if(ind == (n+1)){
    return(c(vect, val))
  }
  return(c(vect[1:(ind-1)], val, vect[ind:n]))
}


# ==== Parallel helpers ====

#' Get doFuture operator
#'
#' @param strategy One of `"sequential"` (default), `"multisession"`, `"multicore"`, or `"mixed"`.
#'
#' @return Returns the appropriate operator to use in a [foreach::foreach()] loop.
#' The \code{\link[foreach]{\%do\%}} operator is returned if `strategy=="sequential"`.
#' Otherwise, the \code{\link[foreach]{\%dopar\%}} operator is returned.
#' @export
#' @importFrom foreach %do% %dopar%
#'
#' @examples `%fun%` <- get_doFuture_operator("sequential")
get_doFuture_operator <- function(strategy=c("sequential", "multisession", "multicore", "mixed")){
  
  strategy <- match.arg(strategy)
  
  if(strategy == "sequential"){
    return(foreach::`%do%`)
  } else {
    return(foreach::`%dopar%`)
  }
}


#' Set a doFuture execution strategy
#'
#' @param strategy One of `"sequential"` (default), `"multisession"`, `"multicore"`, or `"mixed"`.
#' @param n_workers A positive numeric scalar or a function specifying the maximum number of parallel futures
#' that can be active at the same time before blocking.
#' If a function, it is called without arguments when the future is created and its value is used to configure the workers.
#' The function should return a numeric scalar.
#' Defaults to [future::availableCores()]`-1` if `NULL` (default), with `"multicore"` constraint in the relevant case.
#' Ignored if `strategy=="sequential"`.
#'
#' @return The corresponding [get_doFuture_operator()] operator to use in a [foreach::foreach()] loop.
#' @export
#' @importFrom foreach %do% %dopar%
#' @importFrom future availableCores plan sequential multisession multicore tweak
#' @importFrom doFuture registerDoFuture
#'
#' @examples \dontrun{
#' `%fun%` <- set_doFuture_strategy("multisession", n_workers=3)
#' # perform foreach::foreach loop
#' end_doFuture_strategy()
#' }
set_doFuture_strategy <- function(strategy=c("sequential", "multisession", "multicore", "mixed"),
                                  n_workers=NULL){
  strategy <- match.arg(strategy)
  
  doFuture::registerDoFuture()
  if(strategy == "sequential"){
    future::plan(future::sequential)
    
  } else if (strategy == "multisession"){
    if(is.null(n_workers)){
      n_workers <- max(future::availableCores() - 1, 1)
    }
    future::plan(future::multisession, workers = n_workers)
    
  } else if (strategy == "multicore"){
    if(is.null(n_workers)){
      n_workers <- max(future::availableCores(constraints = "multicore") - 1, 1)
    }
    future::plan(future::multicore, workers = n_workers)
    
  } else if (strategy == "mixed"){
    if(is.null(n_workers)){
      n_workers <- max(future::availableCores() - 1, 1)
    }
    strategy_1 <- future::tweak(future::sequential)
    strategy_2 <- future::tweak(future::multisession, workers = n_workers)
    future::plan(list(strategy_1, strategy_2))
  }
  return(get_doFuture_operator(strategy))
}


#' End the currently set doFuture strategy
#'
#' @description Resets the default strategy using `future::plan("default")`.
#'
#' @export
#' @importFrom future plan
#'
#' @examples \dontrun{
#' `%fun%` <- set_doFuture_strategy("multisession", n_workers=3)
#' # perform foreach::foreach loop
#' end_doFuture_strategy()
#' }
end_doFuture_strategy <- function(){
  
  future::plan("default")
}


#' Start a doParallel execution strategy
#'
#' @param strategy One of `"sequential"` (default) or `"parallel"`.
#' @param n_workers Number of parallel workers as an integer.
#' Defaults to [parallel::detectCores()]`-1` if `NULL` (default).
#' Ignored if `strategy=="sequential"`.
#'
#' @return A named list containing:
#' \itemize{
#' \item{par_operator}{the relevant [foreach::foreach()] loop operator,}
#' \item{cl}{the cluster object.}
#' }
#' @importFrom foreach %do% %dopar%
#' @importFrom parallel detectCores makeCluster
#' @importFrom doParallel registerDoParallel
#'
#' @examples \dontrun{
#' cl <- start_doParallel_strategy("parallel", n_workers=3)
#' stop_doParallel_strategy("parallel", cl)
#' }
#' @keywords internal
start_doParallel_strategy <- function(strategy=c("sequential", "parallel"),
                                      n_workers=NULL){
  
  strategy <- match.arg(strategy)
  
  if(is.null(n_workers)){
    n_workers <- max(parallel::detectCores() - 1, 1)
  }
  if(strategy=="parallel"){
    cl <- parallel::makeCluster(n_workers)
    doParallel::registerDoParallel(cl)
    `%fun%` <- foreach::`%dopar%`
  } else {
    cl <- NULL
    `%fun%` <- foreach::`%do%`
  }
  return(list(par_operator=`%fun%`, cl=cl))
}


#' Stop the current doParallel strategy
#'
#' @description Stops the given cluster, using [parallel::stopCluster()], if `strategy=="parallel"`.
#'
#' @param strategy One of `"sequential"` (default) or `"parallel"`.
#' @param cl Cluster object, returned by [start_doParallel_strategy()], called with the same `strategy`.
#'
#' @importFrom parallel stopCluster
#'
#' @examples \dontrun{
#' cl <- start_doParallel_strategy("parallel", n_workers=3)
#' stop_doParallel_strategy("parallel", cl)
#' }
#' @keywords internal
stop_doParallel_strategy <- function(strategy=c("sequential", "parallel"), cl){
  
  strategy <- match.arg(strategy)
  if(strategy=="parallel"){
    parallel::stopCluster(cl)
  }
}


#' Excess Probability Predictions
#'
#' @description A generic function (method) for excess probability predictions from various fitted EQR models. 
#' The function invokes particular methods which depend on the class of the first argument.
#'
#' @param object A model object for which excess probability prediction is desired.
#' @param ... additional model-specific arguments affecting the predictions produced. 
#' See the corresponding method documentation.
#'
#' @export
excess_probability <- function(object, ...){
  UseMethod("excess_probability")
}


