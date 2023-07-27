
.onLoad <- function(libname, pkgname) {
  
  onload_backend_installer()
  
  invisible(TRUE)
}


#' Install Torch Backend
#' 
#' @description This function can be called just after installing the EQRN package. 
#' It installs the necessary LibTorch backend by calling [torch::install_torch()]. 
#' See <https://torch.mlverse.org/docs/articles/installation.html> for more details.
#' Calling this function shouldn't be necessary in most cases, as loading EQRN 
#' (e.g. with `library(EQRN)` or `EQRN::fct()`) should do it automatically (via `.onLoad()`).
#'
#' @param ... Arguments passed to [torch::install_torch()].
#' @import torch
#'
#' @export
#' 
#' @examples 
#' \dontrun{
#' EQRN::install_backend()
#' }
install_backend <- function(...) {
  torch::install_torch(...)
}


#' On-Load Torch Backend Internal Install helper
#'
#' @param ... Arguments passed to [torch::install_torch()].
#' @import torch
onload_backend_installer <- function(...) {
  if(!torch::torch_is_installed()){
    install_backend(...)
    # install_backend(..., .inform_restart = FALSE)
  }
}



