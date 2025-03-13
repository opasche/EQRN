
.onLoad <- function(libname, pkgname) {
  
  onload_backend_installer()
  
  invisible(TRUE)
}


#' Install Torch Backend
#' 
#' @description This function can be called just after installing the EQRN package. 
#' Calling `EQRN::install_backend()` installs the necessary LibTorch and LibLantern backends of the 
#' [`torch`](https://torch.mlverse.org/) dependency by calling [torch::install_torch()]. 
#' See <https://torch.mlverse.org/docs/articles/installation.html> for more details and troubleshooting. 
#' Calling this function shouldn't be necessary in interactive environments, as loading EQRN 
#' (e.g. with `library(EQRN)` or with any `EQRN::fct()`) should do it automatically (via `.onLoad()`). 
#' This bahaviour is inherited from the [`torch`](https://torch.mlverse.org/) package.
#'
#' @param ... Arguments passed to [torch::install_torch()].
#' @import torch
#'
#' @return No return value.
#' @export
install_backend <- function(...) {
  torch::install_torch(...)
}


#' On-Load Torch Backend Internal Install helper
#'
#' @param ... Arguments passed to [torch::install_torch()].
#' @return No return value.
#' @import torch
#' @keywords internal
onload_backend_installer <- function(...) {
  if(!torch::torch_is_installed()){
    install_backend(...)
    # install_backend(..., .inform_restart = FALSE)
  }
}



