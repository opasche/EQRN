
.onLoad <- function(libname, pkgname) {
  
  onload_backend_installer()
  
  invisible(TRUE)
}


#' Install Torch Backend Libraries
#' 
#' @description This function can be called just after installing the EQRN package. 
#' Calling `EQRN::install_backend()` installs the necessary LibTorch and LibLantern backend libraries of the 
#' [`torch`](https://torch.mlverse.org/) dependency by calling [torch::install_torch()]. 
#' See <https://torch.mlverse.org/docs/articles/installation> for more details and troubleshooting. 
#' Calling this function shouldn't be necessary in interactive environments, as loading EQRN 
#' (e.g. with `library(EQRN)` or with any `EQRN::fct()`) should prompt to do it automatically (via `.onLoad()`). 
#' This behaviour is inherited from the [`torch`](https://torch.mlverse.org/) package.
#'
#' @param ... Arguments passed to [torch::install_torch()].
#' @import torch
#'
#' @return No return value.
#' @export
install_backend <- function(...) {
  torch::install_torch(...)
  ensure_backend_installed(behaviour="error")
}


#' Check if Torch Backend Libraries are Installed
#'
#' @param ... Optional parameters passed to [torch::torch_is_installed()].
#' @return Boolean indicating whether the LibTorch and LibLantern backend libraries are installed. 
#' @import torch
#' @export
#'
#' @examples 
#' if(backend_is_installed()){
#'   cat("torch's LibTorch and LibLantern backend libraries are installed!\n")
#' }
backend_is_installed <- function(...) {
  torch::torch_is_installed(...)
}


#' Ensure Torch Backend Libraries are Installed
#'
#' @param behaviour one of "error" or "bool".
#' @param ... Optional parameters passed to [torch::torch_is_installed()].
#' @return Boolean indicating whether the LibTorch and LibLantern backend libraries are installed. 
#' If `behaviour = "error"`, an error is raised if the backend libraries are not installed.
#' @import torch
#' @keywords internal
ensure_backend_installed <- function(behaviour = c("error", "warn", "message", "bool"), ...) {
  behaviour <- match.arg(behaviour)
  messg <- paste0("LibTorch and LibLantern backend libraries are not (correctly) installed. ",
                  "Please run EQRN::install_backend() to install them. ",
                  "They are required for EQRN's `torch` dependency to function properly. ",
                  "If the error persists, please refer to <https://torch.mlverse.org/docs/articles/installation>.")
  if(!backend_is_installed(...)){
    if(behaviour == "error"){
      stop(messg, call. = FALSE)
    }
    if(behaviour == "warn"){
      warning(messg, call. = FALSE)
    }
    if(behaviour == "message"){
      message(messg)
    }
    return(FALSE)
    
  }else{
    return(TRUE)
  }
}


#' On-Load Torch Backend Internal Install helper
#'
#' @param ... Arguments passed to [torch::install_torch()].
#' @return No return value.
#' @import torch
#' @keywords internal
onload_backend_installer <- function(...) {
  
  is_interactive <- interactive() || 
    "JPY_PARENT_PID" %in% names(Sys.getenv()) ||
    identical(getOption("jupyter.in_kernel"), TRUE)
  
  # Only install if not explicitly disabled or if requested explicitly by env. vars
  do_install <- (is_interactive && (Sys.getenv("TORCH_INSTALL", unset = 2) != 0) && (Sys.getenv("EQRN_INSTALL_BACKEND", unset = 2) != 0)) ||
    (Sys.getenv("TORCH_INSTALL", unset = 2) == "1") || (Sys.getenv("EQRN_INSTALL_BACKEND", unset = 2) == 1)
  
  force_install <- (Sys.getenv("EQRN_INSTALL_NOPROMPT", unset = 2) == 1)
  
  promp_msg <- paste0("LibTorch and LibLantern backend libraries need to be installed ",
                      "for EQRN's `torch` dependency to work properly. ",
                      "Do you want to install them now?")
  
  if(do_install || force_install){
    
    if(!backend_is_installed()){
      
      if(is_interactive && !force_install){
        response <- utils::askYesNo(msg = promp_msg)
        if(is.na(response) || (!response)){
          stop("Backend installation aborted. Call EQRN::install_backend() manually, if needed.", 
               call. = FALSE)
        }
      }
      
      install_backend(...)
      # install_backend(..., .inform_restart = FALSE)
    }
    
  }
}



