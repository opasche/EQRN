% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/EQRN.R
\name{predict.EQRN_iid}
\alias{predict.EQRN_iid}
\title{Predict method for an EQRN_iid fitted object}
\usage{
\method{predict}{EQRN_iid}(object, ...)
}
\arguments{
\item{object}{Fitted \code{"EQRN_iid"} object.}

\item{...}{
  Arguments passed on to \code{\link[=EQRN_predict]{EQRN_predict}}
  \describe{
    \item{\code{X}}{Matrix of covariates to predict the corresponding response's conditional quantiles.}
    \item{\code{prob_lvls_predict}}{Vector of probability levels at which to predict the conditional quantiles.}
    \item{\code{intermediate_quantiles}}{Vector of intermediate conditional quantiles at level \code{fit_eqrn$interm_lvl}.}
    \item{\code{interm_lvl}}{Optional, checks that \code{interm_lvl == fit_eqrn$interm_lvl}.}
    \item{\code{device}}{(optional) A \code{\link[torch:torch_device]{torch::torch_device()}}. Defaults to \code{\link[=default_device]{default_device()}}.}
  }}
}
\value{
Matrix of size \code{nrow(X)} times \code{prob_lvls_predict}
containing the conditional quantile estimates of the response associated to each covariate observation at each probability level.
Simplifies to a vector if \code{length(prob_lvls_predict)==1}.
}
\description{
Predict method for an EQRN_iid fitted object
}
\details{
See \code{\link[=EQRN_predict]{EQRN_predict()}} for more details.
}
