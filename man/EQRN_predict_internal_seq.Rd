% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/EQRN_seq.R
\name{EQRN_predict_internal_seq}
\alias{EQRN_predict_internal_seq}
\title{Internal predict function for an EQRN_seq fitted object}
\usage{
EQRN_predict_internal_seq(
  fit_eqrn,
  X,
  Y,
  prob_lvl_predict,
  intermediate_quantiles,
  interm_lvl,
  crop_predictions = FALSE,
  seq_len = fit_eqrn$seq_len,
  device = default_device()
)
}
\arguments{
\item{fit_eqrn}{Fitted \code{"EQRN_seq"} object.}

\item{X}{Matrix of covariates to predict the corresponding response's conditional quantiles.}

\item{Y}{Response variable vector corresponding to the rows of \code{X}.}

\item{prob_lvl_predict}{Probability level at which to predict the conditional quantile.}

\item{intermediate_quantiles}{Vector of intermediate conditional quantiles at level \code{fit_eqrn$interm_lvl}.}

\item{interm_lvl}{Optional, checks that \code{interm_lvl == fit_eqrn$interm_lvl}.}

\item{crop_predictions}{Whether to crop out the fist \code{seq_len} observations (which are \code{NA}) from the returned vector}

\item{seq_len}{Data sequence length (i.e. number of past observations) used to predict each response quantile.
By default, the training \code{fit_eqrn$seq_len} is used.}

\item{device}{(optional) A \code{\link[torch:torch_device]{torch::torch_device()}}. Defaults to \code{\link[=default_device]{default_device()}}.}
}
\value{
Vector of length \code{nrow(X)} (or \code{nrow(X)-seq_len} if \code{crop_predictions})
containing the conditional quantile estimates of the response associated to each covariate observation at each probability level.
}
\description{
Internal predict function for an EQRN_seq fitted object
}
\keyword{internal}
