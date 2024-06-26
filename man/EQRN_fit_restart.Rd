% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/EQRN.R
\name{EQRN_fit_restart}
\alias{EQRN_fit_restart}
\title{Wrapper for fitting EQRN with restart for stability}
\usage{
EQRN_fit_restart(
  X,
  y,
  intermediate_quantiles,
  interm_lvl,
  number_fits = 3,
  ...,
  seed = NULL,
  data_type = c("iid", "seq")
)
}
\arguments{
\item{X}{Matrix of covariates, for training.}

\item{y}{Response variable vector to model the extreme conditional quantile of, for training.}

\item{intermediate_quantiles}{Vector of intermediate conditional quantiles at level \code{interm_lvl}.}

\item{interm_lvl}{Probability level for the intermediate quantiles \code{intermediate_quantiles}.}

\item{number_fits}{Number of restarts.}

\item{...}{Other parameters given to either \code{\link[=EQRN_fit]{EQRN_fit()}} or \code{\link[=EQRN_fit_seq]{EQRN_fit_seq()}}, depending on the \code{data_type}.}

\item{seed}{Integer random seed for reproducibility in network weight initialization.}

\item{data_type}{Type of data dependence, must be one of \code{"iid"} (for iid observations) or \code{"seq"} (for sequentially dependent observations).}
}
\value{
An EQRN object of classes \code{c("EQRN_iid", "EQRN")} or \code{c("EQRN_seq", "EQRN")}, containing the fitted network,
as well as all the relevant information for its usage in other functions.
}
\description{
Wrapper for fitting EQRN with restart for stability
}
