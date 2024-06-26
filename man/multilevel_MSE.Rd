% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/accuracy_metrics.R
\name{multilevel_MSE}
\alias{multilevel_MSE}
\title{Multilevel quantile MSEs}
\usage{
multilevel_MSE(
  True_Q,
  Pred_Q,
  proba_levels,
  prefix = "",
  na.rm = FALSE,
  give_names = TRUE,
  sd = FALSE
)
}
\arguments{
\item{True_Q}{Matrix of size \code{n_obs} times \code{proba_levels},
whose columns are the vectors of ground-truths at each \code{proba_levels} and
each row corresponds to an observation or realisation.}

\item{Pred_Q}{Matrix of the same size as \code{True_Q},
whose columns are the predictions at each \code{proba_levels} and
each row corresponds to an observation or realisation.}

\item{proba_levels}{Vector of probability levels at which the predictions were made.
Must be of length \code{ncol(Pred_Q)}.}

\item{prefix}{A string prefix to add to the output's names (if \code{give_names} is \code{TRUE}).}

\item{na.rm}{A logical value indicating whether \code{NA} values should be stripped before the computation proceeds.}

\item{give_names}{Whether to name the output MSEs (bool).}

\item{sd}{Whether to return the squared error standard deviation (bool).}
}
\value{
A vector of length \code{length(proba_levels)} giving the mean square errors
between each respective columns of \code{True_Q} and \code{Pred_Q}.
If \code{give_names} is \code{TRUE}, the output vector is named \code{paste0(prefix, "MSE_q", proba_levels)}.
If \code{sd==TRUE} a named list is instead returned, containing the \code{"MSEs"} described above and
\code{"SDs"}, their standard deviations.
}
\description{
Multilevel version of \code{\link[=mean_squared_error]{mean_squared_error()}}.
}
