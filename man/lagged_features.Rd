% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{lagged_features}
\alias{lagged_features}
\title{Covariate lagged replication for temporal dependence}
\usage{
lagged_features(X, max_lag, drop_present = TRUE)
}
\arguments{
\item{X}{Covariate matrix.}

\item{max_lag}{Integer giving the maximum lag (i.e. the number of temporal dependence steps).}

\item{drop_present}{Whether to drop the "present" features (bool).}
}
\value{
Matrix with the original columns replicated, and shifted by \code{1:max_lag} if \code{drop_present==TRUE} (default)
or by \code{0:max_lag} if \code{drop_present==FALSE}.
}
\description{
Covariate lagged replication for temporal dependence
}
\examples{
lagged_features(matrix(seq(20), ncol=2), max_lag=3, drop_present=TRUE)
}
