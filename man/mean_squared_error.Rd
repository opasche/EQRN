% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/accuracy_metrics.R
\name{mean_squared_error}
\alias{mean_squared_error}
\title{Mean squared error}
\usage{
mean_squared_error(
  y,
  y_hat,
  return_agg = c("mean", "sum", "vector"),
  na.rm = FALSE
)
}
\arguments{
\item{y}{Vector of observations or ground-truths.}

\item{y_hat}{Vector of predictions.}

\item{return_agg}{Whether to return the \code{"mean"} (default), \code{"sum"}, or \code{"vector"} of errors.}

\item{na.rm}{A logical value indicating whether \code{NA} values should be stripped before the computation proceeds.}
}
\value{
The mean (or total or vectorial) squared error between \code{y} and \code{y_hat}.
}
\description{
Mean squared error
}
\examples{
mean_squared_error(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
}
