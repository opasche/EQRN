% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/accuracy_metrics.R
\name{prediction_residual_variance}
\alias{prediction_residual_variance}
\title{Prediction residual variance}
\usage{
prediction_residual_variance(y, y_hat, na.rm = FALSE)
}
\arguments{
\item{y}{Vector of observations or ground-truths.}

\item{y_hat}{Vector of predictions.}

\item{na.rm}{A logical value indicating whether \code{NA} values should be stripped before the computation proceeds.}
}
\value{
The residual variance of the predictions \code{y_hat} for \code{y}.
}
\description{
Prediction residual variance
}
\examples{
prediction_residual_variance(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
}
