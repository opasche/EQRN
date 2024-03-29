% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/accuracy_metrics.R
\name{prediction_bias}
\alias{prediction_bias}
\title{Prediction bias}
\usage{
prediction_bias(y, y_hat, square_bias = FALSE, na.rm = FALSE)
}
\arguments{
\item{y}{Vector of observations or ground-truths.}

\item{y_hat}{Vector of predictions.}

\item{square_bias}{Whether to return the square bias (bool); defaults to \code{FALSE}.}

\item{na.rm}{A logical value indicating whether \code{NA} values should be stripped before the computation proceeds.}
}
\value{
The (square) bias of the predictions \code{y_hat} for \code{y}.
}
\description{
Prediction bias
}
\examples{
prediction_bias(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
}
