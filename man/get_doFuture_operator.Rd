% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{get_doFuture_operator}
\alias{get_doFuture_operator}
\title{Get doFuture operator}
\usage{
get_doFuture_operator(
  strategy = c("sequential", "multisession", "multicore", "mixed")
)
}
\arguments{
\item{strategy}{One of \code{"sequential"} (default), \code{"multisession"}, \code{"multicore"}, or \code{"mixed"}.}
}
\value{
Returns the appropriate operator to use in a \code{\link[foreach:foreach]{foreach::foreach()}} loop.
The \code{\link[foreach]{\%do\%}} operator is returned if \code{strategy=="sequential"}.
Otherwise, the \code{\link[foreach]{\%dopar\%}} operator is returned.
}
\description{
Get doFuture operator
}
\examples{
`\%fun\%` <- get_doFuture_operator("sequential")
}
