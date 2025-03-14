% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/EVT_utils.R
\name{unconditional_train_valid_GPD_loss}
\alias{unconditional_train_valid_GPD_loss}
\title{Unconditional GPD MLEs and their train-validation likelihoods}
\usage{
unconditional_train_valid_GPD_loss(Y_train, interm_lvl, Y_valid)
}
\arguments{
\item{Y_train}{Vector of "training" observations on which to estimate the MLEs.}

\item{interm_lvl}{Probability level at which the empirical quantile should be used as the threshold.}

\item{Y_valid}{Vector of "validation" observations, on which to estimate the out of training sample GPD loss.}
}
\value{
Named list containing:
\item{scale}{GPD scale MLE inferred from the train set,}
\item{shape}{GPD shape MLE inferred from the train set,}
\item{train_loss}{the negative log-likelihoods of the MLEs over the training samples,}
\item{valid_loss}{the negative log-likelihoods of the MLEs over the validation samples.}
}
\description{
Unconditional GPD MLEs and their train-validation likelihoods
}
