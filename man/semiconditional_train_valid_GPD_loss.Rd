% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/EVT_utils.R
\name{semiconditional_train_valid_GPD_loss}
\alias{semiconditional_train_valid_GPD_loss}
\title{Semi-conditional GPD MLEs and their train-validation likelihoods}
\usage{
semiconditional_train_valid_GPD_loss(
  Y_train,
  Y_valid,
  interm_quant_train,
  interm_quant_valid
)
}
\arguments{
\item{Y_train}{Vector of "training" observations on which to estimate the MLEs.}

\item{Y_valid}{Vector of "validation" observations, on which to estimate the out of training sample GPD loss.}

\item{interm_quant_train}{Vector of intermediate quantiles serving as a varying threshold for each training observation.}

\item{interm_quant_valid}{Vector of intermediate quantiles serving as a varying threshold for each validation observation.}
}
\value{
Named list containing:
\itemize{
\item{scale}{GPD scale MLE inferred from the train set,}
\item{shape}{GPD shape MLE inferred from the train set,}
\item{train_loss}{the negative log-likelihoods of the MLEs over the training samples,}
\item{valid_loss}{the negative log-likelihoods of the MLEs over the validation samples.}
}
}
\description{
Semi-conditional GPD MLEs and their train-validation likelihoods
}
