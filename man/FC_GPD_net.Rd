% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/EQRN_network_structures.R
\name{FC_GPD_net}
\alias{FC_GPD_net}
\title{MLP module for GPD parameter prediction}
\usage{
FC_GPD_net(
  D_in,
  Hidden_vect = c(5, 5, 5),
  activation = torch::nnf_sigmoid,
  p_drop = 0,
  shape_fixed = FALSE,
  device = EQRN::default_device()
)
}
\arguments{
\item{D_in}{the input size (i.e. the number of features),}

\item{Hidden_vect}{a vector of integers whose length determines the number of layers in the neural network
and entries the number of neurons in each corresponding successive layer,}

\item{activation}{the activation function for the hidden layers
(should be either a callable function, preferably from the \code{torch} library),}

\item{p_drop}{probability parameter for dropout before each hidden layer for regularization during training,}

\item{shape_fixed}{whether the shape estimate depends on the covariates or not (bool),}

\item{device}{a \code{\link[torch:torch_device]{torch::torch_device()}} for an internal constant vector. Defaults to \code{\link[=default_device]{default_device()}}.}
}
\value{
The specified MLP GPD network as a \code{\link[torch:nn_module]{torch::nn_module}}.
}
\description{
A fully-connected network (or multi-layer perception) as a \code{\link[torch:nn_module]{torch::nn_module}},
designed for generalized Pareto distribution parameter prediction.
}
\details{
The constructor allows specifying:
\describe{
\item{D_in}{the input size (i.e. the number of features),}
\item{Hidden_vect}{a vector of integers whose length determines the number of layers in the neural network
and entries the number of neurons in each corresponding successive layer,}
\item{activation}{the activation function for the hidden layers
(should be either a callable function, preferably from the \code{torch} library),}
\item{p_drop}{probability parameter for dropout before each hidden layer for regularization during training,}
\item{shape_fixed}{whether the shape estimate depends on the covariates or not (bool),}
\item{device}{a \code{\link[torch:torch_device]{torch::torch_device()}} for an internal constant vector. Defaults to \code{\link[=default_device]{default_device()}}.}
}
}
