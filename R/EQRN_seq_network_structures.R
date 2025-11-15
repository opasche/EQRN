
#' Recurrent network module for GPD parameter prediction
#'
#' @description
#' A recurrent neural network as a [`torch::nn_module`],
#' designed for generalized Pareto distribution parameter prediction, with sequential dependence.
#'
#' @param type the type of recurrent architecture, can be one of `"lstm"` (default) or `"gru"`,
#' @param nb_input_features the input size (i.e. the number of features),
#' @param hidden_size the dimension of the hidden latent state variables in the recurrent network,
#' @param num_layers the number of recurrent layers,
#' @param dropout probability parameter for dropout before each hidden layer for regularization during training,
#' @param shape_fixed whether the shape estimate depends on the covariates or not (bool),
#' @param device a [torch::torch_device()] for an internal constant vector. Defaults to [default_device()].
#'
#' @return The specified recurrent GPD network as a [`torch::nn_module`].
#' @export
#' @importFrom magrittr %>%
#'
#' @details
#' The constructor allows specifying:
#' \describe{
#' \item{type}{the type of recurrent architecture, can be one of `"lstm"` (default) or `"gru"`,}
#' \item{nb_input_features}{the input size (i.e. the number of features),}
#' \item{hidden_size}{the dimension of the hidden latent state variables in the recurrent network,}
#' \item{num_layers}{the number of recurrent layers,}
#' \item{dropout}{probability parameter for dropout before each hidden layer for regularization during training,}
#' \item{shape_fixed}{whether the shape estimate depends on the covariates or not (bool),}
#' \item{device}{a [torch::torch_device()] for an internal constant vector. Defaults to [default_device()].}
#' }
Recurrent_GPD_net <- torch::nn_module(
  "Recurrent_GPD_net",
  initialize = function(type=c("lstm","gru"), nb_input_features, hidden_size,
                        num_layers=1, dropout=0, shape_fixed=FALSE, device=EQRN::default_device()) {
    
    self$device <- device
    self$type <- match.arg(type)
    self$num_layers <- num_layers
    self$shape_fixed <- shape_fixed
    
    self$rnn <- if (self$type == "gru") {
      torch::nn_gru(input_size = nb_input_features, hidden_size = hidden_size,
                    num_layers = num_layers, dropout = dropout, batch_first = TRUE)
    } else {
      torch::nn_lstm(input_size = nb_input_features, hidden_size = hidden_size,
                     num_layers = num_layers, dropout = dropout, batch_first = TRUE)
    }
    
    if(self$shape_fixed){
      self$FC_scale <- torch::nn_linear(hidden_size, 1)
      self$FC_shape <- torch::nn_linear(1, 1, bias = FALSE)
    }else{
      self$FC_out <- torch::nn_linear(hidden_size, 2)
    }
  },
  
  forward = function(x) {
    
    # list of [output, hidden(, cell)]
    # we keep the output, which is of size (batch_size, n_timesteps, hidden_size)
    x <- self$rnn(x)[[1]]
    
    # from the output, we only want the final timestep
    x <- x[ , dim(x)[2], ]# shape now is (batch_size, hidden_size)
    
    
    if(self$shape_fixed){
      x <- x %>% self$FC_scale()
      xi <- torch::torch_ones(c(x$shape[1],1), device=self$device, requires_grad=FALSE)
      xi <- xi %>% self$FC_shape()
      
      x <- torch::torch_cat(list(x$exp(), xi), dim = 2)
      #x <- torch::torch_cat(list(x$exp(), torch::torch_tanh(xi) * 0.6 + 0.1), dim = 2)
    }else{
      # final shape then is (batch_size, 2)
      x <- x %>% self$FC_out()
      
      s <- torch::torch_split(x, 1, dim = 2)
      x <- torch::torch_cat(list(s[[1]]$exp(), torch::torch_tanh(s[[2]]) * 0.6 + 0.1), dim = 2)
    }
    x
  }
)


#' Recurrent quantile regression neural network module
#'
#' @description
#' A recurrent neural network as a [`torch::nn_module`],
#' designed for quantile regression.
#'
#' @param type the type of recurrent architecture, can be one of `"lstm"` (default) or `"gru"`,
#' @param nb_input_features the input size (i.e. the number of features),
#' @param hidden_size the dimension of the hidden latent state variables in the recurrent network,
#' @param num_layers the number of recurrent layers,
#' @param dropout probability parameter for dropout before each hidden layer for regularization during training.
#'
#' @return The specified recurrent QRN as a [`torch::nn_module`].
#' @export
#' @importFrom magrittr %>%
#'
#' @details
#' The constructor allows specifying:
#' \describe{
#' \item{type}{the type of recurrent architecture, can be one of `"lstm"` (default) or `"gru"`,}
#' \item{nb_input_features}{the input size (i.e. the number of features),}
#' \item{hidden_size}{the dimension of the hidden latent state variables in the recurrent network,}
#' \item{num_layers}{the number of recurrent layers,}
#' \item{dropout}{probability parameter for dropout before each hidden layer for regularization during training.}
#' }
QRNN_RNN_net <- torch::nn_module(
  "QRNN_RNN_net",
  initialize = function(type=c("lstm","gru"), nb_input_features, hidden_size, num_layers=1, dropout=0) {
    
    self$type <- match.arg(type)
    self$num_layers <- num_layers
    
    self$rnn <- if (self$type == "gru") {
      torch::nn_gru(input_size = nb_input_features, hidden_size = hidden_size,
                    num_layers = num_layers, dropout = dropout, batch_first = TRUE)
    } else {
      torch::nn_lstm(input_size = nb_input_features, hidden_size = hidden_size,
                     num_layers = num_layers, dropout = dropout, batch_first = TRUE)
    }
    
    self$FC_out <- torch::nn_linear(hidden_size, 1)
    
  },
  
  forward = function(x) {
    
    # list of [output, hidden(, cell)]
    # we keep the output, which is of size (batch_size, n_timesteps, hidden_size)
    x <- self$rnn(x)[[1]]
    
    # from the output, we only want the final timestep
    x <- x[ , dim(x)[2], ]# shape now is (batch_size, hidden_size)
    
    # feed this to a single output neuron
    # final shape then is (batch_size, 1)
    x <- x %>% self$FC_out()
    x
  }
)

