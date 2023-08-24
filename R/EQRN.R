
#' Wrapper for fitting EQRN with restart for stability
#'
#' @param X Matrix of covariates, for training.
#' @param y Response variable vector to model the extreme conditional quantile of, for training.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `interm_lvl`.
#' @param interm_lvl Probability level for the intermediate quantiles `intermediate_quantiles`.
#' @param number_fits Number of restarts.
#' @param ... Other parameters given to either [EQRN_fit()] or [EQRN_fit_seq()], depending on the `data_type`.
#' @param seed Integer random seed for reproducibility in network weight initialization.
#' @param data_type Type of data dependence, must be one of `"iid"` (for iid observations) or `"seq"` (for sequentially dependent observations).
#'
#' @return An EQRN object of classes `c("EQRN_iid", "EQRN")` or `c("EQRN_seq", "EQRN")`, containing the fitted network,
#' as well as all the relevant information for its usage in other functions.
#' @export
#'
#' @examples #TODO
EQRN_fit_restart <- function(X, y, intermediate_quantiles, interm_lvl, number_fits=3, ..., seed=NULL, data_type=c("iid","seq")){#TODO: force_trainloss_select arg
  #
  data_type <- match.arg(data_type)
  if(number_fits<1){stop("'number_fits' must be at least 1 in 'EQRN_fit_restart'.")}
  
  if(!is.null(seed)){torch::torch_manual_seed(seed)}
  
  if(data_type=="seq"){
    fit_fct <- EQRN_fit_seq
  }else{
    fit_fct <- EQRN_fit
  }
  
  fit_final <- fit_fct(X, y, intermediate_quantiles, interm_lvl, ...)
  train_loss_final <- last_elem(fit_final$train_loss)
  if(is.na(train_loss_final)){
    train_loss_final <- Inf
  }
  do_validation <- !is.null(fit_final$valid_loss)
  if(do_validation){
    valid_loss_final <- last_elem(fit_final$valid_loss)
    if(is.na(valid_loss_final)){
      valid_loss_final <- Inf
    }
  }
  
  if(number_fits>=2){
    for(i in 2:number_fits){
      fit_temp <- fit_fct(X, y, intermediate_quantiles, interm_lvl, ...)
      train_loss <- last_elem(fit_temp$train_loss)
      
      if(do_validation){
        valid_loss <- last_elem(fit_temp$valid_loss)
        if(!is.na(valid_loss)){
          if(valid_loss < valid_loss_final){
            fit_final <- fit_temp
            valid_loss_final <- valid_loss
            train_loss_final <- train_loss
          }
        }
        
      } else {
        if(!is.na(train_loss)){
          if(train_loss < train_loss_final){
            fit_final <- fit_temp
            train_loss_final <- train_loss
          }
        }
      }
    }
  }
  if(is.infinite(train_loss_final)){
    warning("NaN final train loss in 'EQRN_fit_restart'.")
  }
  if(do_validation){
    if(is.infinite(valid_loss_final)){
      warning("NaN final validation loss in 'EQRN_fit_restart'.")
    }
  }
  return(fit_final)
}

#' @title EQRN fit function for independent data
#'
#' @description Use the [EQRN_fit_restart()] wrapper instead, with `data_type="iid"`, for better stability using fitting restart.
#'
#' @param X Matrix of covariates, for training.
#' @param y Response variable vector to model the extreme conditional quantile of, for training.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `interm_lvl`.
#' @param interm_lvl Probability level for the intermediate quantiles `intermediate_quantiles`.
#' @param shape_fixed Whether the shape estimate depends on the covariates or not (bool).
#' @param net_structure Vector of integers whose length determines the number of layers in the neural network
#' and entries the number of neurons in each corresponding successive layer.
#' If `hidden_fct=="SSNN"`, should instead be a named list with `"scale"` and `"shape"` vectors for the two respective sub-networks.
#' Can also be a [`torch::nn_module`] network with correct input and output dimensions,
#' which overrides the `hidden_fct`, `shape_fixed` and `p_drop` arguments.
#' @param hidden_fct Activation function for the hidden layers. Can be either a callable function (preferably from the `torch` library),
#' or one of the the strings `"SNN"`, `"SSNN"` for self normalizing networks (with common or separated networks for the scale and shape estimates, respectively).
#' In the latter cases, `shape_fixed` has no effect.
#' @param p_drop Probability parameter for dropout before each hidden layer for regularization during training.
#' `alpha-dropout` is used with SNNs.
#' @param intermediate_q_feature Whether to use the `intermediate_quantiles` as an additional covariate, by appending it to the `X` matrix (bool).
#' @param learning_rate Initial learning rate for the optimizer during training of the neural network.
#' @param L2_pen L2 weight penalty parameter for regularization during training.
#' @param shape_penalty Penalty parameter for the shape estimate, to potentially regularize its variation from the fixed prior estimate.
#' @param scale_features Whether to rescale each input covariates to zero mean and unit variance before applying the network (recommended).
#' @param n_epochs Number of training epochs.
#' @param batch_size Batch size used during training.
#' @param X_valid Covariates in a validation set, or `NULL`.
#' Used for monitoring validation loss during training, enabling learning-rate decay and early stopping.
#' @param y_valid Response variable in a validation set, or `NULL`.
#' Used for monitoring validation loss during training, enabling learning-rate decay and early stopping.
#' @param quant_valid Intermediate conditional quantiles at level `interm_lvl` in a validation set, or `NULL`.
#' Used for monitoring validation loss during training, enabling learning-rate decay and early stopping.
#' @param lr_decay Learning rate decay factor.
#' @param patience_decay Number of epochs of non-improving validation loss before a learning-rate decay is performed.
#' @param min_lr Minimum learning rate, under which no more decay is performed.
#' @param patience_stop Number of epochs of non-improving validation loss before early stopping is performed.
#' @param tol Tolerance for stopping training, in case of no significant training loss improvements.
#' @param orthogonal_gpd Whether to use the orthogonal reparametrization of the estimated GPD parameters (recommended).
#' @param patience_lag The validation loss is considered to be non-improving if it is larger than on any of the previous `patience_lag` epochs.
#' @param optim_met DEPRECATED. Optimization algorithm to use during training. `"adam"` is the default.
#' @param seed Integer random seed for reproducibility in network weight initialization.
#' @param verbose Amount of information printed during training (0:nothing, 1:most important, 2:everything).
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return An EQRN object of classes `c("EQRN_iid", "EQRN")`, containing the fitted network,
#' as well as all the relevant information for its usage in other functions.
#' @export
#' @import torch
#' @importFrom coro loop
#'
#' @examples #TODO
EQRN_fit <- function(X, y, intermediate_quantiles, interm_lvl, shape_fixed=FALSE, net_structure=c(5,3,3), hidden_fct=torch::nnf_sigmoid, p_drop=0,
                     intermediate_q_feature=TRUE, learning_rate=1e-4, L2_pen=0, shape_penalty=0, scale_features=TRUE, n_epochs=500, batch_size=256,
                     X_valid=NULL, y_valid=NULL, quant_valid=NULL, lr_decay=1, patience_decay=n_epochs, min_lr=0, patience_stop=n_epochs,
                     tol=1e-6, orthogonal_gpd=TRUE, patience_lag=1, optim_met="adam", seed=NULL, verbose=2, device=default_device()){
  
  if(!is.null(seed)){torch::torch_manual_seed(seed)}
  
  data_excesses <- get_excesses(X=X, y=y, quantiles=intermediate_quantiles,
                                intermediate_q_feature=intermediate_q_feature, scale_features=scale_features, X_scaling=NULL)
  Y_excesses <- torch::torch_tensor(data_excesses$Y_excesses, device = device)
  X_feats_excesses <- torch::torch_tensor(data_excesses$X_excesses, device = device)
  X_scaling <- data_excesses$X_scaling
  
  # Data Loader
  n_train <- Y_excesses$size()[1]
  trainset <- torch::tensor_dataset(X_feats_excesses, Y_excesses)
  trainloader <- torch::dataloader(trainset, batch_size=batch_size, shuffle=TRUE)
  
  # Validation dataset (if everything needed is given)
  do_validation <- (!is.null(y_valid) & !is.null(X_valid) & !is.null(quant_valid))
  if(do_validation){
    data_valid_excesses <- get_excesses(X=X_valid, y=y_valid, quantiles=quant_valid,
                                        intermediate_q_feature=intermediate_q_feature, scale_features=scale_features, X_scaling=X_scaling)
    y_valid_ex <- torch::torch_tensor(data_valid_excesses$Y_excesses, device = device)
    X_valid_ex <- torch::torch_tensor(data_valid_excesses$X_excesses, device = device)
    
    n_valid <- y_valid_ex$size()[1]
    validset <- torch::tensor_dataset(X_valid_ex, y_valid_ex)
    validloader <- torch::dataloader(validset, batch_size=batch_size, shuffle=FALSE)
  }
  
  # Semi-conditional GPD fit (on y rescaled excesses wrt intermediate_quantiles)
  if(shape_penalty>0){
    semicond_gpd_fit <- fit_GPD_unconditional(c(data_excesses$Y_excesses), interm_lvl=NULL, thresh_quantiles=NULL)
  }else{
    semicond_gpd_fit <- NULL
  }
  
  # Instantiate network
  Dim_in <- ncol(X_feats_excesses)
  network <- instantiate_EQRN_network(net_structure=net_structure, shape_fixed=shape_fixed, D_in=Dim_in,
                                      hidden_fct=hidden_fct, p_drop=p_drop, orthogonal_gpd=orthogonal_gpd, device=device)
  
  # Optimizer
  optimizer <- setup_optimizer(network, learning_rate, L2_pen, hidden_fct, optim_met=optim_met)
  
  # Train the network
  network$train()
  
  loss_log_train <- rep(as.double(NA), n_epochs)
  nb_stable <- 0
  if(do_validation){
    loss_log_valid <- rep(as.double(NA), n_epochs)
    nb_not_improving_val <- 0
    nb_not_improving_lr <- 0
  }
  curent_lr <- learning_rate
  for (e in 1:n_epochs) {
    loss_train <- 0
    coro::loop(for (b in trainloader) {
      b <- fix_dimsimplif(b, Dim_in)#TODO: remove when fixed in torch
      # Forward pass
      net_out <- network(b[[1]])
      # Loss
      loss <- loss_GPD_tensor(net_out, b[[2]], orthogonal_gpd=orthogonal_gpd,
                              shape_penalty=shape_penalty, prior_shape=semicond_gpd_fit$shape, return_agg="mean")
      # Check for bad initialization
      while(e<2 & is.nan(loss$item())){
        network <- instantiate_EQRN_network(net_structure=net_structure, shape_fixed=shape_fixed, D_in=Dim_in,
                                            hidden_fct=hidden_fct, p_drop=p_drop, orthogonal_gpd=orthogonal_gpd, device=device)
        network$train()
        optimizer <- setup_optimizer(network, learning_rate, L2_pen, hidden_fct, optim_met=optim_met)
        net_out <- network(b[[1]])
        loss <- loss_GPD_tensor(net_out, b[[2]], orthogonal_gpd=orthogonal_gpd,
                                shape_penalty=shape_penalty, prior_shape=semicond_gpd_fit$shape, return_agg="mean")
      }
      # zero the gradients accumulated in buffers (not overwritten in pyTorch)
      optimizer$zero_grad()
      # Backward pass: compute gradient of the loss with respect to model parameters
      loss$backward()
      # make one optimizer step
      optimizer$step()
      # store loss
      loss_train <- loss_train + (b[[2]]$size()[1] * loss / n_train)$item()
    })
    # Log loss
    loss_log_train[e] <- loss_train
    if(do_validation){
      network$eval()
      loss_valid <- 0
      coro::loop(for (b in validloader) {
        b <- fix_dimsimplif(b, Dim_in)#TODO: remove when fixed in torch
        valid_out <- network(b[[1]])
        loss <- loss_GPD_tensor(valid_out, b[[2]], orthogonal_gpd=orthogonal_gpd,
                                shape_penalty=0, prior_shape=NULL, return_agg="mean")
        loss_valid <- loss_valid + (b[[2]]$size()[1] * loss / n_valid)$item()
      })
      loss_log_valid[e] <- loss_valid
      # Is the validation loss improving ?
      if(e>patience_lag){
        if(any(is.na(loss_log_valid[(e-patience_lag):e]))){
          if(is.nan(loss_log_valid[e])){
            nb_not_improving_val <- nb_not_improving_val + 1
            nb_not_improving_lr <- nb_not_improving_lr + 1
            if(verbose>1){cat("NaN validation loss at epoch:", e, "\n")}
          }
        }else{
          if(loss_log_valid[e]>(min(loss_log_valid[(e-patience_lag):(e-1)])-tol)){
            nb_not_improving_val <- nb_not_improving_val + 1
            nb_not_improving_lr <- nb_not_improving_lr + 1
          }else{
            nb_not_improving_val <- 0
            nb_not_improving_lr <- 0
          }
        }
      }
      # Learning rate decay
      if(curent_lr>min_lr & nb_not_improving_lr >= patience_decay){
        optimizer <- decay_learning_rate(optimizer,lr_decay)
        curent_lr <- curent_lr*lr_decay
        nb_not_improving_lr <- 0
      }
      if(nb_not_improving_val >= patience_stop){
        if(verbose>0){
          cat("Early stopping at epoch:", e,", average train loss:", loss_log_train[e],
              ", validation loss:", loss_log_valid[e], ", lr=", curent_lr, "\n")
        }
        break
      }
      network$train()
    }
    
    # Tolerance stop
    if(e>1){
      if(abs(loss_log_train[e]-loss_log_train[e-1])<tol){
        nb_stable <- nb_stable + 1
      } else {
        nb_stable <- 0
      }
    }
    if(nb_stable >= patience_stop){
      if(verbose>0){
        cat("Early tolerence stopping at epoch:", e,", average train loss:", loss_log_train[e])
        if(do_validation){cat(", validation loss:", loss_log_valid[e])}
        cat(", lr=", curent_lr, "\n")
      }
      break
    }
    # Print progess
    if(e %% 100 == 0 || (e == 1 || e == n_epochs)){
      if(verbose>1){
        cat("Epoch:", e, "out of", n_epochs ,", average train loss:", loss_log_train[e])
        if(do_validation){cat(", validation loss:", loss_log_valid[e], ", lr=", curent_lr)}
        cat("\n")
      }
    }
  }
  
  network$eval()
  
  fit_eqrn <- list(fit_nn = network, interm_lvl = interm_lvl, intermediate_q_feature = intermediate_q_feature,
                   train_loss = loss_log_train[1:e], X_scaling=X_scaling, orthogonal_gpd=orthogonal_gpd,
                   n_obs = length(y), n_excesses = n_train, excesses_ratio = data_excesses$excesses_ratio)
  if(do_validation){
    fit_eqrn$valid_loss <- loss_log_valid[1:e]
  }
  class(fit_eqrn) <- c("EQRN_iid", "EQRN")
  
  return(fit_eqrn)
}

#' Predict function for an EQRN_iid fitted object
#'
#' @param fit_eqrn Fitted `"EQRN_iid"` object.
#' @param X Matrix of covariates to predict the corresponding response's conditional quantiles.
#' @param quantiles_predict Vector of probability levels at which to predict the conditional quantiles.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Matrix of size `nrow(X)` times `quantiles_predict`
#' containing the conditional quantile estimates of the response associated to each covariate observation at each probability level.
#' Simplifies to a vector if `length(quantiles_predict)==1`.
#' @export
#'
#' @examples #TODO
EQRN_predict <- function(fit_eqrn, X, quantiles_predict, intermediate_quantiles, interm_lvl=fit_eqrn$interm_lvl, device=default_device()){
  
  if(length(dim(quantiles_predict))>1){
    stop("Please provide a single value or 1D vector as quantiles_predict in EQRN_predict.")
  }
  
  if(length(quantiles_predict)==1){
    return(EQRN_predict_internal(fit_eqrn, X, quantiles_predict, intermediate_quantiles, interm_lvl, device=device))
  } else if(length(quantiles_predict)>1){
    nb_quantiles_predict <- length(quantiles_predict)
    predicted_quantiles <- matrix(as.double(NA), nrow=nrow(X), ncol=nb_quantiles_predict)
    for(i in 1:nb_quantiles_predict){
      predicted_quantiles[,i] <- EQRN_predict_internal(fit_eqrn, X, quantiles_predict[i], intermediate_quantiles, interm_lvl, device=device)
    }
    return(predicted_quantiles)
  } else {
    stop("Please provide a single value or 1D vector as quantiles_predict in EQRN_predict.")
  }
}

#' Internal predict function for an EQRN_iid
#'
#' @param fit_eqrn Fitted `"EQRN_iid"` object.
#' @param X Matrix of covariates to predict the corresponding response's conditional quantiles.
#' @param quantile_predict Probability level at which to predict the conditional quantiles.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Vector of length `nrow(X)` containing the conditional quantile estimates of the response associated to each covariate observation
#' at each probability level `quantile_predict`.
#'
#' @examples #TODO
#' @keywords internal
EQRN_predict_internal <- function(fit_eqrn, X, quantile_predict, intermediate_quantiles, interm_lvl, device=default_device()){
  
  GPD_params_pred <- EQRN_predict_params(fit_eqrn, X, intermediate_quantiles,
                                         return_parametrization="classical", interm_lvl, device=device)
  sigmas <- GPD_params_pred$scales
  xis <- GPD_params_pred$shapes
  
  predicted_quantiles <- GPD_quantiles(quantile_predict, interm_lvl, intermediate_quantiles, sigmas, xis)
  
  return(predicted_quantiles)
}

#' GPD parameters prediction function for an EQRN_iid fitted object
#'
#' @param fit_eqrn Fitted `"EQRN_iid"` object.
#' @param X Matrix of covariates to predict conditional GPD parameters.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param return_parametrization Which parametrization to return the parameters in, either `"classical"` or `"orthogonal"`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Named list containing: `"scales"` and `"shapes"` as numerical vectors of length `nrow(X)`.
#' @export
#' @import torch
#' @importFrom coro loop
#'
#' @examples #TODO
EQRN_predict_params <- function(fit_eqrn, X, intermediate_quantiles=NULL, return_parametrization=c("classical","orthogonal"),
                                interm_lvl=fit_eqrn$interm_lvl, device=default_device()){
  ## 'return_parametrization' controls the desired parametrization of the output parameters
  ## (works for both "orthogonal_gpd" EQRN parametrizations, by converting if needed)
  
  return_parametrization <- match.arg(return_parametrization)
  
  if(interm_lvl!=fit_eqrn$interm_lvl){stop("EQRN intermediate quantiles interm_lvl does not match in train and predict.")}
  
  X_feats <- process_features(X, intermediate_q_feature=fit_eqrn$intermediate_q_feature,
                              intermediate_quantiles=intermediate_quantiles, X_scaling=fit_eqrn$X_scaling)$X_scaled
  X_feats <- torch::torch_tensor(X_feats, device = device)
  
  testset <- torch::tensor_dataset(X_feats)
  testloader <- torch::dataloader(testset, batch_size=batch_size_default(testset), shuffle=FALSE)
  
  network <- fit_eqrn$fit_nn
  network$eval()
  
  # GPD_params_pred <- network(X_feats)
  # scales <- as.numeric(GPD_params_pred[,1])
  # shapes <- as.numeric(GPD_params_pred[,2])
  scales <- c()
  shapes <- c()
  coro::loop(for (b in testloader) {
    b <- fix_dimsimplif(b, X_feats$shape[-1], responses=FALSE)#TODO: remove when fixed in torch
    GPD_params_pred <- network(b[[1]])
    scales <- c(scales, as.numeric(GPD_params_pred[,1]))
    shapes <- c(shapes, as.numeric(GPD_params_pred[,2]))
  })
  
  if(fit_eqrn$orthogonal_gpd & return_parametrization=="classical"){
    scales <- scales / (shapes+1.)
  }
  if((!fit_eqrn$orthogonal_gpd) & return_parametrization=="orthogonal"){
    scales <- scales * (shapes+1.)
  }
  
  return(list(scales=scales, shapes=shapes))
}

#' Tail excess probability prediction using an EQRN_iid object
#'
#' @param val Quantile value(s) used to estimate the conditional excess probability or cdf.
#' @param fit_eqrn Fitted `"EQRN_iid"` object.
#' @param X Matrix of covariates to predict the corresponding response's conditional excess probabilities.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param body_proba Value to use when the predicted conditional probability is below `interm_lvl`
#' (in which case it cannot be precisely assessed by the model).
#' If `"default"` is given (the default), `paste0(">",1-interm_lvl)` is used if `proba_type=="excess"`,
#' and `paste0("<",interm_lvl)` is used if `proba_type=="cdf"`.
#' @param proba_type Whether to return the `"excess"` probability over `val` (default) or the `"cdf"` at `val`.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Vector of probabilities (and possibly a few `body_proba` values if `val` is not large enough) of length `nrow(X)`.
#' @export
#'
#' @examples #TODO
EQRN_excess_probability <- function(val, fit_eqrn, X, intermediate_quantiles, interm_lvl=fit_eqrn$interm_lvl,
                                    body_proba="default", proba_type=c("excess","cdf"), device=default_device()){
  
  proba_type <- match.arg(proba_type)
  
  GPD_params_pred <- EQRN_predict_params(fit_eqrn, X, intermediate_quantiles,
                                         return_parametrization="classical", interm_lvl, device=device)
  sigmas <- GPD_params_pred$scales
  xis <- GPD_params_pred$shapes
  
  Probs <- GPD_excess_probability(val, sigma=sigmas, xi=xis, interm_threshold=intermediate_quantiles,
                                  threshold_p=interm_lvl, body_proba=body_proba, proba_type=proba_type)
  return(c(Probs))
}

#' Generalized Pareto likelihood loss of a EQRN_iid predictor
#'
#' @param fit_eqrn Fitted `"EQRN_iid"` object.
#' @param X Matrix of covariates.
#' @param y Response variable vector.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Negative GPD log likelihood of the conditional EQRN predicted parameters
#' over the response exceedances over the intermediate quantiles.
#' @export
#' @import torch
#' @importFrom coro loop
#'
#' @examples #TODO
compute_EQRN_GPDLoss <- function(fit_eqrn, X, y, intermediate_quantiles=NULL, interm_lvl=fit_eqrn$interm_lvl, device=default_device()){#TODO: internal_fct shared with train
  if(interm_lvl!=fit_eqrn$interm_lvl){stop("EQRN intermediate quantiles interm_lvl does not match in train and predict.")}
  data_valid_excesses <- get_excesses(X=X, y=y, quantiles=intermediate_quantiles, intermediate_q_feature=fit_eqrn$intermediate_q_feature,
                                      scale_features=fit_eqrn$X_scaling$scaling, X_scaling=fit_eqrn$X_scaling)
  y_valid_ex <- torch::torch_tensor(data_valid_excesses$Y_excesses, device = device)
  X_valid_ex <- torch::torch_tensor(data_valid_excesses$X_excesses, device = device)
  
  n_valid <- y_valid_ex$size()[1]
  validset <- torch::tensor_dataset(X_valid_ex, y_valid_ex)
  validloader <- torch::dataloader(validset, batch_size=batch_size_default(validset), shuffle=FALSE)
  network <- fit_eqrn$fit_nn
  network$eval()
  loss_valid <- 0
  coro::loop(for (b in validloader) {
    b <- fix_dimsimplif(b, X_valid_ex$shape[-1])#TODO: remove when fixed in torch
    valid_out <- network(b[[1]])
    loss <- loss_GPD_tensor(valid_out, b[[2]], orthogonal_gpd=fit_eqrn$orthogonal_gpd,
                            shape_penalty=0, prior_shape=NULL, return_agg="mean")
    loss_valid <- loss_valid + (b[[2]]$size()[1] * loss / n_valid)$item()
  })
  return(loss_valid)
}

#' Instantiates the default networks for training a EQRN_iid model
#'
#' @param net_structure Vector of integers whose length determines the number of layers in the neural network
#' and entries the number of neurons in each corresponding successive layer.
#' @param shape_fixed Whether the shape estimate depends on the covariates or not (bool).
#' @param D_in Number of covariates (including the intermediate quantile feature if used).
#' @param hidden_fct Activation function for the hidden layers. Can be either a callable function (preferably from the `torch` library),
#' or one of the the strings `"SNN"`, `"SSNN"` for self normalizing networks (with common or separated networks for the scale and shape estimates, respectively).
#' In the latter cases, `shape_fixed` has no effect.
#' @param p_drop Probability parameter for dropout before each hidden layer for regularization during training.
#' `alpha-dropout` is used with SNNs.
#' @param orthogonal_gpd Whether to use the orthogonal reparametrization of the estimated GPD parameters (recommended).
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return A `torch::nn_module` network used to regress the GPD parameters in [EQRN_fit()].
#' @import torch
#'
#' @examples #TODO
#' @keywords internal
instantiate_EQRN_network <- function(net_structure, shape_fixed, D_in, hidden_fct, p_drop=0,
                                     orthogonal_gpd=TRUE, device=default_device()){
  
  err_msg <- "Please give a valid activation function for 'hidden_fct' in EQRN or specify one of the character strings: 'SNN', 'SSNN'."
  
  if(any(class(net_structure)=="nn_module")){
    network <- net_structure
    cat("custom net is used in EQRN (be carefull with dimensions)\n")
    
  }else{
    
    if(is.character(hidden_fct)){
      if(hidden_fct=="SNN"){
        network <- FC_GPD_SNN(D_in=D_in, Hidden_vect=net_structure, p_drop=p_drop)
      } else if (hidden_fct=="SSNN"){
        network <- Separated_GPD_SNN(D_in=D_in, Hidden_vect_scale=net_structure$scale, Hidden_vect_shape=net_structure$shape, p_drop=p_drop)
      } else {
        stop(err_msg)
      }
      
    } else if (is.function(hidden_fct)) {
      network <- FC_GPD_net(D_in=D_in, Hidden_vect=net_structure, activation=hidden_fct, p_drop=p_drop,
                            shape_fixed=shape_fixed, device=device)
      
    } else {
      stop(err_msg)
    }
    
  }
  network$to(device = device)
  return(network)
}

#' Instantiate an optimizer for training an EQRN_iid network
#'
#' @param network A `torch::nn_module` network to be trained in [EQRN_fit()].
#' @param learning_rate Initial learning rate for the optimizer during training of the neural network.
#' @param L2_pen L2 weight penalty parameter for regularization during training.
#' @param hidden_fct Activation function for the hidden layers. Can be either a callable function (preferably from the `torch` library),
#' or one of the the strings `"SNN"`, `"SSNN"` for self normalizing networks (with common or separated networks for the scale and shape estimates, respectively).
#' This will affect the default choice of optimizer.
#' @param optim_met DEPRECATED. Optimization algorithm to use during training. `"adam"` is the default.
#'
#' @return A `torch::optimizer` object used in [EQRN_fit()] for training.
#' @import torch
#'
#' @examples #TODO
#' @keywords internal
setup_optimizer <- function(network, learning_rate, L2_pen, hidden_fct, optim_met="adam"){
  
  err_msg <- "Please give a valid activation function for 'hidden_fct' in EQRN or specify one of the character strings: 'SNN', 'SSNN'."
  
  if(optim_met!="adam"){stop("Other optim methods are deprecated.")}
  
  if(is.character(hidden_fct)){
    if(hidden_fct=="SNN"){
      optimizer <- torch::optim_adam(network$parameters, lr=learning_rate, betas=c(0.9, 0.99), eps=0.01, weight_decay=L2_pen)
      #optimizer <- torch::optim_adadelta(network$parameters, lr=learning_rate, rho=0.95, eps=1e-6, weight_decay=L2_pen)
    } else if (hidden_fct=="SSNN"){
      optimizer <- torch::optim_adam(network$parameters, lr=learning_rate, betas=c(0.9, 0.99), eps=0.01, weight_decay=L2_pen)
    } else {
      stop(err_msg)
    }
    
  } else if (is.function(hidden_fct)) {
    optimizer <- torch::optim_adam(network$parameters, lr=learning_rate, weight_decay=L2_pen)
    
  } else {
    stop(err_msg)
  }
  return(optimizer)
}

#' Performs a learning rate decay step on an optimizer
#'
#' @param optimizer A `torch::optimizer` object.
#' @param decay_rate Learning rate decay factor.
#'
#' @return The `optimizer` with a decayed learning rate.
#' @import torch
#'
#' @examples #TODO
#' @keywords internal
decay_learning_rate <- function(optimizer, decay_rate){
  for (i in seq_along(optimizer$param_groups)){
    optimizer$param_groups[[i]]$lr <- decay_rate * optimizer$param_groups[[i]]$lr
  }
  optimizer
}

#' Save an EQRN object on disc
#'
#' @description Creates a folder named `name` and located in `path`, containing binary save files,
#' so that the given `"EQRN"` object `fit_eqrn` can be loaded back in memory from disc using [EQRN_load()].
#'
#' @param fit_eqrn An `"EQRN"` object
#' @param path Path to save folder as a string.
#' @param name String name of the save.
#' @param no_warning Whether to silence the warning raised if a save folder needed beeing created (bool).
#'
#' @return Nothing
#' @export
#' @import torch
#' @importFrom utils packageVersion
#'
#' @examples #TODO
EQRN_save <- function(fit_eqrn, path, name=NULL, no_warning=TRUE){
  if(is.null(name)){
    name <- paste0("EQRN_fit_", format(Sys.time(),'%Y%m%d_%H%M%S'))
  }
  if(substr(path, nchar(path), nchar(path)) != "/"){
    path <- paste0(path, "/")
  }
  fpath <- paste0(path, name, "/")
  check_directory(fpath, recursive=TRUE, no_warning=no_warning)
  
  torch::torch_save(fit_eqrn$fit_nn, paste0(fpath, "fit_eqrnn_network.pt"))
  fit_infos <- fit_eqrn[names(fit_eqrn)!="fit_nn"]
  if(is.null(fit_infos$torch_version)){
    fit_infos$torch_version <- utils::packageVersion("torch")
  }
  safe_save_rds(fit_infos, paste0(fpath, "fit_eqrnn_infos.rds"))
}

#' Load an EQRN object from disc
#'
#' @description Loads in memory an `"EQRN"` object that has previously been saved on disc using [EQRN_save()].
#'
#' @param path Path to the save location as a string.
#' @param name String name of the save.
#' If `NULL` (default), assumes the save name has been given implicitly in the `path`.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#' @param ... DEPRECATED. Used for back-compatibility.
#'
#' @return The loaded `"EQRN"` model.
#' @export
#' @import torch
#' @importFrom utils packageVersion
#'
#' @examples #TODO
EQRN_load <- function(path, name=NULL, device=default_device(), ...){
  if(substr(path, nchar(path), nchar(path)) != "/"){
    fpath <- paste0(path, "/")
  } else {
    fpath <- path
  }
  if(!is.null(name)){
    fpath <- paste0(fpath, name, "/")
  }
  
  network <- torch::torch_load(paste0(fpath, "fit_eqrnn_network.pt"), device=device)
  fit_infos <- readRDS(paste0(fpath, "fit_eqrnn_infos.rds"))
  
  current_version <- utils::packageVersion("torch")
  if(fit_infos$torch_version != current_version){
    warning(paste0("EQRN object was saved with torch ", fit_infos$torch_version,
                   "but loaded with torch ", current_version))
  }
  network$eval()
  
  fit_eqrn <- append(list(fit_nn = network), fit_infos)
  fit_eqrn <- legacy_names(fit_eqrn, ...)
  return(fit_eqrn)
}

#' GPD tensor loss function for training a EQRN network
#'
#' @param out Batch tensor of GPD parameters output by the network.
#' @param y Batch tensor of corresponding response variable.
#' @param orthogonal_gpd Whether the network is supposed to regress in the orthogonal reparametrization of the GPD parameters (recommended).
#' @param shape_penalty Penalty parameter for the shape estimate, to potentially regularize its variation from the fixed prior estimate.
#' @param prior_shape Prior estimate for the shape, used only if `shape_penalty>0`.
#' @param return_agg The return aggregation of the computed loss over the batch. Must be one of `"mean", "sum", "vector", "nanmean", "nansum"`.
#'
#' @return The GPD loss over the batch between the network output ans the observed responses as a `torch::Tensor`,
#' whose dimensions depend on `return_agg`.
#' @export
#' @import torch
#'
#' @examples #TODO
loss_GPD_tensor <- function(out, y, orthogonal_gpd=TRUE, shape_penalty=0, prior_shape=NULL, return_agg=c("mean", "sum", "vector", "nanmean", "nansum")){
  return_agg <- match.arg(return_agg)
  s <- torch::torch_split(out, 1, dim = 2)
  if(orthogonal_gpd){
    l <- (1 + 1/s[[2]]) * (1 + s[[2]]*(s[[2]]+1)*y/s[[1]])$log() + s[[1]]$log() - (s[[2]]+1)$log()
  }else{
    l <- (1 + 1/s[[2]]) * (1 + s[[2]]*y/s[[1]])$log() + s[[1]]$log()
  }
  
  if(shape_penalty>0){
    if(is.null(prior_shape)){stop("Prior (semi-conditional) shape value estimate needed for shape_penalty in loss_GPD_tensor.")}
    l <- l + shape_penalty * (s[[2]]-prior_shape)$square()
  }
  
  if(return_agg=="mean"){
    return(l$mean())
  } else if(return_agg=="sum"){
    return(l$sum())
  } else if(return_agg=="vector"){
    return(l)
  } else if(return_agg=="nanmean"){
    return( (l$nansum())/((!l$isnan())$sum()$item()) )
  } else if(return_agg=="nansum"){
    return(l$nansum())
  }
}

#' Computes rescaled excesses over the conditional quantiles
#'
#' @param X A covariate matrix. Can be `NULL` if there are no covariates.
#' @param y The response variable vector.
#' @param quantiles The intermediate quantiles over which to compute the excesses of `y`.
#' @param intermediate_q_feature Whether to use the intermediate `quantiles` as an additional covariate,
#' by appending it to the `X` matrix (bool).
#' @param scale_features Whether to rescale each input covariates to zero mean and unit variance before applying the network (recommended).
#' If `X_scaling` is given, `X_scaling$scaling` overrides `scale_features`.
#' @param X_scaling Existing `"X_scaling"` object containing the precomputed mean and variance for each covariate.
#' This enables reusing the scaling choice and parameters from the train set, if computing the excesses on a validation or test set,
#' in order to avoid overfitting. This is performed automatically in the `"EQRN"` objects.
#'
#' @return Named list containing:
#' \itemize{
#' \item{Y_excesses}{thematrix of response excesses,}
#' \item{X_excesses}{the (possibly rescaled and q_feat transformed) covariate matrix,}
#' \item{X_scaling}{object of class `"X_scaling"` to use for consistent scaling on future datasets,}
#' \item{excesses_ratio}{and the ratio of escesses for troubleshooting.}
#' }
#' @export
#'
#' @examples #TODO
get_excesses <- function(X=NULL, y, quantiles, intermediate_q_feature=FALSE, scale_features=FALSE, X_scaling=NULL){
  # Preprocess y
  y_rescaled <- y - quantiles
  y_excesses <- y_rescaled[y_rescaled>=0]
  Y_excesses <- matrix(y_excesses, nrow=length(y_excesses), ncol=1)
  
  if(is.null(X)){
    X_feats_excesses <- NULL
    X_scaling <- NULL
    
  } else {
    # Preprocess X and scaling
    data_scaling <- process_features(X, intermediate_q_feature=intermediate_q_feature,
                                     intermediate_quantiles=quantiles, X_scaling=X_scaling, scale_features=scale_features)
    X_feats <- data_scaling$X_scaled
    X_scaling <- data_scaling$X_scaling
    # Get excesses
    X_feats_excesses <- X_feats[y_rescaled>=0,]
  }
  
  # Diagnostic stats
  n_excesses <- length(Y_excesses)
  excesses_ratio <- n_excesses/length(y)
  
  return(list(X_excesses=X_feats_excesses, Y_excesses=Y_excesses, X_scaling=X_scaling, excesses_ratio=excesses_ratio))
}

#' Feature processor for EQRN
#'
#' @param X A covariate matrix.
#' @param intermediate_q_feature Whether to use the intermediate `quantiles` as an additional covariate,
#' by appending it to the `X` matrix (bool).
#' @param intermediate_quantiles The intermediate conditional quantiles.
#' @param X_scaling Existing `"X_scaling"` object containing the precomputed mean and variance for each covariate.
#' This enables reusing the scaling choice and parameters from the train set, if computing the excesses on a validation or test set,
#' in order to avoid overfitting. This is performed automatically in the `"EQRN"` objects.
#' @param scale_features Whether to rescale each input covariates to zero mean and unit variance before applying the network (recommended).
#' If `X_scaling` is given, `X_scaling$scaling` overrides `scale_features`.
#'
#' @return Named list containing:
#' \itemize{
#' \item{X_excesses}{the (possibly rescaled and q_feat transformed) covariate matrix,}
#' \item{X_scaling}{object of class `"X_scaling"` to use for consistent scaling on future datasets.}
#' }
#' @export
#'
#' @examples #TODO
process_features <- function(X, intermediate_q_feature, intermediate_quantiles=NULL, X_scaling=NULL, scale_features=TRUE){
  X_feats <- vec2mat(X) #verify X is matrix (otherwise transform it to 1-row matrix)
  if(intermediate_q_feature){
    if(is.null(intermediate_quantiles)){stop("intermediate_quantiles needed for intermediate_q_feature.")}
    X_feats <- cbind(X_feats, intermediate_quantiles)
  }
  X_feats_structure <- perform_scaling(X_feats, X_scaling=X_scaling, scale_features=scale_features)
  return(X_feats_structure)
}

#' Performs feature scaling without overfitting
#'
#' @param X A covariate matrix.
#' @param X_scaling Existing `"X_scaling"` object containing the precomputed mean and variance for each covariate.
#' This enables reusing the scaling choice and parameters from the train set, if computing the excesses on a validation or test set,
#' in order to avoid overfitting. This is performed automatically in the `"EQRN"` objects.
#' @param scale_features Whether to rescale each input covariates to zero mean and unit variance before applying the model (recommended).
#' If `X_scaling` is given, `X_scaling$scaling` overrides `scale_features`.
#' @param stat_attr DEPRECATED. Whether to keep attributes in the returned covariate matrix itself.
#'
#' @return Named list containing:
#' \itemize{
#' \item{X_excesses}{the (possibly rescaled and q_feat transformed) covariate matrix,}
#' \item{X_scaling}{object of class `"X_scaling"` to use for consistent scaling on future datasets.}
#' }
#' @export
#'
#' @examples #TODO
perform_scaling <- function(X, X_scaling=NULL, scale_features=TRUE, stat_attr=FALSE){
  
  if(is.null(X_scaling)){
    
    if(scale_features){
      X_scaled <- scale(X)
      X_scaling <- list(scaling=TRUE, center=attr(X_scaled, "scaled:center"),
                        scale=attr(X_scaled, "scaled:scale"))
      if(!stat_attr){
        attr(X_scaled, "scaled:center") <- NULL
        attr(X_scaled, "scaled:scale") <- NULL
      }
    } else {
      X_scaled <- X
      X_scaling <- list(scaling=FALSE, center=FALSE, scale=FALSE)
    }
    class(X_scaling) <- c("X_scaling")
    
  } else {
    if(X_scaling$scaling){
      X_scaled <- scale(X, center=X_scaling$center, scale=X_scaling$scale)
      if(!stat_attr){
        attr(X_scaled, "scaled:center") <- NULL
        attr(X_scaled, "scaled:scale") <- NULL
      }
    } else {
      X_scaled <- X
    }
  }
  return(list(X_scaled=X_scaled, X_scaling=X_scaling))
}


#' Default torch device
#'
#' @return Returns `torch::torch_device("cuda")` if `torch::cuda_is_available()`, or `torch::torch_device("cpu")` otherwise.
#' @export
#' @import torch
#'
#' @examples device <- default_device()
default_device <- function(){
  if(torch::cuda_is_available()) {
    device <- torch::torch_device("cuda")
  } else {
    device <- torch::torch_device("cpu")
  }
  return(device)
}

#' Default batch size (internal)
#'
#' @param tensor_dat A [torch::dataset()].
#' @param batch_size An initial batch size, by default `256`.
#'
#' @return The fixed batch_size.
#'
#' @examples #TODO
#' @keywords internal
batch_size_default <- function(tensor_dat, batch_size=256){
  n <- length(tensor_dat)
  if(n==1){
    return(1)
  }
  batch_size <- min(batch_size, n)
  while(n%%batch_size == 1){
    # as torch::dataloader simplified dimensions due to a bug.
    # TODO: remove when fixed in torch!
    batch_size <- batch_size+1
  }
  return(batch_size)
}


#' Internal renaming function for back-compatibility
#'
#' @param eqrn_fit EQRN fitted object.
#' @param classes If provided, overrides classes of `eqrn_fit`.
#'
#' @return The `eqrn_fit` object with updated attribute names and classes.
#'
#' @examples #TODO
#' @keywords internal
legacy_names <- function(eqrn_fit, classes=NULL){
  if(is.null(eqrn_fit$interm_lvl)){
    eqrn_fit$interm_lvl <- eqrn_fit$threshold
  }
  if(!is.null(classes)){
    class(eqrn_fit) <- classes
  }
  return(eqrn_fit)
}

#' (INTERNAL) Corrects a dimension simplification bug from the torch package
#'
#' @description (INTERNAL) Issue was raised to the `torch` maintainers and should be fixed, deprecating this function.
#'
#' @param dl_i batch object from an itteration over a [torch::dataloader()].
#' @param ... dimension(s) of the covariate object (excluding the first "batch" dimension)
#' @param responses Bolean indicating whether the batch object `dl_i` is a covariates-response pair.
#'
#' @return The fixed dl_i object
#'
#' @examples #TODO
#' @keywords internal
fix_dimsimplif <- function(dl_i, ..., responses=TRUE){
  dl_i[[1]] <- dl_i[[1]]$reshape(c(-1, ...))
  if(responses){
    dl_i[[2]] <- dl_i[[2]]$reshape(c(-1, 1))
  }
  return(dl_i)
}

