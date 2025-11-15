
#' @title EQRN fit function for sequential and time series data
#'
#' @description Use the [EQRN_fit_restart()] wrapper instead, with `data_type="seq"`, for better stability using fitting restart.
#'
#' @param X Matrix of covariates, for training. Entries must be in sequential order.
#' @param y Response variable vector to model the extreme conditional quantile of, for training. Entries must be in sequential order.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `interm_lvl`.
#' @param interm_lvl Probability level for the intermediate quantiles `intermediate_quantiles`.
#' @param shape_fixed Whether the shape estimate depends on the covariates or not (bool).
#' @param hidden_size Dimension of the hidden latent state variables in the recurrent network.
#' @param num_layers Number of recurrent layers.
#' @param rnn_type Type of recurrent architecture, can be one of `"lstm"` (default) or `"gru"`.
#' @param p_drop Probability parameter for dropout before each hidden layer for regularization during training.
#' @param intermediate_q_feature Whether to use the `intermediate_quantiles` as an additional covariate, by appending it to the `X` matrix (bool).
#' @param learning_rate Initial learning rate for the optimizer during training of the neural network.
#' @param L2_pen L2 weight penalty parameter for regularization during training.
#' @param seq_len Data sequence length (i.e. number of past observations) used during training to predict each response quantile.
#' @param shape_penalty Penalty parameter for the shape estimate, to potentially regularize its variation from the fixed prior estimate.
#' @param scale_features Whether to rescale each input covariates to zero mean and unit covariance before applying the network (recommended).
#' @param n_epochs Number of training epochs.
#' @param batch_size Batch size used during training.
#' @param X_valid Covariates in a validation set, or `NULL`. Entries must be in sequential order.
#' Used for monitoring validation loss during training, enabling learning-rate decay and early stopping.
#' @param y_valid Response variable in a validation set, or `NULL`. Entries must be in sequential order.
#' Used for monitoring validation loss during training, enabling learning-rate decay and early stopping.
#' @param quant_valid Intermediate conditional quantiles at level `interm_lvl` in a validation set, or `NULL`.
#' Used for monitoring validation loss during training, enabling learning-rate decay and early stopping.
#' @param lr_decay Learning rate decay factor.
#' @param patience_decay Number of epochs of non-improving validation loss before a learning-rate decay is performed.
#' @param min_lr Minimum learning rate, under which no more decay is performed.
#' @param patience_stop Number of epochs of non-improving validation loss before early stopping is performed.
#' @param tol Tolerance for stopping training, in case of no significant training loss improvements.
#' @param orthogonal_gpd Whether to use the orthogonal reparametrization of the estimated GPD parameters (recommended).
#' @param patience_lag The validation loss is considered to be non-improving
#' if it is larger than on any of the previous `patience_lag` epochs.
#' @param fold_separation Index of fold separation or sequential discontinuity in the data.
#' @param optim_met DEPRECATED. Optimization algorithm to use during training. `"adam"` is the default.
#' @param seed Integer random seed for reproducibility in network weight initialization.
#' @param verbose Amount of information printed during training (0:nothing, 1:most important, 2:everything).
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return An EQRN object of classes `c("EQRN_seq", "EQRN")`, containing the fitted network,
#' as well as all the relevant information for its usage in other functions.
#' @export
#' @importFrom coro loop
EQRN_fit_seq <- function(X, y, intermediate_quantiles, interm_lvl, shape_fixed=FALSE, hidden_size=10, num_layers=1, rnn_type=c("lstm","gru"), p_drop=0,
                         intermediate_q_feature=TRUE, learning_rate=1e-4, L2_pen=0, seq_len=10, shape_penalty=0,
                         scale_features=TRUE, n_epochs=500, batch_size=256, X_valid=NULL, y_valid=NULL, quant_valid=NULL,
                         lr_decay=1, patience_decay=n_epochs, min_lr=0, patience_stop=n_epochs,
                         tol=1e-5, orthogonal_gpd=TRUE, patience_lag=1, fold_separation=NULL, optim_met="adam",
                         seed=NULL, verbose=2, device=default_device()){
  
  ensure_backend_installed()
  if(!is.null(seed)){torch::torch_manual_seed(seed)}
  
  rnn_type <- match.arg(rnn_type)
  iq_feat <- c(intermediate_quantiles[2:length(intermediate_quantiles)],last_elem(intermediate_quantiles))
  data_scaling <- process_features(X=X, intermediate_q_feature=intermediate_q_feature, intermediate_quantiles=iq_feat,
                                   X_scaling=NULL, scale_features=scale_features)
  Xs <- data_scaling$X_scaled
  X_scaling <- data_scaling$X_scaling
  
  # Data Loader
  trainset <- mts_dataset(y, Xs, seq_len, scale_Y=scale_features,
                          intermediate_quantiles=intermediate_quantiles, fold_separation=fold_separation, device=device)
  Y_scaling <- trainset$get_Y_scaling()
  n_train <- length(trainset)
  trainloader <- torch::dataloader(trainset, batch_size=batch_size, shuffle=TRUE)
  
  # Validation dataset (if everything needed is given)
  do_validation <- (!is.null(y_valid) & !is.null(X_valid) & !is.null(quant_valid))
  if(do_validation){
    iq_feat_valid <- c(quant_valid[2:length(quant_valid)],last_elem(quant_valid))
    data_scaling_ex <- process_features(X=X_valid, intermediate_q_feature=intermediate_q_feature, intermediate_quantiles=iq_feat_valid,
                                        X_scaling=X_scaling, scale_features=scale_features)
    validset <- mts_dataset(y_valid, data_scaling_ex$X_scaled, seq_len, scale_Y=Y_scaling, intermediate_quantiles=quant_valid, device=device)
    n_valid <- length(validset)
    validloader <- torch::dataloader(validset, batch_size=batch_size, shuffle=FALSE)
  }
  
  # Semi-conditional GPD fit (on y rescaled excesses wrt intermediate_quantiles)
  if(shape_penalty>0){
    semicond_gpd_fit <- fit_GPD_unconditional(c(as.matrix(trainset$get_Y_responses())), interm_lvl=NULL, thresh_quantiles=NULL)
  }else{
    semicond_gpd_fit <- NULL
  }
  
  # Instantiate network
  Dim_in <- ncol(Xs)+1
  network <- Recurrent_GPD_net(type=rnn_type, nb_input_features=Dim_in, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=p_drop, shape_fixed=shape_fixed, device=device)$to(device=device)
  
  # Optimizer
  optimizer <- setup_optimizer_seq(network, learning_rate, L2_pen, optim_met=optim_met)
  
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
      # b <- fix_dimsimplif(b, trainset[1][[1]]$shape)#TODO: remove when fixed in torch
      # Forward pass
      net_out <- network(b[[1]])
      # Loss
      loss <- loss_GPD_tensor(net_out, b[[2]], orthogonal_gpd=orthogonal_gpd,
                              shape_penalty=shape_penalty, prior_shape=semicond_gpd_fit$shape, return_agg="mean")
      # Check for bad initialization
      while(e<2 & is.nan(loss$item())){
        network <- Recurrent_GPD_net(type=rnn_type, nb_input_features=Dim_in, hidden_size=hidden_size,
                                     num_layers=num_layers, dropout=p_drop, shape_fixed=shape_fixed, device=device)$to(device=device)
        network$train()
        optimizer <- setup_optimizer_seq(network, learning_rate, L2_pen, optim_met=optim_met)
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
        # b <- fix_dimsimplif(b, validset[1][[1]]$shape)#TODO: remove when fixed in torch
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
  
  fit_eqrn_ts <- list(fit_nn = network, interm_lvl = interm_lvl, intermediate_q_feature = intermediate_q_feature, seq_len=seq_len,
                      train_loss = loss_log_train[1:e], X_scaling=X_scaling, Y_scaling=Y_scaling, orthogonal_gpd=orthogonal_gpd,
                      n_obs = length(y), n_excesses = n_train, excesses_ratio = n_train/(length(y)-seq_len))
  if(do_validation){
    fit_eqrn_ts$valid_loss <- loss_log_valid[1:e]
  }
  class(fit_eqrn_ts) <- c("EQRN_seq", "EQRN")
  
  return(fit_eqrn_ts)
}


#' Predict function for an EQRN_seq fitted object
#'
#' @param fit_eqrn Fitted `"EQRN_seq"` object.
#' @param X Matrix of covariates to predict the corresponding response's conditional quantiles.
#' @param Y Response variable vector corresponding to the rows of `X`.
#' @param prob_lvls_predict Vector of probability levels at which to predict the conditional quantiles.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param crop_predictions Whether to crop out the fist `seq_len` observations (which are `NA`) from the returned matrix.
#' @param seq_len Data sequence length (i.e. number of past observations) used to predict each response quantile.
#' By default, the training `fit_eqrn$seq_len` is used.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Matrix of size `nrow(X)` times `prob_lvls_predict`
#' (or `nrow(X)-seq_len` times `prob_lvls_predict` if `crop_predictions`)
#' containing the conditional quantile estimates of the corresponding response observations at each probability level.
#' Simplifies to a vector if `length(prob_lvls_predict)==1`.
#' @export
EQRN_predict_seq <- function(fit_eqrn, X, Y, prob_lvls_predict, intermediate_quantiles, interm_lvl,
                             crop_predictions=FALSE, seq_len=fit_eqrn$seq_len, device=default_device()){
  
  if(length(dim(prob_lvls_predict))>1){
    stop("Please provide a single value or 1D vector as prob_lvls_predict in EQRN_predict_seq.")
  }
  
  if(length(prob_lvls_predict)==1){
    return(EQRN_predict_internal_seq(fit_eqrn, X, Y, prob_lvls_predict, intermediate_quantiles,
                                     interm_lvl, crop_predictions=crop_predictions, seq_len=seq_len, device=device))
  } else if(length(prob_lvls_predict)>1){
    nb_prob_lvls_predict <- length(prob_lvls_predict)
    length_preds <- if(crop_predictions){nrow(X)-seq_len}else{nrow(X)}
    predicted_quantiles <- matrix(as.double(NA), nrow=length_preds, ncol=nb_prob_lvls_predict)
    for(i in 1:nb_prob_lvls_predict){
      predicted_quantiles[,i] <- EQRN_predict_internal_seq(fit_eqrn, X, Y, prob_lvls_predict[i], intermediate_quantiles,
                                                           interm_lvl, crop_predictions=crop_predictions, seq_len=seq_len, device=device)
    }
    return(predicted_quantiles)
  } else {
    stop("Please provide a single value or 1D vector as prob_lvls_predict in EQRN_predict_seq.")
  }
}


#' Predict method for an EQRN_seq fitted object
#'
#' @param object Fitted `"EQRN_seq"` object.
#' @inheritDotParams EQRN_predict_seq -fit_eqrn
#' 
#' @details See [EQRN_predict_seq()] for more details.
#'
#' @inherit EQRN_predict_seq return
#' @method predict EQRN_seq
#' @export
predict.EQRN_seq <- function(object, ...){
  return(EQRN_predict_seq(fit_eqrn=object, ...))
}


#' Internal predict function for an EQRN_seq fitted object
#'
#' @param fit_eqrn Fitted `"EQRN_seq"` object.
#' @param X Matrix of covariates to predict the corresponding response's conditional quantiles.
#' @param Y Response variable vector corresponding to the rows of `X`.
#' @param prob_lvl_predict Probability level at which to predict the conditional quantile.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param crop_predictions Whether to crop out the fist `seq_len` observations (which are `NA`) from the returned vector
#' @param seq_len Data sequence length (i.e. number of past observations) used to predict each response quantile.
#' By default, the training `fit_eqrn$seq_len` is used.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Vector of length `nrow(X)` (or `nrow(X)-seq_len` if `crop_predictions`)
#' containing the conditional quantile estimates of the response associated to each covariate observation at each probability level.
#' 
#' @keywords internal
EQRN_predict_internal_seq <- function(fit_eqrn, X, Y, prob_lvl_predict, intermediate_quantiles, interm_lvl,
                                      crop_predictions=FALSE, seq_len=fit_eqrn$seq_len, device=default_device()){
  
  GPD_params_pred <- EQRN_predict_params_seq(fit_eqrn, X, Y, intermediate_quantiles,
                                             return_parametrization="classical", interm_lvl=interm_lvl, seq_len=seq_len, device=device)
  sigmas <- GPD_params_pred$scales
  xis <- GPD_params_pred$shapes
  
  predicted_quantiles <- matrix(GPD_quantiles(prob_lvl_predict, interm_lvl,
                                              intermediate_quantiles[(seq_len+1):length(intermediate_quantiles)],
                                              sigmas, xis), ncol=1)
  
  if(!crop_predictions){
    predicted_quantiles <- rbind(matrix(as.double(NA), nrow=seq_len),
                                 predicted_quantiles)
  }
  return(predicted_quantiles)
}


#' GPD parameters prediction function for an EQRN_seq fitted object
#'
#' @param fit_eqrn Fitted `"EQRN_seq"` object.
#' @param X Matrix of covariates to predict conditional GPD parameters.
#' @param Y Response variable vector corresponding to the rows of `X`.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param return_parametrization Which parametrization to return the parameters in, either `"classical"` or `"orthogonal"`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param seq_len Data sequence length (i.e. number of past observations) used to predict each response quantile.
#' By default, the training `fit_eqrn$seq_len` is used.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Named list containing: `"scales"` and `"shapes"` as numerical vectors of length `nrow(X)`,
#' and the `seq_len` used.
#' @export
#' @importFrom coro loop
EQRN_predict_params_seq <- function(fit_eqrn, X, Y, intermediate_quantiles=NULL, return_parametrization=c("classical","orthogonal"),
                                    interm_lvl=fit_eqrn$interm_lvl, seq_len=fit_eqrn$seq_len, device=default_device()){
  ## return_parametrization controls the desired parametrization of the output parameters
  ## (works for both "orthogonal_gpd" EQRN parametrizations, by converting if needed)
  
  return_parametrization <- match.arg(return_parametrization)
  
  if(interm_lvl!=fit_eqrn$interm_lvl){stop("EQRN intermediate quantiles interm_lvl does not match in train and predict.")}
  
  iq_feat <- c(intermediate_quantiles[2:length(intermediate_quantiles)],last_elem(intermediate_quantiles))
  X_feats <- process_features(X, intermediate_q_feature=fit_eqrn$intermediate_q_feature,
                              intermediate_quantiles=iq_feat, X_scaling=fit_eqrn$X_scaling)$X_scaled
  testset <- mts_dataset(Y, X_feats, seq_len, scale_Y=fit_eqrn$Y_scaling,
                         intermediate_quantiles=NULL, device=device)
  testloader <- torch::dataloader(testset, batch_size=batch_size_default(testset), shuffle=FALSE)
  
  network <- fit_eqrn$fit_nn
  network$eval()
  
  scales <- c()
  shapes <- c()
  coro::loop(for (b in testloader) {
    # b <- fix_dimsimplif(b, testset[1][[1]]$shape)#TODO: remove when fixed in torch
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
  
  return(list(scales=scales, shapes=shapes, seq_len=seq_len))
}


#' Tail excess probability prediction using an EQRN_seq object
#'
#' @param val Quantile value(s) used to estimate the conditional excess probability or cdf.
#' @param fit_eqrn Fitted `"EQRN_seq"` object.
#' @param X Matrix of covariates to predict the response's conditional excess probabilities.
#' @param Y Response variable vector corresponding to the rows of `X`.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param crop_predictions Whether to crop out the fist `seq_len` observations (which are `NA`) from the returned vector
#' @param body_proba Value to use when the predicted conditional probability is below `interm_lvl`
#' (in which case it cannot be precisely assessed by the model).
#' If `"default"` is given (the default), `paste0(">",1-interm_lvl)` is used if `proba_type=="excess"`,
#' and `paste0("<",interm_lvl)` is used if `proba_type=="cdf"`.
#' @param proba_type Whether to return the `"excess"` probability over `val` (default) or the `"cdf"` at `val`.
#' @param seq_len Data sequence length (i.e. number of past observations) used to predict each response quantile.
#' By default, the training `fit_eqrn$seq_len` is used.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Vector of probabilities (and possibly a few `body_proba` values if `val` is not large enough) of length `nrow(X)`
#' (or `nrow(X)-seq_len` if `crop_predictions`).
#' @export
EQRN_excess_probability_seq <- function(val, fit_eqrn, X, Y, intermediate_quantiles, interm_lvl=fit_eqrn$interm_lvl,
                                        crop_predictions=FALSE, body_proba="default", proba_type=c("excess","cdf"),
                                        seq_len=fit_eqrn$seq_len, device=default_device()){
  
  proba_type <- match.arg(proba_type)
  
  GPD_params_pred <- EQRN_predict_params_seq(fit_eqrn, X, Y, intermediate_quantiles,
                                             return_parametrization="classical", interm_lvl=interm_lvl, seq_len=seq_len, device=device)
  sigmas <- GPD_params_pred$scales
  xis <- GPD_params_pred$shapes
  
  Probs <- GPD_excess_probability(val, sigma=sigmas, xi=xis,
                                  interm_threshold=intermediate_quantiles[(seq_len+1):length(intermediate_quantiles)],
                                  threshold_p=interm_lvl, body_proba=body_proba, proba_type=proba_type)
  
  if(!crop_predictions){
    Probs <- c(rep(as.double(NA), seq_len), c(Probs))
  }
  return(Probs)
}


#' Tail excess probability prediction method using an EQRN_iid object
#'
#' @param object Fitted `"EQRN_seq"` object.
#' @inheritDotParams EQRN_excess_probability_seq -fit_eqrn
#' 
#' @details See [EQRN_excess_probability_seq()] for more details.
#'
#' @inherit EQRN_excess_probability_seq return
#' @method excess_probability EQRN_seq
#' @export
excess_probability.EQRN_seq <- function(object, ...){
  return(EQRN_excess_probability_seq(fit_eqrn=object, ...))
}


#' Generalized Pareto likelihood loss of a EQRN_seq predictor
#'
#' @param fit_eqrn Fitted `"EQRN_seq"` object.
#' @param X Matrix of covariates.
#' @param Y Response variable vector corresponding to the rows of `X`.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `fit_eqrn$interm_lvl`.
#' @param interm_lvl Optional, checks that `interm_lvl == fit_eqrn$interm_lvl`.
#' @param seq_len Data sequence length (i.e. number of past observations) used to predict each response quantile.
#' By default, the training `fit_eqrn$seq_len` is used.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return Negative GPD log likelihood of the conditional EQRN predicted parameters
#' over the response exceedances over the intermediate quantiles.
#' @export
compute_EQRN_seq_GPDLoss <- function(fit_eqrn, X, Y, intermediate_quantiles=NULL, interm_lvl=fit_eqrn$interm_lvl,
                                     seq_len=fit_eqrn$seq_len, device=default_device()){
  params <- EQRN_predict_params_seq(fit_eqrn, X, Y, intermediate_quantiles=intermediate_quantiles, return_parametrization="classical",
                                    interm_lvl=interm_lvl, seq_len=seq_len, device=device)
  loss <- (1 + 1/params$shapes) * log(1 + params$shapes*c(Y)[(seq_len+1):length(Y)]/params$scales) + log(params$scales)
  loss <- mean(loss)
  return(loss)
}


#' Instantiate an optimizer for training an EQRN_seq network
#'
#' @param network A `torch::nn_module` network to be trained in [EQRN_fit_seq()].
#' @param learning_rate Initial learning rate for the optimizer during training of the neural network.
#' @param L2_pen L2 weight penalty parameter for regularization during training.
#' @param optim_met DEPRECATED. Optimization algorithm to use during training. `"adam"` is the default.
#'
#' @return A `torch::optimizer` object used in [EQRN_fit_seq()] for training.
#' 
#' @keywords internal
setup_optimizer_seq <- function(network, learning_rate, L2_pen, optim_met="adam"){
  if(optim_met!="adam"){stop("Other optim methods are deprecated.")}
  
  optimizer <- torch::optim_adam(network$parameters, lr=learning_rate, weight_decay=L2_pen)
  return(optimizer)
}


#' Dataset creator for sequential data
#'
#' @description A `torch::dataset` object that can be initialized with sequential data,
#' used to feed a recurrent network during training or prediction.
#' It is used in [EQRN_fit_seq()] and corresponding predict functions,
#' as well as in other recurrent methods such as [QRN_seq_fit()] and its predict functions.
#' It can perform scaling of the response's past as a covariate, and compute excesses as a response when used in [EQRN_fit_seq()].
#' It also allows for fold separation or sequential discontinuity in the data.
#'
#' @param X Matrix of covariates, for training. Entries must be in sequential order.
#' @param Y Response variable vector to model the extreme conditional quantile of, for training. Entries must be in sequential order.
#' @param seq_len Data sequence length (i.e. number of past observations) used during training to predict each response quantile.
#' @param intermediate_quantiles Vector of intermediate conditional quantiles at level `interm_lvl`.
#' @param scale_Y Whether to rescale the response past, when considered as an input covariate,
#' to zero mean and unit covariance before applying the network (recommended).
#' @param fold_separation Fold separation index, when using concatenated folds as data.
#' @param sample_frac Value between `0` and `1`. If `sample_frac < 1`, a subsample of the data is used. Defaults to `1`.
#' @param device (optional) A [torch::torch_device()]. Defaults to [default_device()].
#'
#' @return The [`torch::dataset`] containing the given data, to be used with a recurrent neural network.
#' @export
#' @importFrom stats sd
mts_dataset <- torch::dataset(
  name = "mts_dataset",
  
  initialize = function(Y, X, seq_len, intermediate_quantiles=NULL, scale_Y=TRUE, 
                        fold_separation=NULL, sample_frac=1, device=EQRN::default_device()) {
    
    self$seq_len <- seq_len
    if(is.logical(scale_Y)){
      scale_Y <- if(scale_Y) list(center=mean(Y), scale=stats::sd(Y)) else list(center=0, scale=1)
    }
    self$Y_scaling <- scale_Y
    Ys <- matrix(((Y-self$Y_scaling$center)/self$Y_scaling$scale), ncol=1)
    self$input_seq <- torch::torch_tensor(cbind(Ys,X), device = device)
    self$response <- torch::torch_tensor(matrix(Y, ncol=1), device = device)
    
    if(!is.null(intermediate_quantiles)){
      self$starts <- which(c(Y)>=c(intermediate_quantiles)) - self$seq_len
      if(any(is.na(intermediate_quantiles[1]))){
        self$starts <- self$starts[self$starts>self$seq_len]
      }else{
        self$starts <- self$starts[self$starts>0]
      }
      self$response <- self$response - torch::torch_tensor(intermediate_quantiles)$reshape(list(-1,1))
    }else{
      n <- length(Y) - self$seq_len
      self$starts <- sort(sample.int(n = n, size = ceiling(n * sample_frac)))
    }
    if(!is.null(fold_separation)){
      self$starts <- setdiff(self$starts, (fold_separation-self$seq_len):(fold_separation-1))
    }
  },
  
  get_Y_responses = function() {
    return(self$response)
  },
  
  get_Y_scaling = function() {
    return(self$Y_scaling)
  },
  
  .getitem = function(i) {
    
    start <- self$starts[i]
    end <- start + self$seq_len - 1
    
    list(x = self$input_seq[start:end,], y = self$response[end + 1,])
  },
  
  .length = function() {
    length(self$starts)
  }
)


