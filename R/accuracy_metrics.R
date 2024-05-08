
# ==== Accuracy metrics ====

#' Mean squared error
#'
#' @param y Vector of observations or ground-truths.
#' @param y_hat Vector of predictions.
#' @param return_agg Whether to return the `"mean"` (default), `"sum"`, or `"vector"` of errors.
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#'
#' @return The mean (or total or vectorial) squared error between `y` and `y_hat`.
#' @export
#'
#' @examples mean_squared_error(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
mean_squared_error <- function(y, y_hat, return_agg=c("mean", "sum", "vector"), na.rm=FALSE){
  return_agg <- match.arg(return_agg)
  l <- (y - y_hat) ^ 2
  if(return_agg=="mean") return(mean(l, na.rm=na.rm))
  else if(return_agg=="sum") return(sum(l, na.rm=na.rm))
  else return(l)
}

#' Mean absolute error
#'
#' @inheritParams mean_squared_error
#'
#' @return The mean (or total or vectorial) absolute error between `y` and `y_hat`.
#' @export
#'
#' @examples mean_absolute_error(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
mean_absolute_error <- function(y, y_hat, return_agg=c("mean", "sum", "vector"), na.rm=FALSE){
  return_agg <- match.arg(return_agg)
  l <- abs(y - y_hat)
  if(return_agg=="mean") return(mean(l, na.rm=na.rm))
  else if(return_agg=="sum") return(sum(l, na.rm=na.rm))
  else return(l)
}

#' Square loss
#'
#' @param y Vector of observations or ground-truths.
#' @param y_hat Vector of predictions.
#'
#' @return The vector of square errors between `y` and `y_hat`.
#' @export
#'
#' @examples square_loss(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
square_loss <- function(y, y_hat){
  mean_squared_error(y, y_hat, return_agg="vector")
}

#' Quantile loss
#'
#' @param y Vector of observations.
#' @param y_hat Vector of predicted quantiles at probability level `q`.
#' @param q Probability level of the predicted quantile.
#' @param return_agg Whether to return the `"mean"` (default), `"sum"`, or `"vector"` of losses.
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#'
#' @return The mean (or total or vectorial) quantile loss between `y` and `y_hat` at level `q`.
#' @export
#'
#' @examples quantile_loss(c(2.3, 4.2, 1.8), c(2.9, 5.6, 2.7), q=0.8)
quantile_loss <- function(y, y_hat, q, return_agg=c("mean", "sum", "vector"), na.rm=FALSE){
  return_agg <- match.arg(return_agg)
  u <- y - y_hat
  l <-(q-(u<=0))*u
  if(return_agg=="mean") return(mean(l, na.rm=na.rm))
  else if(return_agg=="sum") return(sum(l, na.rm=na.rm))
  else return(l)
}

#' Prediction bias
#'
#' @param y Vector of observations or ground-truths.
#' @param y_hat Vector of predictions.
#' @param square_bias Whether to return the square bias (bool); defaults to `FALSE`.
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#'
#' @return The (square) bias of the predictions `y_hat` for `y`.
#' @export
#'
#' @examples prediction_bias(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
prediction_bias <- function(y, y_hat, square_bias=FALSE, na.rm=FALSE){
  b <- mean((y_hat - y), na.rm=na.rm)
  if(square_bias) return(b^2)
  else return(b)
}

#' Prediction residual variance
#'
#' @param y Vector of observations or ground-truths.
#' @param y_hat Vector of predictions.
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#'
#' @return The residual variance of the predictions `y_hat` for `y`.
#' @export
#' @importFrom stats var
#'
#' @examples prediction_residual_variance(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
prediction_residual_variance <- function(y, y_hat, na.rm=FALSE){
  rv <- stats::var((y_hat - y), na.rm=na.rm)
  return(rv)
}

#' R squared
#'
#' @description The coefficient of determination, often called R squared, is the proportion of data variance explained by the predictions.
#'
#' @param y Vector of observations or ground-truths.
#' @param y_hat Vector of predictions.
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#'
#' @return The R squared of the predictions `y_hat` for `y`.
#' @export
#'
#' @examples R_squared(c(2.3, 4.2, 1.8), c(2.2, 4.6, 1.7))
R_squared <- function(y, y_hat, na.rm=FALSE){
  RSS <- mean_squared_error(y, y_hat, return_agg="sum", na.rm=na.rm)
  TSS <- sum((y - mean(y, na.rm=na.rm))^2, na.rm=na.rm)
  R2 <- 1 - (RSS / TSS)
  return(R2)
}

#' Proportion of observations below conditional quantile vector
#'
#' @param y Vector of observations.
#' @param Q_hat Vector of predicted quantiles.
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#'
#' @return The proportion of observation below the predictions.
#' @export
#'
#' @examples proportion_below(c(2.3, 4.2, 1.8), c(2.9, 5.6, 1.7))
proportion_below <- function(y, Q_hat, na.rm=FALSE){
  n <- length(y)
  if(n!=length(Q_hat)){stop("y and Q_hat should be of same length in 'proportion_below'.")}
  prop_below <- sum((y<=Q_hat), na.rm=na.rm)/n
  return(prop_below)
}

#' Quantile prediction calibration error
#'
#' @param y Vector of observations.
#' @param Q_hat Vector of predicted quantiles at probability level `prob_level`.
#' @param prob_level Probability level of the predicted quantile.
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#'
#' @return The quantile prediction error calibration metric.
#' @export
#'
#' @examples quantile_prediction_error(c(2.3, 4.2, 1.8), c(2.9, 5.6, 2.7), prob_level=0.8)
quantile_prediction_error <- function(y, Q_hat, prob_level, na.rm=FALSE){
  n <- length(y)
  if(n!=length(Q_hat)){stop("y and Q_hat should be of same length in 'quantile_prediction_error'.")}
  l <- (sum((y<Q_hat), na.rm=na.rm)-n*prob_level)/sqrt(n*prob_level*(1-prob_level))
  return(l)
}

#' Quantile exceedance probability prediction calibration error
#'
#' @param Probs Predicted probabilities to exceed or be smaller than a fixed quantile.
#' @param prob_level Probability level of the quantile.
#' @param return_years The probability level can be given in term or return years instead.
#' Only used if `prob_level` is not given.
#' @param type_probs Whether the predictions are the `"cdf"` (default) or `"exceedance"` probabilities.
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#'
#' @return The calibration metric for the predicted probabilities.
#' @export
#'
#' @examples quantile_exceedance_proba_error(c(0.1, 0.3, 0.2), prob_level=0.8)
quantile_exceedance_proba_error <- function(Probs, prob_level=NULL, return_years=NULL,
                                            type_probs=c("cdf","exceedance"), na.rm=FALSE){
  type_probs <- match.arg(type_probs)
  n <- length(Probs)
  if(is.null(prob_level)){
    if(is.null(return_years)){
      stop("One of prob_level or return_years should be provided in 'quantile_exceedance_proba_error'.")
    }else{
      prob_level <- 1 - 1/(365.25 * return_years)
    }
  }
  if(type_probs=="exceedance"){
    Fyx <- 1 - as.double(Probs)
  } else {
    Fyx <- as.double(Probs)
  }
  l <- mean(Fyx, na.rm=na.rm) - prob_level
  return(l)
}


# ==== Multilevel accuracy metric helpers ====

#' Multilevel quantile MSEs
#'
#' @description Multilevel version of [mean_squared_error()].
#'
#' @param True_Q Matrix of size `n_obs` times `proba_levels`,
#' whose columns are the vectors of ground-truths at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param Pred_Q Matrix of the same size as `True_Q`,
#' whose columns are the predictions at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param proba_levels Vector of probability levels at which the predictions were made.
#' Must be of length `ncol(Pred_Q)`.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output MSEs (bool).
#' @param sd Whether to return the squared error standard deviation (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the mean square errors
#' between each respective columns of `True_Q` and `Pred_Q`.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "MSE_q", proba_levels)`.
#' If `sd==TRUE` a named list is instead returned, containing the `"MSEs"` described above and
#' `"SDs"`, their standard deviations.
#' @export
multilevel_MSE <- function(True_Q, Pred_Q, proba_levels, prefix="", na.rm=FALSE, give_names=TRUE, sd=FALSE){
  nb_prob_lvls_predict <- length(proba_levels)
  MSEs <- rep(as.double(NA), nb_prob_lvls_predict)
  if(sd){SDs <- rep(as.double(NA), nb_prob_lvls_predict)}
  for(i in 1:nb_prob_lvls_predict){
    MSEs[i] <- mean_squared_error(True_Q[,i], Pred_Q[,i], return_agg="mean", na.rm=na.rm)
    if(sd){SDs[i] <- sd(mean_squared_error(True_Q[,i], Pred_Q[,i], return_agg="vector", na.rm=na.rm), na.rm=na.rm)}
  }
  if(give_names){
    names(MSEs) <- paste0(prefix, "MSE_q", proba_levels)
    if(sd){names(SDs) <- paste0(prefix, "MSE_sd_q", proba_levels)}
  }
  if(sd){return(list(MSEs=MSEs, SDs=SDs))}
  return(MSEs)
}

#' Multilevel quantile MAEs
#'
#' @description Multilevel version of [mean_absolute_error()].
#'
#' @param True_Q Matrix of size `n_obs` times `proba_levels`,
#' whose columns are the vectors of ground-truths at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param Pred_Q Matrix of the same size as `True_Q`,
#' whose columns are the predictions at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param proba_levels Vector of probability levels at which the predictions were made.
#' Must be of length `ncol(Pred_Q)`.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output MAEs (bool).
#' @param sd Whether to return the absolute error standard deviation (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the mean absolute errors
#' between each respective columns of `True_Q` and `Pred_Q`.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "MAE_q", proba_levels)`.
#' If `sd==TRUE` a named list is instead returned, containing the `"MAEs"` described above and
#' `"SDs"`, their standard deviations.
#' @export
multilevel_MAE <- function(True_Q, Pred_Q, proba_levels, prefix="", na.rm=FALSE, give_names=TRUE, sd=FALSE){
  nb_prob_lvls_predict <- length(proba_levels)
  MAEs <- rep(as.double(NA), nb_prob_lvls_predict)
  if(sd){SDs <- rep(as.double(NA), nb_prob_lvls_predict)}
  for(i in 1:nb_prob_lvls_predict){
    MAEs[i] <- mean_absolute_error(True_Q[,i], Pred_Q[,i], return_agg="mean", na.rm=na.rm)
    if(sd){SDs[i] <- sd(mean_absolute_error(True_Q[,i], Pred_Q[,i], return_agg="vector", na.rm=na.rm), na.rm=na.rm)}
  }
  if(give_names){
    names(MAEs) <- paste0(prefix, "MAE_q", proba_levels)
    if(sd){names(SDs) <- paste0(prefix, "MAE_sd_q", proba_levels)}
  }
  if(sd){return(list(MAEs=MAEs, SDs=SDs))}
  return(MAEs)
}

#' Multilevel quantile losses
#'
#' @description Multilevel version of [quantile_loss()].
#'
#' @param y Vector of observations.
#' @param Pred_Q Matrix of of size `length(y)` times `proba_levels`,
#' whose columns are the quantile predictions at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param proba_levels Vector of probability levels at which the predictions were made.
#' Must be of length `ncol(Pred_Q)`.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output quantile errors (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the average quantile losses
#' between each column of `Pred_Q` and the observations.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "qloss_q", proba_levels)`.
#' @export
multilevel_q_loss <- function(y, Pred_Q, proba_levels, prefix="", na.rm=FALSE, give_names=TRUE){
  nb_prob_lvls_predict <- length(proba_levels)
  q_losses <- rep(as.double(NA), nb_prob_lvls_predict)
  for(i in 1:nb_prob_lvls_predict){
    q_losses[i] <- quantile_loss(y=y, y_hat=Pred_Q[,i], q=proba_levels[i], return_agg="mean", na.rm=na.rm)
  }
  if(give_names){names(q_losses) <- paste0(prefix, "qloss_q", proba_levels)}
  return(q_losses)
}

#' Multilevel prediction bias
#'
#' @description Multilevel version of [prediction_bias()].
#'
#' @param True_Q Matrix of size `n_obs` times `proba_levels`,
#' whose columns are the vectors of ground-truths at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param Pred_Q Matrix of the same size as `True_Q`,
#' whose columns are the predictions at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param proba_levels Vector of probability levels at which the predictions were made.
#' Must be of length `ncol(Pred_Q)`.
#' @param square_bias Whether to return the square bias (bool); defaults to `FALSE`.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output MSEs (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the (square) bias
#' of each columns of predictions in `Pred_Q` for the respective `True_Q`.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "MSE_q", proba_levels)`.
#' @export
multilevel_pred_bias <- function(True_Q, Pred_Q, proba_levels, square_bias=FALSE, prefix="", na.rm=FALSE, give_names=TRUE){
  nb_prob_lvls_predict <- length(proba_levels)
  biases <- rep(as.double(NA), nb_prob_lvls_predict)
  for(i in 1:nb_prob_lvls_predict){
    biases[i] <- prediction_bias(True_Q[,i], Pred_Q[,i], square_bias=square_bias, na.rm=na.rm)
  }
  if(give_names){names(biases) <- paste0(prefix, "bias_q", proba_levels)}
  return(biases)
}

#' Multilevel residual variance
#'
#' @description Multilevel version of [prediction_residual_variance()].
#'
#' @param True_Q Matrix of size `n_obs` times `proba_levels`,
#' whose columns are the vectors of ground-truths at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param Pred_Q Matrix of the same size as `True_Q`,
#' whose columns are the predictions at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param proba_levels Vector of probability levels at which the predictions were made.
#' Must be of length `ncol(Pred_Q)`.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output MSEs (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the residual variances
#' of each columns of predictions in `Pred_Q` for the respective `True_Q`.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "MSE_q", proba_levels)`.
#' @export
multilevel_resid_var <- function(True_Q, Pred_Q, proba_levels, prefix="", na.rm=FALSE, give_names=TRUE){
  nb_prob_lvls_predict <- length(proba_levels)
  vars <- rep(as.double(NA), nb_prob_lvls_predict)
  for(i in 1:nb_prob_lvls_predict){
    vars[i] <- prediction_residual_variance(True_Q[,i], Pred_Q[,i], na.rm=na.rm)
  }
  if(give_names){names(vars) <- paste0(prefix, "rvar_q", proba_levels)}
  return(vars)
}

#' Multilevel R squared
#'
#' @description Multilevel version of [R_squared()].
#'
#' @param True_Q Matrix of size `n_obs` times `proba_levels`,
#' whose columns are the vectors of ground-truths at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param Pred_Q Matrix of the same size as `True_Q`,
#' whose columns are the predictions at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param proba_levels Vector of probability levels at which the predictions were made.
#' Must be of length `ncol(Pred_Q)`.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output MSEs (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the R squared coefficient of determination
#' of each columns of predictions in `Pred_Q` for the respective `True_Q`.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "MSE_q", proba_levels)`.
#' @export
multilevel_R_squared <- function(True_Q, Pred_Q, proba_levels, prefix="", na.rm=FALSE, give_names=TRUE){
  nb_prob_lvls_predict <- length(proba_levels)
  R2s <- rep(as.double(NA), nb_prob_lvls_predict)
  for(i in 1:nb_prob_lvls_predict){
    R2s[i] <- R_squared(True_Q[,i], Pred_Q[,i], na.rm=na.rm)
  }
  if(give_names){names(R2s) <- paste0(prefix, "R2_q", proba_levels)}
  return(R2s)
}

#' Multilevel 'proportion_below'
#'
#' @description Multilevel version of [proportion_below()].
#'
#' @param y Vector of observations.
#' @param Pred_Q Matrix of of size `length(y)` times `proba_levels`,
#' whose columns are the quantile predictions at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param proba_levels Vector of probability levels at which the predictions were made.
#' Must be of length `ncol(Pred_Q)`.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output proportions (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the proportion of observations
#' below the predictions (`Pred_Q`) at each probability level.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "propBelow_q", proba_levels)`.
#' @export
multilevel_prop_below <- function(y, Pred_Q, proba_levels, prefix="", na.rm=FALSE, give_names=TRUE){
  nb_prob_lvls_predict <- length(proba_levels)
  props <- rep(as.double(NA), nb_prob_lvls_predict)
  for(i in 1:nb_prob_lvls_predict){
    props[i] <- proportion_below(y=y, Q_hat=Pred_Q[,i], na.rm=na.rm)
  }
  if(give_names){names(props) <- paste0(prefix, "propBelow_q", proba_levels)}#"Pred_loss_q"
  return(props)
}

#' Multilevel 'quantile_prediction_error'
#'
#' @description Multilevel version of [quantile_prediction_error()].
#'
#' @param y Vector of observations.
#' @param Pred_Q Matrix of of size `length(y)` times `proba_levels`,
#' whose columns are the quantile predictions at each `proba_levels` and
#' each row corresponds to an observation or realisation.
#' @param proba_levels Vector of probability levels at which the predictions were made.
#' Must be of length `ncol(Pred_Q)`.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output errors (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the quantile prediction error calibration metrics
#' between each column of `Pred_Q` and the observations.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "qPredErr_q", proba_levels)`.
#' @export
multilevel_q_pred_error <- function(y, Pred_Q, proba_levels, prefix="", na.rm=FALSE, give_names=TRUE){
  nb_prob_lvls_predict <- length(proba_levels)
  Pred_errs <- rep(as.double(NA), nb_prob_lvls_predict)
  for(i in 1:nb_prob_lvls_predict){
    Pred_errs[i] <- quantile_prediction_error(y=y, Q_hat=Pred_Q[,i], prob_level=proba_levels[i], na.rm=na.rm)
  }
  if(give_names){names(Pred_errs) <- paste0(prefix, "qPredErr_q", proba_levels)}#"Pred_loss_q"
  return(Pred_errs)
}

#' Multilevel 'quantile_exceedance_proba_error'
#'
#' @description Multilevel version of [quantile_exceedance_proba_error()].
#'
#' @param Probs Matrix, whose columns give, for each `proba_levels`,
#' the predicted probabilities to exceed or be smaller than a fixed quantile.
#' @param proba_levels Vector of probability levels of the quantiles.
#' @param return_years The probability levels can be given in term or return years instead.
#' Only used if `proba_levels` is not given.
#' @param type_probs Whether the predictions are the `"cdf"` (default) or `"exceedance"` probabilities.
#' @param prefix A string prefix to add to the output's names (if `give_names` is `TRUE`).
#' @param na.rm A logical value indicating whether `NA` values should be stripped before the computation proceeds.
#' @param give_names Whether to name the output errors (bool).
#'
#' @return A vector of length `length(proba_levels)` giving the [quantile_exceedance_proba_error()]
#' calibration metric of each column of `Probs` at the corresponding `proba_levels`.
#' If `give_names` is `TRUE`, the output vector is named `paste0(prefix, "exPrErr_q", proba_levels)`
#' (or `paste0(prefix, "exPrErr_", return_years,"y")` if `return_years` are given instead of `proba_levels`).
#' @export
multilevel_exceedance_proba_error <- function(Probs, proba_levels=NULL, return_years=NULL,
                                              type_probs=c("cdf","exceedance"), prefix="", na.rm=FALSE, give_names=TRUE){
  if((is.null(proba_levels)+is.null(return_years))!=1){
    stop("Exactly one of proba_levels or return_years should be provided in 'multilevel_exceedance_proba_error'.")
  }
  nb_prob_lvls_predict <- if(is.null(return_years)){length(proba_levels)}else{length(return_years)}
  cdf_errs <- rep(as.double(NA), nb_prob_lvls_predict)
  for(i in 1:nb_prob_lvls_predict){
    cdf_errs[i] <- quantile_exceedance_proba_error(Probs=Probs[,i], prob_level=proba_levels[i], return_years=return_years[i],
                                                   type_probs=type_probs, na.rm=na.rm)
  }
  if(give_names){
    if(is.null(return_years)){
      names(cdf_errs) <- paste0(prefix, "exPrErr_q", proba_levels)
    }else{
      names(cdf_errs) <- paste0(prefix, "exPrErr_", return_years,"y")
    }
  }
  return(cdf_errs)
}

