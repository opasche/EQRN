
#' Tail excess probability prediction based on conditional GPD parameters
#'
#' @param val Quantile value(s) used to estimate the conditional excess probability or cdf.
#' @param sigma Value(s) for the GPD scale parameter.
#' @param xi Value(s) for the GPD shape parameter.
#' @param interm_threshold Intermediate (conditional) quantile(s) at level `threshold_p` used as a (varying) threshold.
#' @param threshold_p Probability level of the intermediate conditional quantiles `interm_threshold`.
#' @param body_proba Value to use when the predicted conditional probability is below `threshold_p`
#' (in which case it cannot be precisely assessed by the model).
#' If `"default"` is given (the default), `paste0(">",1-threshold_p)` is used if `proba_type=="excess"`,
#' and `paste0("<",threshold_p)` is used if `proba_type=="cdf"`.
#' @param proba_type Whether to return the `"excess"` probability over `val` (default) or the `"cdf"` at `val`.
#'
#' @return Vector of probabilities (and possibly a few `body_proba` values if `val` is not large enough)
#' of the same length as the longest vector between `val`, `sigma`, `xi` and `interm_threshold`.
#' @export
#' @importFrom evd pgpd
GPD_excess_probability <- function(val, sigma, xi, interm_threshold, threshold_p, body_proba="default", proba_type=c("excess","cdf")){
  
  proba_type <- match.arg(proba_type)
  if(is.character(body_proba)){
    if(body_proba=="default"){
      body_proba <- if(proba_type=="excess"){paste0(">",1-threshold_p)}else{paste0("<",threshold_p)}
    }
  }
  
  n <- max(length(val), length(sigma), length(xi), length(interm_threshold))
  for(v in list(val, sigma, xi, interm_threshold)){
    if(length(v)!=1 & length(v)!=n){stop("val, sigma, xi, interm_threshold should be of equal or unit length in GPD_excess_probability.")}
  }
  val <- rep_len(c(val), n)
  sigma <- rep_len(c(sigma), n)
  xi <- rep_len(c(xi), n)
  interm_threshold <- rep_len(c(interm_threshold), n)
  
  Prob <- rep(as.double(NA), n)
  
  Prob[val==interm_threshold] <- threshold_p
  
  inds_ex <- val>interm_threshold
  Z <- val - interm_threshold
  gpd_cdf <- mapply(evd::pgpd, q=Z[inds_ex], loc=0, scale=sigma[inds_ex], shape=xi[inds_ex], SIMPLIFY=TRUE)
  Prob[inds_ex] <- threshold_p + gpd_cdf*(1-threshold_p)
  
  if(proba_type=="excess"){
    Prob[val>=interm_threshold] <- 1 - Prob[val>=interm_threshold]
  }
  if(any(val<interm_threshold)){
    Prob[val<interm_threshold] <- body_proba
  }
  
  return(Prob)
}


#' Maximum likelihood estimates for the GPD distribution using peaks over threshold
#'
#' @param Y Vector of observations
#' @param interm_lvl Probability level at which the empirical quantile should be used as the threshold,
#' if `thresh_quantiles` is not given.
#' @param thresh_quantiles Numerical value or numerical vector of the same length as `Y`
#' representing either a fixed or a varying threshold, respectively.
#'
#' @return Named list containing:
#' \item{scale}{the GPD scale MLE,}
#' \item{shape}{the GPD shape MLE,}
#' \item{fit}{the fitted [ismev::gpd.fit()] object.}
#' @export
#' @importFrom stats quantile
#' @importFrom ismev gpd.fit
fit_GPD_unconditional <- function(Y, interm_lvl=NULL, thresh_quantiles=NULL){
  ##
  if(is.null(interm_lvl) & is.null(thresh_quantiles)){
    fit <- ismev::gpd.fit(Y, 0, show = FALSE)
  } else {
    if(is.null(thresh_quantiles)){
      u <- stats::quantile(Y, interm_lvl)
      fit <- ismev::gpd.fit(Y, u, show = FALSE)
    } else {
      Y_rescaled <- Y - thresh_quantiles
      Y_excesses <- Y_rescaled[Y_rescaled>=0]
      fit <- ismev::gpd.fit(Y_excesses, 0, show = FALSE)
    }
  }
  sigma <- fit$mle[1]
  xi <- fit$mle[2]
  return(list(scale = sigma, shape = xi, fit = fit))
}


#' Predict unconditional extreme quantiles using peaks over threshold
#'
#' @param interm_lvl Probability level at which the empirical quantile should be used as the intermediate threshold.
#' @param quantiles Probability levels at which to predict the extreme quantiles.
#' @param Y Vector of ("training") observations.
#' @param ntest Number of "test" observations.
#'
#' @return Named list containing:
#' \item{predictions}{matrix of dimension `ntest` times `length(quantiles)`
#' containing the estimated extreme quantile at levels `quantile`, repeated `ntest` times,}
#' \item{pars}{matrix of dimension `ntest` times `2`
#' containing the two GPD parameter MLEs, repeated `ntest` times.}
#' \item{threshold}{The threshold for the peaks-over-threshold GPD model.
#' It is the empirical quantile of `Y` at level `interm_lvl`, i.e. `stats::quantile(Y, interm_lvl)`.}
#' @export
#' @importFrom stats quantile
#' @importFrom ismev gpd.fit
predict_unconditional_quantiles <- function(interm_lvl, quantiles = c(0.99), Y, ntest=1){
  
  p0 <- interm_lvl
  t0 <- stats::quantile(Y, p0)
  pars <- ismev::gpd.fit(Y, t0, show = FALSE)$mle
  sigma <- pars[1]
  xi <- pars[2]
  
  q_hat <- GPD_quantiles(quantiles, p0, t0, sigma, xi)
  
  predictions <- matrix(q_hat, nrow = ntest, ncol = length(quantiles), byrow = T)
  pars <- cbind(rep(sigma, ntest) , rep(xi, ntest))
  return(list(predictions = predictions, pars = pars, threshold=t0))
}


#' Predict semi-conditional extreme quantiles using peaks over threshold
#'
#' @param Y Vector of ("training") observations.
#' @param interm_lvl Probability level at which the empirical quantile should be used as the intermediate threshold.
#' @param thresh_quantiles Numerical vector of the same length as `Y`
#' representing the varying intermediate threshold on the train set.
#' @param interm_quantiles_test Numerical vector of the same length as `Y`
#' representing the varying intermediate threshold used for prediction on the test set.
#' @param prob_lvls_predict Probability levels at which to predict the extreme semi-conditional quantiles.
#'
#' @return Named list containing:
#' \item{predictions}{matrix of dimension `length(interm_quantiles_test)` times `length(prob_lvls_predict)`
#' containing the estimated extreme quantile at levels `quantile`, for each `interm_quantiles_test`,}
#' \item{pars}{matrix of dimension `ntest` times `2`
#' containing the two GPD parameter MLEs, repeated `length(interm_quantiles_test)` times.}
#' @export
predict_GPD_semiconditional <- function(Y, interm_lvl, thresh_quantiles, interm_quantiles_test=thresh_quantiles,
                                        prob_lvls_predict = c(0.99)){
  ##
  fit <- fit_GPD_unconditional(Y, interm_lvl=interm_lvl, thresh_quantiles=thresh_quantiles)
  nb_prob_lvls_predict <- length(prob_lvls_predict)
  predicted_quantiles <- matrix(as.double(NA),nrow=length(interm_quantiles_test), ncol=nb_prob_lvls_predict)
  for(i in 1:nb_prob_lvls_predict){
    predicted_quantiles[,i] <- GPD_quantiles(prob_lvls_predict[i], interm_lvl, interm_quantiles_test, fit$scale, fit$shape)
  }
  return(list(predictions=predicted_quantiles, pars=c(scale=fit$scale, shape=fit$shape)))
}


#' Generalized Pareto likelihood loss
#'
#' @param sigma Value(s) for the GPD scale parameter.
#' @param xi Value(s) for the GPD shape parameter.
#' @param y Vector of observations
#' @param rescaled Whether y already is a vector of excesses (TRUE) or needs rescaling (FALSE).
#' @param interm_lvl Probability level at which the empirical quantile should be used as the intermediate threshold
#' to compute the excesses, if `rescaled==FALSE`.
#' @param return_vector Whether to return the the vector of GPD losses for each observation
#' instead of the negative log-likelihood (average loss).
#'
#' @return GPD negative log-likelihood of the GPD parameters over the sample of observations
#' @export
#' @importFrom stats quantile
loss_GPD <- function(sigma, xi, y, rescaled=TRUE, interm_lvl=NULL, return_vector=FALSE){
  if(!rescaled){
    if(is.null(interm_lvl)){stop("Must provide POT interm_lvl value to rescale y vector in 'loss_GPD'.")}
    u <- stats::quantile(y, interm_lvl)
    y_rescaled <- y - u
    y_excesses <- y_rescaled[y_rescaled>=0]
  }else{
    y_excesses <- y
  }
  loss <- (1 + 1/xi) * log(1 + xi*y_excesses/sigma) + log(sigma)
  if(!return_vector){
    loss <- mean(loss)
  }
  return(loss)
}


#' Unconditional GPD MLEs and their train-validation likelihoods
#'
#' @param Y_train Vector of "training" observations on which to estimate the MLEs.
#' @param interm_lvl Probability level at which the empirical quantile should be used as the threshold.
#' @param Y_valid Vector of "validation" observations, on which to estimate the out of training sample GPD loss.
#'
#' @return Named list containing:
#' \item{scale}{GPD scale MLE inferred from the train set,}
#' \item{shape}{GPD shape MLE inferred from the train set,}
#' \item{train_loss}{the negative log-likelihoods of the MLEs over the training samples,}
#' \item{valid_loss}{the negative log-likelihoods of the MLEs over the validation samples.}
#' @export
unconditional_train_valid_GPD_loss <- function(Y_train, interm_lvl, Y_valid){
  ##
  fit <- fit_GPD_unconditional(Y_train, interm_lvl=interm_lvl, thresh_quantiles=NULL)
  train_loss <- loss_GPD(fit$scale, fit$shape, Y_train, rescaled=FALSE, interm_lvl=interm_lvl, return_vector=FALSE)
  valid_loss <- loss_GPD(fit$scale, fit$shape, Y_valid, rescaled=FALSE, interm_lvl=interm_lvl, return_vector=FALSE)
  return(list(train_loss = train_loss, valid_loss = valid_loss, scale = fit$scale, shape = fit$shape))
}


#' Semi-conditional GPD MLEs and their train-validation likelihoods
#'
#' @param Y_train Vector of "training" observations on which to estimate the MLEs.
#' @param Y_valid Vector of "validation" observations, on which to estimate the out of training sample GPD loss.
#' @param interm_quant_train Vector of intermediate quantiles serving as a varying threshold for each training observation.
#' @param interm_quant_valid Vector of intermediate quantiles serving as a varying threshold for each validation observation.
#'
#' @return Named list containing:
#' \item{scale}{GPD scale MLE inferred from the train set,}
#' \item{shape}{GPD shape MLE inferred from the train set,}
#' \item{train_loss}{the negative log-likelihoods of the MLEs over the training samples,}
#' \item{valid_loss}{the negative log-likelihoods of the MLEs over the validation samples.}
#' @export
semiconditional_train_valid_GPD_loss <- function(Y_train, Y_valid, interm_quant_train, interm_quant_valid){
  ##
  Y_tr_rescaled <- Y_train - interm_quant_train
  Y_tr_excesses <- Y_tr_rescaled[Y_tr_rescaled>=0]
  Y_va_rescaled <- Y_valid - interm_quant_valid
  Y_va_excesses <- Y_va_rescaled[Y_va_rescaled>=0]
  fit <- fit_GPD_unconditional(Y_tr_excesses, interm_lvl=NULL, thresh_quantiles=NULL)
  train_loss <- loss_GPD(fit$scale, fit$shape, Y_tr_excesses, rescaled=TRUE, interm_lvl=NULL, return_vector=FALSE)
  valid_loss <- loss_GPD(fit$scale, fit$shape, Y_va_excesses, rescaled=TRUE, interm_lvl=NULL, return_vector=FALSE)
  return(list(train_loss = train_loss, valid_loss = valid_loss, scale = fit$scale, shape = fit$shape))
}


#' Compute extreme quantile from GPD parameters
#'
#' @param p Probability level of the desired extreme quantile.
#' @param p0 Probability level of the (possibly varying) intermediate threshold/quantile.
#' @param t_x0 Value(s) of the (possibly varying) intermediate threshold/quantile.
#' @param sigma Value(s) for the GPD scale parameter.
#' @param xi Value(s) for the GPD shape parameter.
#'
#' @return The quantile value at probability level `p`.
#' @export
GPD_quantiles <- function(p, p0, t_x0, sigma, xi){
  ## numeric(0, 1) numeric(0, 1) numeric_vector numeric_vector numeric_vector
  ## -> numeric_vector
  ## produce the estimated extreme quantiles of GPD
  
  if(any(xi==0)){
    gpd_qs <- rep(as.double(NA), max(length(t_x0),length(sigma),length(xi)))
    if(length(xi)==1){xi <- rep(xi,length(gpd_qs))}
    gpd_qs[xi!=0] <- ((((1-p)/(1-p0))^{-xi} - 1) * (sigma / xi) + t_x0)[xi!=0]
    gpd_qs[xi==0] <- (log((1-p0)/(1-p)) * sigma + t_x0)[xi==0]
  }else{
    gpd_qs <- (((1-p)/(1-p0))^{-xi} - 1) * (sigma / xi) + t_x0
  }
  return(gpd_qs)
}

