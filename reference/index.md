# Package index

## Fitting EQRN Tail Neural Networks

Functions to fit a tail GPD EQRN network to intermediate quantile
exceedences

- [`EQRN_fit_restart()`](https://opasche.github.io/EQRN/reference/EQRN_fit_restart.md)
  : Wrapper for fitting EQRN with restart for stability
- [`EQRN_fit()`](https://opasche.github.io/EQRN/reference/EQRN_fit.md) :
  EQRN fit function for independent data
- [`EQRN_fit_seq()`](https://opasche.github.io/EQRN/reference/EQRN_fit_seq.md)
  : EQRN fit function for sequential and time series data

## Predicting using fitted EQRN networks

- [`EQRN_predict()`](https://opasche.github.io/EQRN/reference/EQRN_predict.md)
  : Predict function for an EQRN_iid fitted object
- [`EQRN_predict_seq()`](https://opasche.github.io/EQRN/reference/EQRN_predict_seq.md)
  : Predict function for an EQRN_seq fitted object
- [`EQRN_predict_params()`](https://opasche.github.io/EQRN/reference/EQRN_predict_params.md)
  : GPD parameters prediction function for an EQRN_iid fitted object
- [`EQRN_predict_params_seq()`](https://opasche.github.io/EQRN/reference/EQRN_predict_params_seq.md)
  : GPD parameters prediction function for an EQRN_seq fitted object
- [`EQRN_excess_probability()`](https://opasche.github.io/EQRN/reference/EQRN_excess_probability.md)
  : Tail excess probability prediction using an EQRN_iid object
- [`EQRN_excess_probability_seq()`](https://opasche.github.io/EQRN/reference/EQRN_excess_probability_seq.md)
  : Tail excess probability prediction using an EQRN_seq object
- [`compute_EQRN_GPDLoss()`](https://opasche.github.io/EQRN/reference/compute_EQRN_GPDLoss.md)
  : Generalized Pareto likelihood loss of a EQRN_iid predictor
- [`compute_EQRN_seq_GPDLoss()`](https://opasche.github.io/EQRN/reference/compute_EQRN_seq_GPDLoss.md)
  : Generalized Pareto likelihood loss of a EQRN_seq predictor

## Saving and Loading

- [`EQRN_save()`](https://opasche.github.io/EQRN/reference/EQRN_save.md)
  : Save an EQRN object on disc
- [`EQRN_load()`](https://opasche.github.io/EQRN/reference/EQRN_load.md)
  : Load an EQRN object from disc

## Neural Networks Helpers

- [`install_backend()`](https://opasche.github.io/EQRN/reference/install_backend.md)
  : Install Torch Backend Libraries
- [`backend_is_installed()`](https://opasche.github.io/EQRN/reference/backend_is_installed.md)
  : Check if Torch Backend Libraries are Installed
- [`default_device()`](https://opasche.github.io/EQRN/reference/default_device.md)
  : Default torch device
- [`loss_GPD_tensor()`](https://opasche.github.io/EQRN/reference/loss_GPD_tensor.md)
  : GPD tensor loss function for training a EQRN network
- [`quantile_loss_tensor()`](https://opasche.github.io/EQRN/reference/quantile_loss_tensor.md)
  : Tensor quantile loss function for training a QRN network
- [`get_excesses()`](https://opasche.github.io/EQRN/reference/get_excesses.md)
  : Computes rescaled excesses over the conditional quantiles
- [`process_features()`](https://opasche.github.io/EQRN/reference/process_features.md)
  : Feature processor for EQRN
- [`perform_scaling()`](https://opasche.github.io/EQRN/reference/perform_scaling.md)
  : Performs feature scaling without overfitting
- [`mts_dataset()`](https://opasche.github.io/EQRN/reference/mts_dataset.md)
  : Dataset creator for sequential data
- [`FC_GPD_net()`](https://opasche.github.io/EQRN/reference/FC_GPD_net.md)
  : MLP module for GPD parameter prediction
- [`FC_GPD_SNN()`](https://opasche.github.io/EQRN/reference/FC_GPD_SNN.md)
  : Self-normalized fully-connected network module for GPD parameter
  prediction
- [`Separated_GPD_SNN()`](https://opasche.github.io/EQRN/reference/Separated_GPD_SNN.md)
  : Self-normalized separated network module for GPD parameter
  prediction
- [`Recurrent_GPD_net()`](https://opasche.github.io/EQRN/reference/Recurrent_GPD_net.md)
  : Recurrent network module for GPD parameter prediction
- [`QRNN_RNN_net()`](https://opasche.github.io/EQRN/reference/QRNN_RNN_net.md)
  : Recurrent quantile regression neural network module

## Extreme Value Analysis Helpers

- [`GPD_excess_probability()`](https://opasche.github.io/EQRN/reference/GPD_excess_probability.md)
  : Tail excess probability prediction based on conditional GPD
  parameters
- [`fit_GPD_unconditional()`](https://opasche.github.io/EQRN/reference/fit_GPD_unconditional.md)
  : Maximum likelihood estimates for the GPD distribution using peaks
  over threshold
- [`predict_unconditional_quantiles()`](https://opasche.github.io/EQRN/reference/predict_unconditional_quantiles.md)
  : Predict unconditional extreme quantiles using peaks over threshold
- [`predict_GPD_semiconditional()`](https://opasche.github.io/EQRN/reference/predict_GPD_semiconditional.md)
  : Predict semi-conditional extreme quantiles using peaks over
  threshold
- [`loss_GPD()`](https://opasche.github.io/EQRN/reference/loss_GPD.md) :
  Generalized Pareto likelihood loss
- [`unconditional_train_valid_GPD_loss()`](https://opasche.github.io/EQRN/reference/unconditional_train_valid_GPD_loss.md)
  : Unconditional GPD MLEs and their train-validation likelihoods
- [`semiconditional_train_valid_GPD_loss()`](https://opasche.github.io/EQRN/reference/semiconditional_train_valid_GPD_loss.md)
  : Semi-conditional GPD MLEs and their train-validation likelihoods
- [`GPD_quantiles()`](https://opasche.github.io/EQRN/reference/GPD_quantiles.md)
  : Compute extreme quantile from GPD parameters

## Intermediate Models

- [`QRN_seq_fit()`](https://opasche.github.io/EQRN/reference/QRN_seq_fit.md)
  : Recurrent QRN fitting function
- [`QRN_fit_multiple()`](https://opasche.github.io/EQRN/reference/QRN_fit_multiple.md)
  : Wrapper for fitting a recurrent QRN with restart for stability
- [`QRN_seq_predict()`](https://opasche.github.io/EQRN/reference/QRN_seq_predict.md)
  : Predict function for a QRN_seq fitted object
- [`QRN_seq_predict_foldwise()`](https://opasche.github.io/EQRN/reference/QRN_seq_predict_foldwise.md)
  : Foldwise fit-predict function using a recurrent QRN
- [`QRN_seq_predict_foldwise_sep()`](https://opasche.github.io/EQRN/reference/QRN_seq_predict_foldwise_sep.md)
  : Sigle-fold foldwise fit-predict function using a recurrent QRN

## Accuracy metrics

- [`mean_squared_error()`](https://opasche.github.io/EQRN/reference/mean_squared_error.md)
  : Mean squared error
- [`mean_absolute_error()`](https://opasche.github.io/EQRN/reference/mean_absolute_error.md)
  : Mean absolute error
- [`square_loss()`](https://opasche.github.io/EQRN/reference/square_loss.md)
  : Square loss
- [`quantile_loss()`](https://opasche.github.io/EQRN/reference/quantile_loss.md)
  : Quantile loss
- [`prediction_bias()`](https://opasche.github.io/EQRN/reference/prediction_bias.md)
  : Prediction bias
- [`prediction_residual_variance()`](https://opasche.github.io/EQRN/reference/prediction_residual_variance.md)
  : Prediction residual variance
- [`R_squared()`](https://opasche.github.io/EQRN/reference/R_squared.md)
  : R squared
- [`proportion_below()`](https://opasche.github.io/EQRN/reference/proportion_below.md)
  : Proportion of observations below conditional quantile vector
- [`quantile_prediction_error()`](https://opasche.github.io/EQRN/reference/quantile_prediction_error.md)
  : Quantile prediction calibration error
- [`quantile_exceedance_proba_error()`](https://opasche.github.io/EQRN/reference/quantile_exceedance_proba_error.md)
  : Quantile exceedance probability prediction calibration error
- [`multilevel_MSE()`](https://opasche.github.io/EQRN/reference/multilevel_MSE.md)
  : Multilevel quantile MSEs
- [`multilevel_MAE()`](https://opasche.github.io/EQRN/reference/multilevel_MAE.md)
  : Multilevel quantile MAEs
- [`multilevel_q_loss()`](https://opasche.github.io/EQRN/reference/multilevel_q_loss.md)
  : Multilevel quantile losses
- [`multilevel_pred_bias()`](https://opasche.github.io/EQRN/reference/multilevel_pred_bias.md)
  : Multilevel prediction bias
- [`multilevel_resid_var()`](https://opasche.github.io/EQRN/reference/multilevel_resid_var.md)
  : Multilevel residual variance
- [`multilevel_R_squared()`](https://opasche.github.io/EQRN/reference/multilevel_R_squared.md)
  : Multilevel R squared
- [`multilevel_prop_below()`](https://opasche.github.io/EQRN/reference/multilevel_prop_below.md)
  : Multilevel 'proportion_below'
- [`multilevel_q_pred_error()`](https://opasche.github.io/EQRN/reference/multilevel_q_pred_error.md)
  : Multilevel 'quantile_prediction_error'
- [`multilevel_exceedance_proba_error()`](https://opasche.github.io/EQRN/reference/multilevel_exceedance_proba_error.md)
  : Multilevel 'quantile_exceedance_proba_error'

## Other Helpers

- [`check_directory()`](https://opasche.github.io/EQRN/reference/check_directory.md)
  : Check directory existence
- [`safe_save_rds()`](https://opasche.github.io/EQRN/reference/safe_save_rds.md)
  : Safe RDS save
- [`last_elem()`](https://opasche.github.io/EQRN/reference/last_elem.md)
  : Last element of a vector
- [`roundm()`](https://opasche.github.io/EQRN/reference/roundm.md) :
  Mathematical number rounding
- [`vec2mat()`](https://opasche.github.io/EQRN/reference/vec2mat.md) :
  Convert a vector to a matrix
- [`make_folds()`](https://opasche.github.io/EQRN/reference/make_folds.md)
  : Create cross-validation folds
- [`lagged_features()`](https://opasche.github.io/EQRN/reference/lagged_features.md)
  : Covariate lagged replication for temporal dependence
- [`vector_insert()`](https://opasche.github.io/EQRN/reference/vector_insert.md)
  : Insert value in vector
- [`get_doFuture_operator()`](https://opasche.github.io/EQRN/reference/get_doFuture_operator.md)
  : Get doFuture operator
- [`set_doFuture_strategy()`](https://opasche.github.io/EQRN/reference/set_doFuture_strategy.md)
  : Set a doFuture execution strategy
- [`end_doFuture_strategy()`](https://opasche.github.io/EQRN/reference/end_doFuture_strategy.md)
  : End the currently set doFuture strategy

## Prediction methods

S3 class method support for classes `EQRN_iid`, `EQRN_seq` and
`QRN_seq`, and methods `predict` and `excess_probability`, as a
facultative alternative to their respective functions above.

- [`predict(`*`<EQRN_iid>`*`)`](https://opasche.github.io/EQRN/reference/predict.EQRN_iid.md)
  : Predict method for an EQRN_iid fitted object
- [`predict(`*`<EQRN_seq>`*`)`](https://opasche.github.io/EQRN/reference/predict.EQRN_seq.md)
  : Predict method for an EQRN_seq fitted object
- [`predict(`*`<QRN_seq>`*`)`](https://opasche.github.io/EQRN/reference/predict.QRN_seq.md)
  : Predict method for a QRN_seq fitted object
- [`excess_probability()`](https://opasche.github.io/EQRN/reference/excess_probability.md)
  : Excess Probability Predictions
- [`excess_probability(`*`<EQRN_iid>`*`)`](https://opasche.github.io/EQRN/reference/excess_probability.EQRN_iid.md)
  : Tail excess probability prediction method using an EQRN_iid object
- [`excess_probability(`*`<EQRN_seq>`*`)`](https://opasche.github.io/EQRN/reference/excess_probability.EQRN_seq.md)
  : Tail excess probability prediction method using an EQRN_iid object
