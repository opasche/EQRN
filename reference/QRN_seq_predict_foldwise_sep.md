# Sigle-fold foldwise fit-predict function using a recurrent QRN

Separated single-fold version of
[`QRN_seq_predict_foldwise()`](https://opasche.github.io/EQRN/reference/QRN_seq_predict_foldwise.md),
for computation purposes.

## Usage

``` r
QRN_seq_predict_foldwise_sep(
  X,
  y,
  q_level,
  n_folds = 3,
  fold_todo = 1,
  number_fits = 3,
  seq_len = 10,
  seed = NULL,
  ...
)
```

## Arguments

- X:

  Matrix of covariates, for training. Entries must be in sequential
  order.

- y:

  Response variable vector to model the conditional quantile of, for
  training. Entries must be in sequential order.

- q_level:

  Probability level of the desired conditional quantiles to predict.

- n_folds:

  Number of folds.

- fold_todo:

  Index of the fold to do (integer in 1:n_folds).

- number_fits:

  Number of restarts, for stability.

- seq_len:

  Data sequence length (i.e. number of past observations) used during
  training to predict each response quantile.

- seed:

  Integer random seed for reproducibility in network weight
  initialization.

- ...:

  Other parameters given to
  [`QRN_seq_fit()`](https://opasche.github.io/EQRN/reference/QRN_seq_fit.md).

## Value

A named list containing the foldwise predictions and fits. It namely
contains:

- predictions:

  the numerical vector of quantile predictions for each observation
  entry in y,

- fits:

  a list containing the `"QRN_seq"` fitted networks for each fold,

- cuts:

  the fold cuts indices,

- folds:

  a list of lists containing the train indices, validation indices and
  fold separations as a list for each fold setup,

- n_folds:

  number of folds,

- q_level:

  probability level of the predicted quantiles,

- train_losses:

  the vector of train losses on each fold,

- valid_losses:

  the vector of validation losses on each fold,

- min_valid_losses:

  the minimal validation losses obtained on each fold,

- min_valid_e:

  the epoch index of the minimal validation losses obtained on each
  fold.
