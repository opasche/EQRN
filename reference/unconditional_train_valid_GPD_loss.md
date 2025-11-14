# Unconditional GPD MLEs and their train-validation likelihoods

Unconditional GPD MLEs and their train-validation likelihoods

## Usage

``` r
unconditional_train_valid_GPD_loss(Y_train, interm_lvl, Y_valid)
```

## Arguments

- Y_train:

  Vector of "training" observations on which to estimate the MLEs.

- interm_lvl:

  Probability level at which the empirical quantile should be used as
  the threshold.

- Y_valid:

  Vector of "validation" observations, on which to estimate the out of
  training sample GPD loss.

## Value

Named list containing:

- scale:

  GPD scale MLE inferred from the train set,

- shape:

  GPD shape MLE inferred from the train set,

- train_loss:

  the negative log-likelihoods of the MLEs over the training samples,

- valid_loss:

  the negative log-likelihoods of the MLEs over the validation samples.
