# Semi-conditional GPD MLEs and their train-validation likelihoods

Semi-conditional GPD MLEs and their train-validation likelihoods

## Usage

``` r
semiconditional_train_valid_GPD_loss(
  Y_train,
  Y_valid,
  interm_quant_train,
  interm_quant_valid
)
```

## Arguments

- Y_train:

  Vector of "training" observations on which to estimate the MLEs.

- Y_valid:

  Vector of "validation" observations, on which to estimate the out of
  training sample GPD loss.

- interm_quant_train:

  Vector of intermediate quantiles serving as a varying threshold for
  each training observation.

- interm_quant_valid:

  Vector of intermediate quantiles serving as a varying threshold for
  each validation observation.

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
