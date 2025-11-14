# Excess Probability Predictions

A generic function (method) for excess probability predictions from
various fitted EQR models. The function invokes particular methods which
depend on the class of the first argument.

## Usage

``` r
excess_probability(object, ...)
```

## Arguments

- object:

  A model object for which excess probability prediction is desired.

- ...:

  additional model-specific arguments affecting the predictions
  produced. See the corresponding method documentation.

## Value

The excess probability estimates from the given EQR model.
