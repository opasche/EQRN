
<!-- README.md is generated from README.Rmd. Please edit that file -->

# EQRN: Extreme Quantile Regression Neural Networks for Conditionnal Risk Assessment

<!-- badges: start -->

[![R-CMD-check](https://github.com/opasche/EQRN/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/opasche/EQRN/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

A user-friendly framework for forecasting and extrapolating extreme
measures of conditional risk using flexible neural network
architectures. It allows for capturing complex multivariate
dependencies, including dependencies between observations, such as
sequential dependence (time-series). The implementation is based on the
article “Neural Networks for Extreme Quantile Regression with an
Application to Forecasting of Flood Risk” by Olivier C. Pasche and
Sebastian Engelke
([ArXiv:2208.07590](https://arxiv.org/abs/2208.07590)).

## Installation

To install the development version of EQRN, simply run from R:

``` r
# install.packages("devtools")
devtools::install_github("opasche/EQRN")
```

When the package is first loaded after installation (e.g. with
`library(EQRN)` or `EQRN::fct()`), the necessary backend software from
the [`torch`](https://torch.mlverse.org/) dependency is automatically
installed. (Alternatively, `EQRN::install_backend()` can be called to
perform the backend installation manually.) For more information about
the torch backend and troubleshooting, visit the [torch installation
guide](https://torch.mlverse.org/docs/articles/installation.html).

## Motivation

Risk assessment for extreme events requires accurate estimation of high
quantiles that go beyond the range of historical observations. When the
risk depends on the values of observed predictors, regression techniques
are used to interpolate in the predictor space. In this package we
propose the EQRN model that combines tools from neural networks and
extreme value theory into a method capable of extrapolation in the
presence of complex predictor dependence. Neural networks can naturally
incorporate additional structure in the data. The recurrent version of
EQRN is able to capture complex sequential dependence in time series.

In [the corresponding article](https://arxiv.org/abs/2208.07590), EQRN
is applied to forecasting of flood risk in the Swiss Aare catchment. It
exploits information from multiple covariates in space and time to
provide one-day-ahead predictions of return levels and exceedances
probabilities. This output complements the static return level from a
traditional extreme value analysis and the predictions are able to adapt
to distributional shifts as experienced in a changing climate. Our model
can help authorities to manage flooding more effectively and to minimize
their disastrous impacts through early warning systems.

## Basic Usage Example for Exchangeable Data

The example below shows in three simple steps how to fit the EQRN model
and predict extreme conditional quantiles and other metrics on new test
data. In this example, a toy i.i.d. dataset is used.

### 0. Generate a toy dataset

``` r
scale_fct <- function(x1,x2){ 3 + cos(x1 + x2 + 0.5) }

set.seed(1)
X_train <- matrix(stats::runif(5120), ncol=2)
y_train <- scale_fct(X_train[,1], X_train[,2]) * stats::rt(2560, 4)

X_test <- matrix(stats::runif(2560), ncol=2)
```

### Step 1. Construct intermediate quantiles

This can be achieved with any suitable quantile regression method. We
here use generalised random forests from the
[`grf`](https://grf-labs.github.io/grf/) package, as they are very easy
to use and already quite flexible. One could for example use a quantile
regression neural network instead.

``` r
library(grf)

# Choose an intermediate probability level.
interm_lvl <- 0.8

# Fit a GRF for quantile regression with 500 trees (the more the better) on the training set (with a seed for reproducibility).
fit_grf <- grf::quantile_forest(X_train, y_train, num.trees=1000, seed=21)

# Construct out-of-bag intermediate quantiles on the training set.
intermediateq_train <- predict(fit_grf, newdata=NULL, quantiles=c(interm_lvl))$predictions
```

### Step 2. Fit the tail model

Fit the EQRN network on the training set, with the intermediate
quantiles as a varying threshold. Here:

-   the argument `shape_fixed=TRUE` removes covariate dependence from
    the shape output,
-   the argument `net_structure=c(5,5)` sets two hidden layers of 5
    neurons each as an architecture,
-   the network is trained for 100 epochs (with a seed for
    reproducibility).

``` r
library(EQRN)

fit_eqrn <- EQRN_fit(X_train, y_train, intermediateq_train, interm_lvl,
                     shape_fixed=TRUE, net_structure=c(5,5), n_epochs=100, seed=42)
#> Epoch: 1 out of 100 , average train loss: 2.404647
#> Epoch: 100 out of 100 , average train loss: 2.311854
```

### Step 3. Predict conditional quantiles and risk metrics for new test observations

``` r
# Desired probability levels at which to predict the conditional quantiles.
levels_predict <- c(0.999, 0.9999)

# Predict intermediate test quantiles using the intermediate model.
intermediateq_test <- predict(fit_grf, newdata=X_test, quantiles=c(interm_lvl))$predictions

# Predict the desired conditional extreme quantiles on the test set.
qpred_eqrn <- EQRN_predict(fit_eqrn, X_test, levels_predict, intermediateq_test)

# Forecast the probability that Y_test would exceed a certain large value.
large_value <- 10
ppred_eqrn <- EQRN_excess_probability(large_value, fit_eqrn, X_test, intermediateq_test)
```

``` r
# Print some predictions:
hn <- 10
results <- data.frame(X1=X_test[1:hn,1], X2=X_test[1:hn,2], pred_Y_Q_80=intermediateq_test[1:hn],
                      pred_Y_Q_99.9=qpred_eqrn[1:hn,1], pred_Y_Q_99.99=qpred_eqrn[1:hn,2], Pr_Y_exceed_10=ppred_eqrn[1:hn])

print(results)
#>           X1         X2 pred_Y_Q_80 pred_Y_Q_99.9 pred_Y_Q_99.99 Pr_Y_exceed_10
#> 1  0.5876351 0.83797214    2.741546      15.87502       33.91469    0.004054095
#> 2  0.9493471 0.74616973    2.123687      15.33688       33.48605    0.003427602
#> 3  0.7456916 0.24237508    2.698117      15.73987       33.65356    0.003939998
#> 4  0.2319869 0.70261432    3.041587      15.96008       33.70446    0.004288765
#> 5  0.7706744 0.19874048    4.725495      18.06010       36.37603    0.008155741
#> 6  0.7746018 0.03440777    5.414535      18.79252       37.16805    0.010750530
#> 7  0.7776956 0.33728896    3.136478      16.29012       34.35749    0.004594990
#> 8  0.2586140 0.49574342    3.457248      16.39834       34.17376    0.004917901
#> 9  0.7935616 0.60815766    2.557475      15.71578       33.78956    0.003852441
#> 10 0.1613134 0.52083200    2.045515      14.71575       32.11914    0.003049184
```

------------------------------------------------------------------------

Package created by Olivier C. PASCHE  
Research Center for Statistics, University of Geneva (CH), 2022.  
Supported by the Swiss National Science Foundation.
