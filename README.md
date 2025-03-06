
<!-- README.md is generated from README.Rmd. Please edit that file -->

# EQRN: Extreme Quantile Regression Neural Networks for Conditionnal Risk Prediction

<!-- badges: start -->

[![R-CMD-check](https://github.com/opasche/EQRN/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/opasche/EQRN/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

A user-friendly framework for forecasting and extrapolating extreme
measures of conditional risk using flexible neural network
architectures. It allows for capturing complex multivariate
dependencies, including dependencies between observations, such as
sequential (time) dependence. The implementation is based on the article
“Neural networks for extreme quantile regression with an application to
forecasting of flood risk” (Pasche and Engelke, 2024,
[DOI:10.1214/24-AOAS1907](https://doi.org/10.1214/24-AOAS1907)).

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

## Basic usage example for exchangeable data

The minimal example below illustrates, in three simple steps, how to use
the package functions to fit the EQRN model and predict extreme
conditional quantiles and other metrics on new test data. In this
example, a toy i.i.d. dataset is used.

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

- the argument `shape_fixed=TRUE` removes covariate dependence from the
  shape output,
- the argument `net_structure=c(5,5)` sets two hidden layers of 5
  neurons each as an architecture,
- the network is trained for 100 epochs (with a seed for
  reproducibility).

``` r
library(EQRN)

fit_eqrn <- EQRN_fit(X_train, y_train, intermediateq_train, interm_lvl,
                     shape_fixed=TRUE, net_structure=c(5,5), n_epochs=100, seed=42)
#> Epoch: 1 out of 100 , average train loss: 2.371921
#> Epoch: 100 out of 100 , average train loss: 2.281698
```

The arguments values are here arbitrarily chosen for illustration. As
for any machine learning approach, hyperparameters should be tuned using
set-aside validation data to obtain an accurate fit. Stopping criteria
are also available for the number of fitting epochs. Refer to the
[documentation](https://opasche.github.io/EQRN/reference/index.html#fitting-eqrn-tail-neural-networks)
for a detailed description of the arguments, and to the [article’s
repository](https://github.com/opasche/EQRN_Results) for more advanced
examples.

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
#> 1  0.5876351 0.83797214    2.763170      15.98885       34.31405    0.004118278
#> 2  0.9493471 0.74616973    2.123687      15.42404       33.85271    0.003459142
#> 3  0.7456916 0.24237508    2.335307      15.40873       33.52297    0.003537242
#> 4  0.2319869 0.70261432    3.041587      16.04881       34.07131    0.004325655
#> 5  0.7706744 0.19874048    4.725495      18.14131       36.72994    0.008187588
#> 6  0.7746018 0.03440777    5.414535      18.87249       37.51951    0.010778085
#> 7  0.7776956 0.33728896    2.796728      15.98395       34.25587    0.004133909
#> 8  0.2586140 0.49574342    3.775091      16.84983       34.96588    0.005546904
#> 9  0.7935616 0.60815766    2.608452      15.86056       34.22237    0.003949164
#> 10 0.1613134 0.52083200    1.770021      14.48766       32.10893    0.002831878
```

## References and links

> Pasche, O. C. and Engelke, S. (2024). “Neural networks for extreme
> quantile regression with an application to forecasting of flood risk”.
> <i>Annals of Applied Statistics</i> 18(4), 2818–2839.
> <https://doi.org/10.1214/24-AOAS1907>

**Published article:**
[DOI:10.1214/24-AOAS1907](https://doi.org/10.1214/24-AOAS1907)
([PDF](https://raw.githubusercontent.com/opasche/EQRN_Results/main/article/24-AOAS1907.pdf),
[Supplement](https://raw.githubusercontent.com/opasche/EQRN_Results/main/article/aoas1907suppa.pdf)).  
**Article’s usage examples:** <https://github.com/opasche/EQRN_Results>

Preprint (obsolete):
[ArXiv:2208.07590](https://arxiv.org/abs/2208.07590)
([PDF](https://arxiv.org/pdf/2208.07590)).

------------------------------------------------------------------------

Package created by Olivier C. PASCHE  
Research Center for Statistics, University of Geneva (CH), 2022.  
Supported by the Swiss National Science Foundation.
