crossfit: Cross-Fitting Engine for Double/Debiased ML
================

# crossfit

A small cross-fitting engine for **double / debiased machine learning**
and other meta-learners.

The package lets you define:

- a **target** functional (e.g. ATE, risk, regression error),
- a **graph of nuisance models** (propensity scores, regressions, etc.),
- how many **folds** each node trains on (`train_fold`),
- how many **folds** the target evaluates on (`eval_fold`),

and then runs a cross-fitting schedule with configurable aggregation
over **panels** and **repetitions**.

## Installation

You can install the development version from GitHub:

``` r
# install.packages("remotes")
remotes::install_github("EtiennePeyrot/crossfit-R")
```

Then load it as usual:

``` r
library(crossfit)
```

## Overview

`crossfit` is designed for settings where:

- you care about a **low-dimensional target** (ATE, a coefficient, a
  risk, …),
- the target depends on **high-dimensional nuisance functions**
  estimated by ML.

The engine:

- enforces **out-of-sample** use of nuisances via K-fold cross-fitting,

- supports an arbitrary **DAG of nuisances** (not just one or two),

- lets each node choose its own `train_fold` (how many folds it trains
  on),

- lets the target choose its `eval_fold` (how many folds it evaluates
  on),

- supports several fold allocation schemes: `"independence"`,
  `"overlap"`, `"disjoint"`,

- has two modes:

  - `mode = "estimate"` → returns a numeric estimate of the target,
  - `mode = "predict"` → returns a cross-fitted **prediction function**.

Internally, the graph is normalized into a set of **instances** with
structural signatures, so that identical models can share fits and be
cached efficiently.

## Quick example: cross-fitted MSE

Here is a minimal example on a simple regression problem.  
We define a nuisance $`m(x) = E[Y \mid X]`$ and use the cross-fitted
mean squared error of this nuisance as our target.

``` r
library(crossfit)

set.seed(1)
n <- 200
x <- rnorm(n)
y <- x + rnorm(n)
data <- data.frame(x = x, y = y)

# 1) Nuisance: regression m(x) = E[Y | X]
nuis_y <- create_nuisance(
  fit = function(data, ...) lm(y ~ x, data = data),
  predict = function(model, data, ...) {
    as.numeric(predict(model, newdata = data))
  }
)

# 2) Target: cross-fitted MSE of m(x)
target_mse <- function(data, nuis_y, ...) {
  mean((data$y - nuis_y)^2)
}

# 3) Method: use 4 folds, 3 repetitions, DML-style "independence" allocation
method <- create_method(
  target = target_mse,
  list_nuisance = list(nuis_y = nuis_y),
  folds = 4,
  repeats = 3,
  eval_fold = 1,
  mode = "estimate",
  fold_allocation = "independence",
  aggregate_panels  = mean_estimate,
  aggregate_repeats = mean_estimate
)

res <- crossfit(data, method)

str(res$estimates)
res$estimates[[1]]
```

The `crossfit()` call:

- builds the nuisance / target graph,

- runs K-fold cross-fitting for `repeats` repetitions,

- aggregates over panels and repetitions using `mean_estimate()`,

- returns a list with:

  - `estimates` – one entry per method (here just one),

  - `per_method` – panel-wise and repetition-wise values and errors,

  - `repeats_done` – number of successful repetitions per method,

  - `K`, `K_required`, `methods`, `plan` – diagnostics and internals.

## Multiple methods and shared nuisances

You can run several methods in parallel, sharing some or all nuisances.
For example, we can estimate both:

- the cross-fitted MSE of $`m(x)`$,
- the cross-fitted **mean** of $`m(x)`$,

in a single call:

``` r
target_mean <- function(data, nuis_y, ...) {
  mean(nuis_y)
}

m_mse <- create_method(
  target = target_mse,
  list_nuisance = list(nuis_y = nuis_y),
  folds = 4,
  repeats = 3,
  eval_fold = 1,
  mode = "estimate",
  fold_allocation = "independence",
  aggregate_panels  = mean_estimate,
  aggregate_repeats = mean_estimate
)

m_mean <- create_method(
  target = target_mean,
  list_nuisance = list(nuis_y = nuis_y),
  folds = 4,
  repeats = 3,
  eval_fold = 1,
  mode = "estimate",
  fold_allocation = "overlap",
  aggregate_panels  = mean_estimate,
  aggregate_repeats = mean_estimate
)

cf_multi <- crossfit_multi(
  data    = data,
  methods = list(mse = m_mse, mean = m_mean),
  aggregate_panels  = mean_estimate,
  aggregate_repeats = mean_estimate
)

cf_multi$estimates
```

The two methods share the fitted nuisance models whenever their
structure and training folds coincide, which can save a lot of
computation when you compare multiple learners or targets.

## Predict mode: cross-fitted predictor

In `"predict"` mode, the engine returns a **prediction function**
instead of a numeric estimate. This is useful if you want a cross-fitted
predictor you can re-use on new data.

Here we build a cross-fitted regression function:

``` r
library(crossfit)

set.seed(1)

# Toy nonlinear regression problem
n  <- 200
x  <- runif(n, -2, 2)
y  <- sin(x) + rnorm(n, sd = 0.3)
data <- data.frame(x = x, y = y)

# Two simple nuisances: linear and quadratic regressions
nuis_lin <- create_nuisance(
  fit = function(data, ...) lm(y ~ x, data = data),
  predict = function(model, data, ...) {
    as.numeric(predict(model, newdata = data))
  },
  train_fold = 2
)

nuis_quad <- create_nuisance(
  fit = function(data, ...) lm(y ~ poly(x, 2), data = data),
  predict = function(model, data, ...) {
    as.numeric(predict(model, newdata = data))
  },
  train_fold = 2
)

# Target in "predict" mode: ensemble of the two nuisances
target_ensemble <- function(data, m_lin, m_quad, ...) {
  0.5 * m_lin + 0.5 * m_quad
}

method_ens <- create_method(
  target        = target_ensemble,
  list_nuisance = list(m_lin  = nuis_lin,
                       m_quad = nuis_quad),
  folds         = 4,
  repeats       = 3,
  eval_fold     = 0, # no eval window in predict mode
  mode          = "predict",
  fold_allocation = "independence"
)

res <- crossfit_multi(
  data    = data,
  methods = list(ensemble = method_ens),
  aggregate_panels  = mean_predictor,
  aggregate_repeats = mean_predictor
)

# Cross-fitted ensemble predictor on new data
f_hat <- res$estimates$ensemble
newdata <- data.frame(x = seq(-2, 2, length.out = 5))
cbind(x = newdata$x, y_hat = f_hat(newdata))
```

Here:

- Each repetition builds cross-fitted predictors,
- `mean_predictor()` aggregates the list of predictors into a single
  ensemble,
- `f_hat(newdata)` gives cross-fitted predictions on future data.

## Key functions

- `create_nuisance()`  
  Define a nuisance node via `fit` / `predict`, `train_fold`, and
  optional dependency mappings (`fit_deps`, `pred_deps`).

- `create_method()`  
  Define a method:

  - `target` function,
  - nuisance list,
  - `folds`, `repeats`,
  - `mode` (`"estimate"` or `"predict"`),
  - `eval_fold`,
  - `fold_allocation`,
  - optional `aggregate_panels`, `aggregate_repeats`.

- `crossfit()`  
  Run cross-fitting for a **single** method.

- `crossfit_multi()`  
  Run cross-fitting for **several** methods in parallel, with shared
  nuisances and shared K-fold splits.

- Aggregators:

  - `mean_estimate()`, `median_estimate()` – combine numeric panel /
    repetition results.
  - `mean_predictor()`, `median_predictor()` – combine lists of
    prediction functions when `mode = "predict"`.

## Further documentation

See:

``` r
?crossfit
?crossfit_multi
?create_method
?create_nuisance
```

You can find a more detailed introduction in the package vignette:

``` r
browseVignettes("crossfit")
# or directly:
vignette("crossfit-intro", package = "crossfit")
```

If you encounter a bug or have a feature request, please open an issue
at: <https://github.com/EtiennePeyrot/crossfit-R/issues>.

## License

`crossfit` is free software released under the GPL-3 license.
