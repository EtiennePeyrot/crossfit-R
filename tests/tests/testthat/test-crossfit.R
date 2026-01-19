test_that("create_nuisance builds a valid nuisance spec", {
  set.seed(1)
  n <- 20
  x <- rnorm(n)
  y <- x + rnorm(n)
  dat <- data.frame(x = x, y = y)

  nuis <- create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) predict(model, newdata = data)
  )

  expect_type(nuis, "list")
  expect_true(is.function(nuis$fit))
  expect_true(is.function(nuis$predict))
  expect_identical(nuis$train_fold, 1L)

  # Nuisance can be used directly
  mdl <- nuis$fit(dat)
  preds <- nuis$predict(mdl, dat)
  expect_length(preds, n)
})

test_that("create_method validates target and nuisances", {
  set.seed(1)
  n <- 20
  x <- rnorm(n)
  y <- x + rnorm(n)
  dat <- data.frame(x = x, y = y)

  nuis_y <- create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) predict(model, newdata = data)
  )

  target_mse <- function(data, nuis_y, ...) {
    mean((data$y - nuis_y)^2)
  }

  m <- create_method(
    target = target_mse,
    list_nuisance = list(nuis_y = nuis_y),
    folds = 2,
    repeats = 1,
    eval_fold = 1L,
    mode = "estimate",
    fold_allocation = "independence",
    aggregate_panels  = mean_estimate,
    aggregate_repeats = mean_estimate
  )

  expect_type(m, "list")
  expect_true(is.function(m$target))
  expect_equal(m$folds, 2)
  expect_equal(m$repeats, 1L)

  # Wrong target arg name should error
  bad_target <- function(data, something_else, ...) mean(data$y)
  expect_error(
    create_method(
      target = bad_target,
      list_nuisance = list(nuis_y = nuis_y),
      folds = 2,
      repeats = 1,
      eval_fold = 1L,
      mode = "estimate",
      fold_allocation = "independence"
    ),
    "target\\(\\): required args not in nuisance",
    fixed = FALSE
  )
})

test_that("crossfit works on a simple regression problem", {
  set.seed(1)
  n <- 100
  x <- rnorm(n)
  y <- x + rnorm(n)
  dat <- data.frame(x = x, y = y)

  nuis_y <- create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) predict(model, newdata = data)
  )

  target_mse <- function(data, nuis_y, ...) {
    mean((data$y - nuis_y)^2)
  }

  method <- create_method(
    target = target_mse,
    list_nuisance = list(nuis_y = nuis_y),
    folds = 2,
    repeats = 2,
    eval_fold = 1L,
    mode = "estimate",
    fold_allocation = "independence",
    aggregate_panels  = mean_estimate,
    aggregate_repeats = mean_estimate
  )

  res <- crossfit(dat, method)

  expect_type(res, "list")
  expect_equal(length(res$estimates), 1L)

  mname <- names(res$estimates)
  expect_true(is.character(mname))
  expect_equal(length(mname), 1L)
  expect_true(nzchar(mname))

  expect_true(is.numeric(res$estimates[[mname]]))
  expect_equal(length(res$estimates[[mname]]), 1L)

  # repetitions and K are as requested
  expect_equal(res$repeats_done[[mname]], 2L)
  expect_equal(res$K, 2L)
})

test_that("crossfit_multi supports multiple methods with shared nuisances", {
  set.seed(2)
  n <- 80
  x <- rnorm(n)
  y <- x + rnorm(n)
  dat <- data.frame(x = x, y = y)

  nuis_y <- create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) predict(model, newdata = data)
  )

  target_mse <- function(data, nuis_y, ...) {
    mean((data$y - nuis_y)^2)
  }

  target_mean <- function(data, nuis_y, ...) {
    mean(nuis_y)
  }

  m1 <- create_method(
    target = target_mse,
    list_nuisance = list(nuis_y = nuis_y),
    folds = 2,
    repeats = 2,
    eval_fold = 1L,
    mode = "estimate",
    fold_allocation = "independence"
  )

  m2 <- create_method(
    target = target_mean,
    list_nuisance = list(nuis_y = nuis_y),
    folds = 2,
    repeats = 2,
    eval_fold = 1L,
    mode = "estimate",
    fold_allocation = "overlap"
  )

  cf_multi <- crossfit_multi(
    data    = dat,
    methods = list(mse = m1, mean = m2),
    aggregate_panels  = mean_estimate,
    aggregate_repeats = mean_estimate
  )

  expect_named(cf_multi$estimates, c("mse", "mean"))
  expect_true(is.numeric(cf_multi$estimates$mse))
  expect_true(is.numeric(cf_multi$estimates$mean))
})

test_that("predict mode returns a predictor function", {
  set.seed(3)
  n <- 60
  x <- rnorm(n)
  y <- 2 * x + rnorm(n)
  dat <- data.frame(x = x, y = y)

  nuis_y <- create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) as.numeric(predict(model, newdata = data))
  )

  # Target in predict mode: just forwards the nuisance prediction
  target_pred <- function(data, nuis_y, ...) {
    nuis_y
  }

  m_pred <- create_method(
    target = target_pred,
    list_nuisance = list(nuis_y = nuis_y),
    folds = 2,
    repeats = 2,
    eval_fold = 0L,
    mode = "predict",
    fold_allocation = "independence"
  )

  res_pred <- crossfit_multi(
    data    = dat,
    methods = list(pred = m_pred),
    aggregate_panels  = mean_predictor,
    aggregate_repeats = mean_predictor
  )

  # In predict mode, estimates$pred should be a prediction function
  expect_true(is.function(res_pred$estimates$pred))

  newdata <- data.frame(x = seq(-1, 1, length.out = 5))
  vals <- res_pred$estimates$pred(newdata)
  expect_type(vals, "double")
  expect_length(vals, nrow(newdata))
})

test_that("aggregator helpers behave as expected", {
  xs <- list(c(1, 2), 3, c(4, 5))
  expect_equal(mean_estimate(xs), mean(unlist(xs)))
  expect_equal(median_estimate(xs), median(unlist(xs)))

  f1 <- function(newdata, ...) newdata$x
  f2 <- function(newdata, ...) 2 * newdata$x

  ens_mean   <- mean_predictor(list(f1, f2))
  ens_median <- median_predictor(list(f1, f2))

  nd <- data.frame(x = 1:4)

  expect_equal(ens_mean(nd), (nd$x + 2 * nd$x) / 2)
  expect_equal(ens_median(nd), (nd$x + 2 * nd$x) / 2)  # median of {x, 2x}
})
