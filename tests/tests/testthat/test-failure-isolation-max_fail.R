# tests/testthat/test-failure-isolation-max_fail.R

test_that("failure isolation: one failing method does not crash others; max_fail disables it", {
  set.seed(61)
  n <- 80
  dat <- data.frame(x = rnorm(n), y = rnorm(n))

  K <- 3L

  # Safe aggregator: no warning when xs is empty
  safe_mean_estimate <- function(xs) {
    v <- unlist(xs)
    if (!length(v)) return(NA_real_)
    mean(v)
  }

  # Good method
  nuis_good <- create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) as.numeric(predict(model, newdata = data))
  )
  target_mse <- function(data, nuis_good, ...) mean((data$y - nuis_good)^2)

  m_good <- create_method(
    target = target_mse,
    list_nuisance = list(nuis_good = nuis_good),
    folds = K,
    repeats = 1L,
    eval_fold = 1L,
    mode = "estimate",
    fold_allocation = "independence",
    aggregate_panels  = mean_estimate,
    aggregate_repeats = safe_mean_estimate   # <-- important
  )

  # Bad method: nuisance fit always fails
  nuis_bad <- create_nuisance(
    fit = function(data, ...) stop("intentional failure"),
    predict = function(model, data, ...) rep(0, nrow(data))
  )
  target_mean <- function(data, nuis_bad, ...) mean(nuis_bad)

  m_bad <- create_method(
    target = target_mean,
    list_nuisance = list(nuis_bad = nuis_bad),
    folds = K,
    repeats = 2L,
    eval_fold = 1L,
    mode = "estimate",
    fold_allocation = "independence",
    aggregate_panels  = mean_estimate,
    aggregate_repeats = safe_mean_estimate   # <-- important
  )

  res <- crossfit_multi(
    data = dat,
    methods = list(good = m_good, bad = m_bad),
    seed = 1,
    max_fail = 0L,
    aggregate_panels  = mean_estimate,
    aggregate_repeats = safe_mean_estimate   # optional, but consistent
  )

  expect_true(is.numeric(res$estimates$good))
  expect_true(is.na(res$estimates$bad))

  expect_equal(res$repeats_done[["good"]], 1L)
  expect_equal(res$repeats_done[["bad"]], 0L)

  expect_true(length(res$per_method$bad$errors) >= 1L)
  expect_true(grepl("intentional failure", res$per_method$bad$errors[[1]][[1]], fixed = TRUE))
})
