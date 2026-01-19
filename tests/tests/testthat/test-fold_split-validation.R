# tests/testthat/test-fold_split-validation.R

test_that("fold_split() must return length nrow(data)", {
  dat <- make_regression_data(seed = 51, n = 30)
  nuis <- make_lm_nuisance()
  method <- make_simple_method(nuis_obj = nuis, folds = 3L, repeats = 1L)

  bad_fold_split <- function(data, K) as.integer(rep_len(1:K, nrow(data) - 1L))

  expect_error(
    crossfit_multi(dat, methods = list(m = method), fold_split = bad_fold_split, seed = 1),
    "must return a vector of length nrow\\(data\\)",
    fixed = FALSE
  )
})

test_that("fold_split() must return integer labels in {1,...,K}", {
  dat <- make_regression_data(seed = 52, n = 30)
  nuis <- make_lm_nuisance()
  method <- make_simple_method(nuis_obj = nuis, folds = 3L, repeats = 1L)

  bad_fold_split <- function(data, K) rep_len(seq_len(K), nrow(data)) + 0.25

  expect_error(
    crossfit_multi(dat, methods = list(m = method), fold_split = bad_fold_split, seed = 1),
    "must return integer fold labels",
    fixed = FALSE
  )
})

test_that("fold_split() must not create empty folds", {
  dat <- make_regression_data(seed = 53, n = 30)
  nuis <- make_lm_nuisance()
  method <- make_simple_method(nuis_obj = nuis, folds = 3L, repeats = 1L)

  # Only uses folds 1 and 2 => fold 3 is empty
  bad_fold_split <- function(data, K) as.integer(rep_len(1:(K - 1L), nrow(data)))

  expect_error(
    crossfit_multi(dat, methods = list(m = method), fold_split = bad_fold_split, seed = 1),
    "no empty folds",
    fixed = FALSE
  )
})
