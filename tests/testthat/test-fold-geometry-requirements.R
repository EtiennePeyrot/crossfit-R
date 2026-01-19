# tests/testthat/test-fold-geometry-requirements.R

test_that("fold geometry: folds must be >= minimal_K implied by eval/train widths", {
  dat <- make_regression_data(seed = 41, n = 60)

  nuis <- make_lm_nuisance(train_fold = 2L)  # with eval_fold=1, independence => minimal_K = 3
  method <- make_simple_method(
    nuis_obj = nuis,
    folds = 2L,              # too small on purpose
    repeats = 1L,
    eval_fold = 1L,
    mode = "estimate",
    fold_allocation = "independence"
  )

  expect_error(
    crossfit_multi(
      data = dat,
      methods = list(m = method),
      seed = 1
    ),
    "folds.*smaller than required",
    fixed = FALSE
  )
})
