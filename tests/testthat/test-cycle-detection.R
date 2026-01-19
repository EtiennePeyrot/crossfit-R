# tests/testthat/test-cycle-detection.R

test_that("cycle detection rejects cyclic nuisance dependencies", {
  dat <- make_regression_data(seed = 21, n = 30)

  nuis_a <- create_nuisance(
    fit = function(data, nuis_b, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) as.numeric(predict(model, newdata = data))
  )

  nuis_b <- create_nuisance(
    fit = function(data, nuis_a, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) as.numeric(predict(model, newdata = data))
  )

  target <- function(data, nuis_a, ...) mean(nuis_a)

  expect_error(
    create_method(
      target = target,
      list_nuisance = list(nuis_a = nuis_a, nuis_b = nuis_b),
      folds = 2L,
      repeats = 1L,
      eval_fold = 1L,
      mode = "estimate",
      fold_allocation = "independence"
    ),
    "Cycle detected",
    fixed = FALSE
  )
})
