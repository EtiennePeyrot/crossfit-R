# tests/testthat/test-dependency-coverage.R

test_that("dependency coverage: required fit() args must have a nuisance mapping", {
  nuis_bad_fit <- create_nuisance(
    fit = function(data, foo, ...) lm(y ~ x, data = data),  # foo is unmapped
    predict = function(model, data, ...) as.numeric(predict(model, newdata = data))
  )

  target <- function(data, nuis_bad_fit, ...) mean((data$y - nuis_bad_fit)^2)

  expect_error(
    create_method(
      target = target,
      list_nuisance = list(nuis_bad_fit = nuis_bad_fit),
      folds = 2L,
      repeats = 1L,
      eval_fold = 1L,
      mode = "estimate",
      fold_allocation = "independence"
    ),
    regexp = "required fit\\(\\) args without a nuisance mapping"
  )
})

test_that("dependency coverage: required predict() args must have a nuisance mapping", {
  nuis_bad_pred <- create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, bar, ...) as.numeric(predict(model, newdata = data)) # bar unmapped
  )

  target <- function(data, nuis_bad_pred, ...) mean((data$y - nuis_bad_pred)^2)

  expect_error(
    create_method(
      target = target,
      list_nuisance = list(nuis_bad_pred = nuis_bad_pred),
      folds = 2L,
      repeats = 1L,
      eval_fold = 1L,
      mode = "estimate",
      fold_allocation = "independence"
    ),
    regexp = "required predict\\(\\) args without a nuisance mapping"
  )
})
