# tests/testthat/test-reserved-nuisance-names.R

test_that("reserved nuisance names are rejected", {
  nuis <- create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) as.numeric(predict(model, newdata = data))
  )

  # Use a target that requires no nuisance args (only 'data' and '...')
  # so we don't introduce any other validation failures.
  target <- function(data, ...) mean(data$y)

  expect_error(
    create_method(
      target = target,
      list_nuisance = list(data = nuis),   # reserved name
      folds = 2L,
      repeats = 1L,
      eval_fold = 1L,
      mode = "estimate",
      fold_allocation = "independence"
    ),
    "reserved",
    fixed = FALSE
  )

  expect_error(
    create_method(
      target = target,
      list_nuisance = list("__TARGET__" = nuis),  # reserved name
      folds = 2L,
      repeats = 1L,
      eval_fold = 1L,
      mode = "estimate",
      fold_allocation = "independence"
    ),
    "reserved",
    fixed = FALSE
  )
})
