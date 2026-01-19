# tests/testthat/test-mode-eval_fold-constraints.R

test_that("mode/eval_fold compatibility is enforced", {
  nuis <- make_lm_nuisance()

  target <- function(data, nuis_y, ...) mean(nuis_y)

  # predict mode => eval_fold must be 0
  expect_error(
    create_method(
      target = target,
      list_nuisance = list(nuis_y = nuis),
      folds = 2L,
      repeats = 1L,
      eval_fold = 1L,
      mode = "predict",
      fold_allocation = "independence"
    ),
    "in predict mode, 'eval_fold' must be 0",
    fixed = TRUE
  )

  # estimate mode => eval_fold must be > 0
  expect_error(
    create_method(
      target = target,
      list_nuisance = list(nuis_y = nuis),
      folds = 2L,
      repeats = 1L,
      eval_fold = 0L,
      mode = "estimate",
      fold_allocation = "independence"
    ),
    "'eval_fold' must be a positive integer when mode = 'estimate'",
    fixed = TRUE
  )
})
