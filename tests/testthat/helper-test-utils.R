# tests/testthat/helper-test-utils.R

make_regression_data <- function(seed = 1, n = 80, beta = 1) {
  set.seed(seed)
  x <- rnorm(n)
  y <- beta * x + rnorm(n)
  data.frame(row_id = seq_len(n), x = x, y = y)
}

make_lm_nuisance <- function(train_fold = 1L) {
  create_nuisance(
    fit = function(data, ...) lm(y ~ x, data = data),
    predict = function(model, data, ...) as.numeric(predict(model, newdata = data)),
    train_fold = as.integer(train_fold)
  )
}

make_simple_method <- function(
    nuis_name = "nuis_y",
    nuis_obj,
    folds = 2L,
    repeats = 1L,
    eval_fold = 1L,
    mode = "estimate",
    fold_allocation = "independence",
    aggregate_panels = mean_estimate,
    aggregate_repeats = mean_estimate
) {
  target_mse <- function(data, nuis_y, ...) mean((data$y - nuis_y)^2)

  create_method(
    target = target_mse,
    list_nuisance = setNames(list(nuis_obj), nuis_name),
    folds = as.integer(folds),
    repeats = as.integer(repeats),
    eval_fold = as.integer(eval_fold),
    mode = mode,
    fold_allocation = fold_allocation,
    aggregate_panels  = aggregate_panels,
    aggregate_repeats = aggregate_repeats
  )
}
