test_that("failure controls are complete and validated", {
  control <- crossfit_failure_control()

  expect_named(
    control,
    c(
      "fail_repetition_on_error",
      "prune_on_shared_fit_failure",
      "max_failed_repetitions"
    )
  )
  expect_true(control$fail_repetition_on_error)
  expect_false(control$prune_on_shared_fit_failure)
  expect_identical(control$max_failed_repetitions, Inf)

  expect_error(
    crossfit_failure_control(fail_repetition_on_error = NA),
    "must be TRUE or FALSE",
    fixed = TRUE
  )
  expect_error(
    crossfit_failure_control(max_failed_repetitions = -1L),
    "must be a non-negative integer or Inf",
    fixed = TRUE
  )
  expect_error(
    crossfit_failure_control(
      fail_repetition_on_error = FALSE,
      prune_on_shared_fit_failure = TRUE
    ),
    "requires",
    fixed = TRUE
  )
})

test_that("create_method accepts and completes a partial failure-control list", {
  method <- create_method(
    target = function(data, ...) mean(data$x),
    folds = 2L,
    repeats = 1L,
    mode = "estimate",
    eval_fold = 1L,
    fold_allocation = "independence",
    aggregate_panels = mean_estimate,
    aggregate_repeats = mean_estimate,
    failure_control = list(fail_repetition_on_error = FALSE)
  )

  expect_false(method$failure_control$fail_repetition_on_error)
  expect_false(method$failure_control$prune_on_shared_fit_failure)
  expect_identical(method$failure_control$max_failed_repetitions, Inf)

  expect_error(
    create_method(
      target = function(data, ...) mean(data$x),
      folds = 2L,
      repeats = 1L,
      mode = "estimate",
      eval_fold = 1L,
      fold_allocation = "independence",
      failure_control = list(unknown_option = TRUE)
    ),
    "unknown 'failure_control' entries",
    fixed = TRUE
  )
})

test_that("a missingness-aware panel aggregator can retain a repetition", {
  K <- 3L
  dat <- data.frame(fold = seq_len(K), x = seq_len(K))
  calls <- new.env(parent = emptyenv())
  calls$n <- 0L

  nuisance <- create_nuisance(
    fit = function(data, ...) {
      calls$n <- calls$n + 1L
      if (calls$n == 1L) stop("transient fit failure")
      mean(data$x)
    },
    predict = function(model, data, ...) rep(model, nrow(data))
  )
  available_mean <- function(xs) mean(unlist(xs), na.rm = TRUE)

  method <- create_method(
    target = function(data, nuisance, ...) mean(nuisance),
    list_nuisance = list(nuisance = nuisance),
    folds = K,
    repeats = 1L,
    mode = "estimate",
    eval_fold = 1L,
    fold_allocation = "independence",
    aggregate_panels = available_mean,
    aggregate_repeats = mean_estimate,
    failure_control = crossfit_failure_control(
      fail_repetition_on_error = FALSE
    )
  )

  result <- crossfit(
    dat,
    method,
    fold_split = function(data, K) as.integer(data$fold)
  )

  expect_true(is.finite(result$estimate))
  expect_equal(result$repeats_done, 1L)
  expect_equal(calls$n, K)
  expect_length(result$results$errors, 1L)
  expect_match(result$results$errors[[1L]][[1L]], "transient fit failure")
})

test_that("failed-fit pruning is opt-in", {
  K <- 3L
  dat <- data.frame(fold = seq_len(K), x = seq_len(K))
  calls <- new.env(parent = emptyenv())
  calls$n <- 0L

  shared_nuisance <- create_nuisance(
    fit = function(data, ...) {
      calls$n <- calls$n + 1L
      if (calls$n == 1L) stop("first call fails")
      mean(data$x)
    },
    predict = function(model, data, ...) rep(model, nrow(data))
  )
  target <- function(data, shared_nuisance, ...) mean(shared_nuisance)
  make_method <- function(control) create_method(
    target = target,
    list_nuisance = list(shared_nuisance = shared_nuisance),
    folds = K,
    repeats = 1L,
    mode = "estimate",
    eval_fold = 1L,
    fold_allocation = "independence",
    aggregate_panels = mean_estimate,
    aggregate_repeats = function(xs) if (length(xs)) mean_estimate(xs) else NA_real_,
    failure_control = control
  )

  first <- make_method(crossfit_failure_control(max_failed_repetitions = 0L))
  second <- make_method(crossfit_failure_control())
  result <- crossfit_multi(
    dat,
    methods = list(first = first, second = second),
    fold_split = function(data, K) as.integer(data$fold)
  )

  expect_equal(result$repeats_done[["first"]], 0L)
  expect_equal(result$repeats_done[["second"]], 1L)
  expect_true(is.finite(result$estimates$second))
  expect_equal(calls$n, K + 1L)
})

test_that("opt-in pruning skips a shared structural fit failure", {
  K <- 3L
  dat <- data.frame(fold = seq_len(K), x = seq_len(K))
  calls <- new.env(parent = emptyenv())
  calls$n <- 0L

  shared_nuisance <- create_nuisance(
    fit = function(data, ...) {
      calls$n <- calls$n + 1L
      stop("deterministic failure")
    },
    predict = function(model, data, ...) rep(model, nrow(data))
  )
  target <- function(data, shared_nuisance, ...) mean(shared_nuisance)
  safe_mean <- function(xs) if (length(xs)) mean_estimate(xs) else NA_real_
  make_method <- function(prune) create_method(
    target = target,
    list_nuisance = list(shared_nuisance = shared_nuisance),
    folds = K,
    repeats = 1L,
    mode = "estimate",
    eval_fold = 1L,
    fold_allocation = "independence",
    aggregate_panels = mean_estimate,
    aggregate_repeats = safe_mean,
    failure_control = crossfit_failure_control(
      prune_on_shared_fit_failure = prune,
      max_failed_repetitions = 0L
    )
  )

  result <- crossfit_multi(
    dat,
    methods = list(first = make_method(FALSE), second = make_method(TRUE)),
    fold_split = function(data, K) as.integer(data$fold)
  )

  expect_equal(calls$n, 1L)
  expect_equal(unname(result$repeats_done), c(0L, 0L))
  expect_match(
    result$per_method$second$errors[[1L]]$structural_failure,
    "skipped in repetition",
    fixed = TRUE
  )
})

test_that("closures with different captured state are not shared", {
  K <- 2L
  dat <- data.frame(fold = seq_len(K), x = seq_len(K))
  counter_one <- new.env(parent = emptyenv())
  counter_two <- new.env(parent = emptyenv())
  counter_one$n <- 0L
  counter_two$n <- 0L

  make_nuisance <- function(value, counter) create_nuisance(
    fit = function(data, ...) {
      counter$n <- counter$n + 1L
      value
    },
    predict = function(model, data, ...) rep(model, nrow(data))
  )
  target <- function(data, nuisance, ...) mean(nuisance)
  make_method <- function(nuisance) create_method(
    target = target,
    list_nuisance = list(nuisance = nuisance),
    folds = K,
    repeats = 1L,
    mode = "estimate",
    eval_fold = 1L,
    fold_allocation = "independence",
    aggregate_panels = mean_estimate,
    aggregate_repeats = mean_estimate
  )

  result <- crossfit_multi(
    dat,
    methods = list(
      one = make_method(make_nuisance(1, counter_one)),
      two = make_method(make_nuisance(2, counter_two))
    ),
    fold_split = function(data, K) as.integer(data$fold)
  )

  expect_equal(result$estimates$one, 1)
  expect_equal(result$estimates$two, 2)
  expect_equal(counter_one$n, K)
  expect_equal(counter_two$n, K)
})
