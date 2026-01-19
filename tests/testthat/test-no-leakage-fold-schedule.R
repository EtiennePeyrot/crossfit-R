# tests/testthat/test-no-leakage-fold-schedule.R

test_that("no leakage: nuisance training folds never overlap target evaluation folds", {
  K <- 5L
  dat <- make_regression_data(seed = 11, n = 100, beta = 1)

  # Deterministic fold assignment (no randomness)
  fids <- rep_len(seq_len(K), nrow(dat))
  fold_split_det <- function(data, K) as.integer(fids)

  # Env to store checks made inside target()
  log_env <- new.env(parent = emptyenv())
  log_env$checks <- list()

  # Nuisance stores which folds it trained on; predict attaches that info to predictions
  nuis_y <- create_nuisance(
    train_fold = 2L,
    fit = function(data, ...) {
      list(train_folds = sort(unique(fids[data$row_id])))
    },
    predict = function(model, data, ...) {
      out <- rep(0, nrow(data))
      attr(out, "train_folds") <- model$train_folds
      out
    }
  )

  # Target asserts "no leakage"
  target_check <- function(data, nuis_y, ...) {
    eval_folds  <- sort(unique(fids[data$row_id]))
    train_folds <- attr(nuis_y, "train_folds")

    ok <- length(intersect(eval_folds, train_folds)) == 0L
    log_env$checks[[length(log_env$checks) + 1L]] <- list(
      eval_folds = eval_folds,
      train_folds = train_folds,
      ok = ok
    )
    if (!ok) stop("Leakage detected: training folds overlap evaluation folds")

    0.0
  }

  for (alloc in c("independence", "overlap", "disjoint")) {
    method <- create_method(
      target = target_check,
      list_nuisance = list(nuis_y = nuis_y),
      folds = K,
      repeats = 1L,
      eval_fold = 1L,
      mode = "estimate",
      fold_allocation = alloc,
      aggregate_panels  = mean_estimate,
      aggregate_repeats = mean_estimate
    )

    expect_silent(
      crossfit_multi(
        data = dat,
        methods = list(m = method),
        fold_split = fold_split_det,
        seed = 1,
        aggregate_panels  = mean_estimate,
        aggregate_repeats = mean_estimate
      )
    )
  }

  # Target should have been evaluated K times per repetition per method.
  # We ran 3 methods (allocations) each with repeats=1 and K panels.
  expect_equal(length(log_env$checks), 3L * K)
  expect_true(all(vapply(log_env$checks, `[[`, logical(1), "ok")))
})
