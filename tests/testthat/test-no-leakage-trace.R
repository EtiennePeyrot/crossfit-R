test_that("no leakage: target never uses nuisance trained on its evaluation fold (trace-based)", {

  # We make folds explicit in the data, and we provide a deterministic fold_split().
  K <- 5L
  n_per_fold <- 3L
  dat <- data.frame(
    fold = rep(seq_len(K), each = n_per_fold),
    x = 0, y = 0
  )

  fold_split_det <- function(data, K) as.integer(data$fold)

  # Helper: extract the list of training folds from the model trace string
  extract_train_folds <- function(model_trace) {
    # model_trace looks like: "nuis_y trained_on [1,2,3]"
    inside <- sub(".*trained_on \\[([^\\]]+)\\].*", "\\1", model_trace)
    as.integer(strsplit(inside, ",", fixed = TRUE)[[1]])
  }

  # For counting target evaluations (one per panel)
  log_env <- new.env(parent = emptyenv())
  log_env$n_calls <- 0L
  log_env$msgs <- character(0)

  # Nuisance: fit() returns a string describing training folds.
  # predict() returns strings describing which model predicted on which eval fold.
  nuis_y <- create_nuisance(
    train_fold = 1L, # simplest: train on K-1 folds in standard cross-fitting
    fit = function(data, ...) {
      tr <- paste(sort(unique(data$fold)), collapse = ",")
      paste0("nuis_y trained_on [", tr, "]")
    },
    predict = function(model, data, ...) {
      ev <- paste(sort(unique(data$fold)), collapse = ",")
      rep(paste0(model, " predict_on [", ev, "]"), nrow(data))
    }
  )

  # Aggregators that work for character outputs (do not call mean())
  pick_first <- function(xs) xs[[1]]

  # Target: returns a trace string; ALSO asserts no leakage.
  target_trace <- function(data, nuis_y, ...) {
    log_env$n_calls <- log_env$n_calls + 1L

    eval_fold <- unique(data$fold)
    # With eval_fold = 1, target should see exactly one fold
    if (length(eval_fold) != 1L) {
      stop("Test expects eval window to be one fold; got: ", paste(eval_fold, collapse = ","))
    }

    # nuis_y is a vector of strings; take one representative and parse training folds
    one_pred <- nuis_y[[1]]
    # one_pred starts with the model string that contains trained_on [...]
    # e.g. "nuis_y trained_on [1,2,3,4] predict_on [5]"
    model_part <- sub("(nuis_y trained_on \\[[^\\]]+\\]).*", "\\1", one_pred)
    train_folds <- extract_train_folds(model_part)

    if (eval_fold %in% train_folds) {
      stop("Leakage detected: eval fold ", eval_fold,
           " appears in train folds [", paste(train_folds, collapse = ","), "]")
    }

    msg <- paste0(
      "target evaluated on fold [", eval_fold, "]",
      " with nuisance prediction: ", one_pred
    )
    log_env$msgs <- c(log_env$msgs, msg)
    msg
  }

  # Run the test for each allocation scheme (all must respect no leakage)
  for (alloc in c("independence", "overlap", "disjoint")) {

    log_env$n_calls <- 0L
    log_env$msgs <- character(0)

    method <- create_method(
      target = target_trace,
      list_nuisance = list(nuis_y = nuis_y),
      folds = K,
      repeats = 1L,
      eval_fold = 1L,
      mode = "estimate",
      fold_allocation = alloc,
      aggregate_panels  = pick_first,
      aggregate_repeats = pick_first
    )

    res <- crossfit_multi(
      data = dat,
      methods = list(trace = method),
      fold_split = fold_split_det,
      seed = 123,
      aggregate_panels  = pick_first,
      aggregate_repeats = pick_first
    )

    # If leakage happened, the target would have stopped with an error.
    expect_true(is.list(res))
    expect_true(is.character(res$estimates$trace))

    # target called once per panel
    expect_equal(log_env$n_calls, K)

    # Basic sanity: each msg should mention "target evaluated on fold"
    expect_true(all(grepl("target evaluated on fold", log_env$msgs, fixed = TRUE)))
  }
})
