# Tokens ------------------------------------------------------------------

#' Internal: create an evaluation token
#'
#' Constructs a token describing an evaluation context for a given method
#' and panel index. If \code{inst_key} is \code{NULL}, the token refers
#' to the synthetic \code{"__TARGET__"} node of method \code{mi}.
#'
#' @param inst_key Optional instance key; if \code{NULL}, defaults to
#'   \code{paste0("M", mi, "|__TARGET__")}.
#' @param mi Integer method index.
#' @param p Integer panel index (fold position in the cyclic schedule).
#'
#' @return A list token with fields \code{type = "eval"},
#'   \code{inst_key}, \code{mi}, \code{p}.
#'
#' @keywords internal
make_eval_token = function(inst_key = NULL, mi, p) {
  if (is.null(inst_key)) inst_key = paste0("M", mi, "|__TARGET__")
  list(type = "eval", inst_key = inst_key, mi = as.integer(mi), p = as.integer(p))
}

#' Internal: create a training token
#'
#' Constructs a token describing a training context for a given instance
#' key, method index and panel index.
#'
#' @param inst_key Instance key for the nuisance to be trained.
#' @param mi Integer method index.
#' @param p Integer panel index (fold position in the cyclic schedule).
#'
#' @return A list token with fields \code{type = "train"},
#'   \code{inst_key}, \code{mi}, \code{p}.
#'
#' @keywords internal
make_train_token = function(inst_key, mi, p) {
  list(type = "train", inst_key = inst_key, mi = as.integer(mi), p = as.integer(p))
}


# Fold selectors ----------------------------------------------------------

#' Internal: indices used for target evaluation
#'
#' Given an evaluation token and the cross-fitting plan, returns the row
#' indices and fold labels used to evaluate the target model for that
#' panel. The evaluation window has width \code{plan$eval_width[mi]} and
#' is wrapped cyclically on \code{K} folds.
#'
#' @param eval_token An evaluation token created by
#'   \code{\link{make_eval_token}}.
#' @param K Total number of folds.
#' @param fold_idx A list mapping fold labels \code{1:K} to integer row
#'   indices in \code{data}.
#' @param plan The cross-fitting plan as returned by
#'   \code{build_instances()}.
#'
#' @return A list with components \code{idx} (row indices) and
#'   \code{folds} (fold labels).
#'
#' @keywords internal
obs_for_eval = function(eval_token, K, fold_idx, plan) {
  mi = eval_token$mi
  w_eval = plan$eval_width[mi]
  folds = if (w_eval == 0L) seq_len(K) else idx_mod(eval_token$p + seq_len(w_eval) - 1L, K)

  list(idx = unlist(fold_idx[folds], use.names = FALSE), folds = folds)
}

#' Internal: indices used for nuisance training
#'
#' Given a training token and the cross-fitting plan, returns the row
#' indices and fold labels used to train a given nuisance instance. The
#' training window is determined by the instance-specific offset and
#' \code{train_fold} value.
#'
#' @param train_token A training token created by
#'   \code{\link{make_train_token}}.
#' @param K Total number of folds.
#' @param fold_idx A list mapping fold labels \code{1:K} to integer row
#'   indices.
#' @param plan The cross-fitting plan as returned by
#'   \code{build_instances()}.
#' @param methods The methods list used to build \code{plan}.
#'
#' @return A list with components \code{idx} (row indices) and
#'   \code{folds} (fold labels).
#'
#' @keywords internal
obs_for_train = function(train_token, K, fold_idx, plan, methods) {
  mi = train_token$mi
  inst_key = train_token$inst_key
  inst = plan$instances[[inst_key]]

  if (inst$method_idx != mi)
    stop("Method index mismatch between TRAIN instance and eval method index")

  if (identical(inst$name, "__TARGET__")) {
    f = idx_mod(train_token$p, K)
    return(list(idx = fold_idx[[f]], folds = f))
  }

  offset = plan$inst_offset[[inst_key]]
  w = methods[[mi]]$nuisance[[inst$name]]$train_fold
  folds = idx_mod(train_token$p + offset + seq_len(w) - 1L, K)
  list(idx = unlist(fold_idx[folds], use.names = FALSE), folds = folds)
}


# Caches ------------------------------------------------------------------

#' Internal: get a model from the cache
#'
#' Simple environment-based cache lookup used to store fitted models
#' keyed by their model signature.
#'
#' @param cache_env Environment acting as a hash table for models.
#' @param key Character key, typically produced by
#'   \code{\link{model_signature}}.
#'
#' @return The cached model object or \code{NULL} if absent.
#'
#' @keywords internal
model_cache_get = function(cache_env, key) cache_env[[key]]

#' Internal: set a model in the cache
#'
#' Stores a fitted model in an environment-based cache under the given
#' key.
#'
#' @param cache_env Environment acting as a hash table for models.
#' @param key Character key, typically produced by
#'   \code{\link{model_signature}}.
#' @param value Fitted model object to store.
#'
#' @return Invisibly returns \code{NULL}.
#'
#' @keywords internal
model_cache_set = function(cache_env, key, value) {
  cache_env[[key]] = value
  invisible(NULL)
}


# Core compute ------------------------------------------------------------

#' Internal: ensure a fitted model exists for an instance and panel
#'
#' For a given instance and token, this function either retrieves a
#' cached fitted model (based on its model signature) or fits a new one
#' using the appropriate training folds, then optionally caches it.
#'
#' Child nuisances are recursively predicted on the training window and
#' passed as additional arguments into the instance's \code{fit()}
#' function.
#'
#' Structural failures are recorded in \code{plan$fail_env} so that
#' methods relying on the same structural model can be skipped for the
#' current repetition.
#'
#' @param inst_key Instance key in \code{plan$instances}.
#' @param token An evaluation or training token (see
#'   \code{\link{make_eval_token}} and \code{\link{make_train_token}}).
#' @param data Training data (matrix or data frame).
#' @param methods Methods list used to build \code{plan}.
#' @param plan Cross-fitting plan as returned by
#'   \code{build_instances()}.
#' @param fit_child_map Child map for \code{fit()} dependencies (from
#'   \code{\link{build_child_maps}}).
#' @param pred_child_map Child map for \code{predict()} dependencies.
#' @param K Number of folds.
#' @param fold_idx List mapping folds \code{1:K} to row indices.
#' @param model_cache Environment used to store fitted models.
#'
#' @return A fitted model object for the given instance and token.
#'
#' @keywords internal
ensure_model = function(inst_key, token, data, methods, plan, fit_child_map, pred_child_map,
                        K, fold_idx, model_cache) {

  eval_token = if (identical(token$type, "eval")) token else make_eval_token(inst_key, token$mi, token$p)

  # Model cache key: structural signature + training folds for this eval panel.
  mkey = model_signature(inst_key, eval_token, methods, plan, K)
  mdl = model_cache_get(model_cache, mkey)
  if (!is.null(mdl)) {
    if (show_log) print("used cached model!")
    return(mdl)
  }

  train_token = make_train_token(inst_key, mi = token$mi, p = token$p)

  # Gather fitted-child predictions as inputs to this fit.
  fc = fit_child_map[[inst_key]]
  deps_fit = lapply(
    fc,
    predict_instance_for_token,
    token = train_token, data = data,
    methods = methods, plan = plan,
    fit_child_map = fit_child_map,
    pred_child_map = pred_child_map, K = K,
    fold_idx = fold_idx, model_cache = model_cache,
    mode = "estimate"
  )

  inst = plan$instances[[inst_key]]
  fit_fun = methods[[inst$method_idx]]$nuisance[[inst$name]]$fit
  tr_obs = obs_for_train(train_token, K, fold_idx, plan, methods)
  if (show_log) log_train(inst, token, tr_obs$folds)

  mdl = try(
    pass_named(fit_fun, c(list(data = data[tr_obs$idx, , drop = FALSE]), deps_fit)),
    silent = TRUE
  )
  if (inherits(mdl, "try-error")) {
    # Mark this structural model as failed so methods using it can be pruned.
    sig = inst$struct_sig
    if (!is.null(plan$fail_env)) plan$fail_env[[sig]] = TRUE
    stop(mdl)
  }

  if (isTRUE(inst$cache_model)) model_cache_set(model_cache, mkey, mdl)
  mdl
}

#' Internal: predict nuisance values for an instance and token
#'
#' Recursively produces predictions for a given instance under a given
#' token. This ensures the corresponding model is fitted (via
#' \code{\link{ensure_model}}), gathers predictions from child nuisances
#' as additional inputs, and then calls the instance's \code{predict()}
#' function.
#'
#' In \code{mode = "predict"}, this returns a \emph{layered} prediction
#' function on newdata, suitable for building cross-fitted predictors
#' for the target. In \code{mode = "estimate"}, it returns the
#' prediction values on the appropriate folds (eval or train).
#'
#' @param inst_key Instance key in \code{plan$instances}.
#' @param token An evaluation or training token.
#' @param data Training data (matrix or data frame).
#' @param methods Methods list.
#' @param plan Cross-fitting plan.
#' @param fit_child_map Child map for \code{fit()} dependencies.
#' @param pred_child_map Child map for \code{predict()} dependencies.
#' @param K Number of folds.
#' @param fold_idx List mapping folds to row indices.
#' @param model_cache Environment used to store fitted models.
#' @param mode Either \code{"estimate"} or \code{"predict"} (method
#'   mode).
#'
#' @return In \code{mode = "predict"}, a prediction function of
#'   \code{newdata}. In \code{mode = "estimate"}, a vector or matrix of
#'   predictions on the relevant subset of \code{data}.
#'
#' @keywords internal
predict_instance_for_token = function(inst_key, token, data, methods, plan, fit_child_map,
                                      pred_child_map, K, fold_idx, model_cache, mode) {

  mdl = ensure_model(inst_key, token, data, methods, plan, fit_child_map, pred_child_map,
                     K, fold_idx, model_cache)

  token_child = if (identical(token$type, "eval"))
    make_eval_token(inst_key, token$mi, token$p) else token

  # Predictions from child nuisances serve as additional inputs.
  pc = pred_child_map[[inst_key]]
  deps_pred = lapply(
    pc,
    predict_instance_for_token,
    token = token_child, data = data,
    methods = methods, plan = plan,
    fit_child_map = fit_child_map,
    pred_child_map = pred_child_map, K = K,
    fold_idx = fold_idx, model_cache = model_cache,
    mode = mode
  )

  inst = plan$instances[[inst_key]]
  pred_fun = methods[[inst$method_idx]]$nuisance[[inst$name]]$predict

  if (identical(mode, "predict"))
    return(layer(pred = pred_fun, model = mdl, pred_deps_predict = deps_pred))

  obs = if (identical(token$type, "eval"))
    obs_for_eval(token, K, fold_idx, plan) else
      obs_for_train(token, K, fold_idx, plan, methods)
  if (show_log) log_pred(inst, token, obs$folds, plan$instances)

  pass_named(pred_fun, c(list(model = mdl, data = data[obs$idx, , drop = FALSE]), deps_pred))
}


# Orchestrator ------------------------------------------------------------

#' Cross-fitting for multiple methods
#'
#' Runs cross-fitting for one or more methods defined via
#' \code{\link{create_method}} and \code{\link{create_nuisance}}. This
#' is the main engine that:
#' \itemize{
#'   \item validates and normalizes method specifications,
#'   \item builds the global instance graph and fold geometry,
#'   \item repeatedly draws K-fold splits and evaluates all active
#'         methods,
#'   \item aggregates results across panels and repetitions.
#' }
#'
#' Each method can operate in either \code{mode = "estimate"} (target
#' returns numeric values) or \code{mode = "predict"} (target returns a
#' prediction function). Cross-fitting ensures that nuisance models are
#' always trained on folds disjoint from the folds on which their
#' predictions are used in the target.
#'
#' @param data Data frame or matrix of size \eqn{n \times p} containing
#'   the observations.
#' @param methods A (named) list of method specifications, typically
#'   created with \code{\link{create_method}}.
#' @param fold_split A function of the form \code{function(data, K)}
#'   returning a vector of length \code{nrow(data)} with integer fold
#'   labels in \code{1:K}. It must assign at least one observation to
#'   each fold.
#' @param seed Integer base random seed used for the K-fold splits; each
#'   repetition uses \code{seed + rep_id - 1}.
#' @param aggregate_panels Function used as the \emph{default} aggregator
#'   over panels (folds) for each method. It is applied to the list of
#'   per-panel values. Methods can override this via their own
#'   \code{aggregate_panels}.
#' @param aggregate_repeats Function used as the \emph{default}
#'   aggregator over repetitions for each method. It is applied to the
#'   list of per-repetition aggregated values. Methods can override this
#'   via their own \code{aggregate_repeats}.
#' @param max_fail Non-negative integer or \code{Inf} controlling how
#'   many repetitions a method is allowed to fail before being disabled.
#'   Structural model failures and panel-level errors both count toward
#'   this limit.
#' @param verbose Logical; if \code{TRUE}, prints a compact status line
#'   per repetition.
#'
#' @return A list with components:
#'   \describe{
#'     \item{\code{estimates}}{Named list of final estimates per method
#'           (after aggregating over panels and repetitions).}
#'     \item{\code{per_method}}{For each method, a list with
#'           \code{values} (per-repetition aggregated results) and
#'           \code{errors} (error traces).}
#'     \item{\code{repeats_done}}{Number of repetitions successfully
#'           completed for each method.}
#'     \item{\code{K}}{Number of folds used in the plan.}
#'     \item{\code{K_required}}{Per-method minimal required K based on
#'           their dependency structure.}
#'     \item{\code{methods}}{The validated and normalized method
#'           specifications.}
#'     \item{\code{plan}}{The cross-fitting plan produced by
#'           \code{build_instances()}.}
#'   }
#'
#' @importFrom stats setNames
#' @importFrom utils flush.console
#'
#' @export
#' @examples
#' set.seed(1)
#' n <- 100
#' x <- rnorm(n)
#' y <- x + rnorm(n)
#'
#' data <- data.frame(x = x, y = y)
#'
#' # Shared nuisance: E[Y | X]
#' nuis_y <- create_nuisance(
#'   fit = function(data, ...) lm(y ~ x, data = data),
#'   predict = function(model, data, ...) predict(model, newdata = data)
#' )
#'
#' # Method 1: MSE of nuisance predictor
#' target_mse <- function(data, nuis_y, ...) {
#'   mean((data$y - nuis_y)^2)
#' }
#'
#' # Method 2: mean fitted value
#' target_mean <- function(data, nuis_y, ...) {
#'   mean(nuis_y)
#' }
#'
#' m1 <- create_method(
#'   target = target_mse,
#'   list_nuisance = list(nuis_y = nuis_y),
#'   folds = 2,
#'   repeats = 2,
#'   eval_fold = 1L,
#'   mode = "estimate",
#'   fold_allocation = "independence"
#' )
#'
#' m2 <- create_method(
#'   target = target_mean,
#'   list_nuisance = list(nuis_y = nuis_y),
#'   folds = 2,
#'   repeats = 2,
#'   eval_fold = 1L,
#'   mode = "estimate",
#'   fold_allocation = "overlap"
#' )
#'
#' cf_multi <- crossfit_multi(
#'   data    = data,
#'   methods = list(mse = m1, mean = m2),
#'   aggregate_panels  = mean_estimate,
#'   aggregate_repeats = mean_estimate
#' )
#'
#' cf_multi$estimates
crossfit_multi = function(
    data,
    methods,
    fold_split = function(data, K) sample(rep_len(1:K, nrow(data))),
    seed = NULL,
    aggregate_panels  = identity,
    aggregate_repeats = identity,
    max_fail = Inf,
    verbose = FALSE
) {

  if (!(is.infinite(max_fail) && max_fail > 0) && !is.int(max_fail))
    stop("'max_fail' must be a non-negative integer or Inf")
  if (!(is.null(seed) || (is.numeric(seed) && length(seed) == 1L && is.finite(seed) && seed == as.integer(seed))))
    stop("'seed' must be either NULL or an integer")


  if (!is.function(aggregate_panels))
    stop("'aggregate_panels' must be a function")
  if (!is.function(aggregate_repeats))
    stop("'aggregate_repeats' must be a function")

  # validate and build the global instance graph (also fixes per-method K).
  methods = validate_batch(methods, default_agg_panels = aggregate_panels,
                           default_agg_repeats = aggregate_repeats)
  plan = build_instances(methods)
  # Track structural model failures across repetitions.
  plan$fail_env = new.env(parent = emptyenv())
  methods = plan$methods
  K = plan$K
  K_required = plan$K_required

  n = nrow(data)
  if (K > n)
    stop(sprintf("fold count K must be <= number of observations n; got K=%d > n=%d", K, n))

  idx_map = setNames(seq_along(methods), names(methods))
  R_need = sapply(methods, `[[`, "repeats")
  rep_done = setNames(integer(length(methods)), names(methods))
  fail_count = setNames(integer(length(methods)), names(methods))

  est_per_method = lapply(methods, function(x) list(values = list(), errors = list()))
  active = rep(TRUE, length(methods))
  rep_id = 0L

  while (any(active)) {
    # Reset structural failure registry for this repetition
    plan$fail_env = new.env(parent = emptyenv())

    rep_id = rep_id + 1L

    # Random K-fold split for this repetition (shared across methods).
    if (!is.function(fold_split))
      stop("'fold_split' must be a function")
    if (!is.null(seed)) set.seed(seed + rep_id - 1L)
    fids = fold_split(data, K)
    if (length(fids) != n)
      stop("'fold_split' must return a vector of length nrow(data)")
    if (!isTRUE(all(vapply(fids, is.int, logical(1)) & fids %in% 1:K)))
      stop("'fold_split' must return integer fold labels in {1, ..., K}")
    fids = as.integer(fids)
    if (length(unique(fids)) != K)
      stop("'fold_split' must assign at least one observation to each fold (no empty folds allowed)")
    fold_idx = split(seq_len(n), factor(fids, levels = 1:K))

    # Per-repetition model cache (models are not reused across repetitions).
    model_cache  = new.env(parent = emptyenv())

    # Evaluate each active method on all panels.
    for (mi in which(active)) {
      m_name = names(methods)[mi]
      mode_m = methods[[mi]]$mode

      # If any structural nuisance has already failed in this repetition,
      # skip this method for the current repetition only
      m_structs = plan$method_structs[[m_name]]
      fail_sig = intersect(m_structs, ls(plan$fail_env))
      if (length(fail_sig)) {
        met_insts = plan$instances[plan$method_inst_keys[[m_name]]]
        inst_sigs = vapply(met_insts, `[[`, character(1), "struct_sig")
        nf_fail = vapply(met_insts[inst_sigs %in% fail_sig], `[[`, character(1), "name")

        err_l = length(est_per_method[[m_name]]$errors)
        est_per_method[[m_name]]$errors[[err_l + 1L]] = list(
          structural_failure = sprintf(
            "Method '%s' skipped in repetition %d because it uses nuisance(s) whose structural model failed to fit in this repetition: %s",
            m_name, rep_id,
            paste(unique(nf_fail), collapse = ", ")
          )
        )
        fail_count[[m_name]] = fail_count[[m_name]] + 1L
        if (fail_count[[m_name]] > max_fail) active[idx_map[m_name]] = FALSE
        next
      }

      tgt_key = plan$roots[[mi]]

      panel_vals = vector("list", K)
      panel_errs = vector("list", K)

      for (p in 1:K) {
        eval_token = make_eval_token(mi = mi, p = p)
        val = try(
          predict_instance_for_token(
            tgt_key, eval_token, data = data, methods = methods, plan = plan,
            fit_child_map = plan$child_maps$fit,
            pred_child_map = plan$child_maps$pred, K = K,
            fold_idx = fold_idx, model_cache = model_cache, mode = mode_m
          ),
          silent = TRUE
        )
        if (inherits(val, "try-error")) {
          panel_errs[[p]] = as.character(val)
          panel_vals[[p]] = NA_real_
          break
        } else {
          panel_vals[[p]] = val
        }
      }

      # If any panel threw an error, keep the error traces.
      has_err = any(!vapply(panel_errs, is.null, logical(1)))
      if (has_err) {
        err_l = length(est_per_method[[m_name]]$errors)
        est_per_method[[m_name]]$errors[[err_l + 1L]] = panel_errs
        fail_count[[m_name]] = fail_count[[m_name]] + 1L
        if (fail_count[[m_name]] > max_fail) active[idx_map[m_name]] = FALSE
      } else {
        # Otherwise aggregate over panels and record this repetition.
        agg_fun_panels = methods[[mi]]$aggregate_panels
        agg = agg_fun_panels(panel_vals)
        est_l = length(est_per_method[[m_name]]$values)
        est_per_method[[m_name]]$values[[est_l + 1L]] = agg

        rep_done[[m_name]] = rep_done[[m_name]] + 1L
        if (rep_done[[m_name]] >= R_need[m_name]) active[idx_map[m_name]] = FALSE
      }
    }

    if (verbose) {
      msg = sprintf(
        "\r>>> repetition %d [ %s ]",
        rep_id,
        paste0(names(methods), " (", rep_done, "/", R_need, ")", collapse = ", ")
      )
      cat(msg, if (!any(active)) "\n")
      flush.console()
    }
  }

  # Final aggregation across repetitions for each method (method-specific).
  final = lapply(names(methods), function(nm) {
    agg_fun_repeats = methods[[nm]]$aggregate_repeats
    agg_fun_repeats(est_per_method[[nm]]$values)
  })
  names(final) = names(methods)

  list(
    estimates    = final,
    per_method   = est_per_method,
    repeats_done = rep_done,
    K            = K,
    K_required   = K_required,
    methods      = methods,
    plan         = plan
  )
}

# Convenience wrapper -----------------------------------------------------

#' Cross-fitting for a single method
#'
#' Convenience wrapper around \code{\link{crossfit_multi}} for the
#' common case of a single method. It enforces that \code{method} is a
#' single method specification and forwards the aggregation functions
#' stored inside \code{method}.
#'
#' @param data Data frame or matrix with the observations.
#' @param method A single method specification (list) created by
#'   \code{\link{create_method}}. It must contain a \code{target}
#'   function and \code{aggregate_panels} / \code{aggregate_repeats}
#'   must be functions.
#' @param fold_split A function producing a K-fold split of the data
#'   (see \code{\link{crossfit_multi}}).
#' @param seed Integer base random seed.
#' @param max_fail Non-negative integer or \code{Inf} controlling how
#'   many repetitions the method may fail before being disabled.
#' @param verbose Logical; if \code{TRUE}, prints a compact status line
#'   per repetition.
#'
#' @return The same structure as \code{\link{crossfit_multi}}, but with
#'   a single method named \code{"method"}. The final estimate is in
#'   \code{$estimates$method}.
#'
#' @export
#' @examples
#' set.seed(1)
#' n <- 100
#' x <- rnorm(n)
#' y <- x + rnorm(n)
#'
#' data <- data.frame(x = x, y = y)
#'
#' # Nuisance: E[Y | X]
#' nuis_y <- create_nuisance(
#'   fit = function(data, ...) lm(y ~ x, data = data),
#'   predict = function(model, data, ...) predict(model, newdata = data)
#' )
#'
#' # Target: mean squared error of the nuisance predictor
#' target_mse <- function(data, nuis_y, ...) {
#'   mean((data$y - nuis_y)^2)
#' }
#'
#' method <- create_method(
#'   target = target_mse,
#'   list_nuisance = list(nuis_y = nuis_y),
#'   folds = 2,
#'   repeats = 2,
#'   eval_fold = 1L,
#'   mode = "estimate",
#'   fold_allocation = "independence",
#'   aggregate_panels  = mean_estimate,
#'   aggregate_repeats = mean_estimate
#' )
#'
#' cf <- crossfit(data, method)
#' cf$estimates
crossfit = function(data, method,
                    fold_split = function(data, K) sample(rep_len(1:K, nrow(data))),
                    seed = NULL, max_fail = Inf, verbose = FALSE) {

  # Reject lists of methods: ask user to call crossfit_multi() directly.
  if (!is.list(method) || is.null(method$target) || !is.function(method$target)) {
    if (is.list(method) && length(method) > 0L &&
        all(vapply(method, function(m)
          is.list(m) && !is.null(m$target) && is.function(m$target),
          logical(1)))) {
      stop("'method' appears to be a list of methods; use crossfit_multi() instead.")
    }
    stop("'method' must be a single method specification (list with a 'target' function).")
  }

  ap = method$aggregate_panels
  ar = method$aggregate_repeats

  if (is.null(ap) || !is.function(ap))
    stop("'method$aggregate_panels' must be a function for crossfit(); see identity/mean_estimate/mean_predictor for examples.")
  if (is.null(ar) || !is.function(ar))
    stop("'method$aggregate_repeats' must be a function for crossfit(); see identity/median_estimate/median_predictor for examples.")

  crossfit_multi(
    data    = data,
    methods = list(method),
    fold_split = fold_split,
    seed    = seed,
    aggregate_panels  = ap,
    aggregate_repeats = ar,
    max_fail = max_fail,
    verbose  = verbose
  )
}
