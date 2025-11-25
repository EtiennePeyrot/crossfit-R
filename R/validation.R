# Validation / cycles -----------------------------------------------------

#' Internal: check for cycles in the nuisance graph
#'
#' Validates that the combined dependency graph over all nuisances is
#' acyclic. The graph edges are obtained from the union of
#' \code{fit_deps} and \code{pred_deps} for each nuisance (including the
#' synthetic \code{"__TARGET__"} node).
#'
#' If a cycle is detected, an error is thrown with a textual
#' representation of the cycle. Otherwise the function returns
#' \code{TRUE} invisibly.
#'
#' @param nfs A named list of normalized nuisance specifications. Each
#'   element must contain components \code{fit_deps} and
#'   \code{pred_deps}, as constructed by \code{validate_method()}.
#'
#' @return Invisibly returns \code{TRUE} on success, or throws an error
#'   if a cycle is found.
#'
#' @importFrom stats setNames
#'
#' @keywords internal
check_cycles = function(nfs) {
  pred_map = lapply(nfs, function(nf) nf$pred_deps)
  fit_map  = lapply(nfs, function(nf) nf$fit_deps)
  g = mapply(function(p, f) unique(c(p, f)), pred_map, fit_map, SIMPLIFY = FALSE)
  state = setNames(integer(length(g)), names(g)) # 0=unseen, 1=visiting, 2=done
  stack = character(0)
  visit = function(u) {
    if (state[[u]] == 1L) stop(sprintf(
      "Cycle detected in dependency graph (pred_deps/fit_deps): %s",
      paste(c(stack, u), collapse = " -> ")
    ))
    if (state[[u]] == 2L) return(invisible(NULL))
    state[[u]] <<- 1L
    stack <<- c(stack, u)
    for (v in g[[u]]) visit(v)
    if (length(stack)) stack <<- stack[-length(stack)]
    state[[u]] <<- 2L
    invisible(NULL)
  }
  for (u in names(g)) visit(u)
  invisible(TRUE)
}

# Shared validation helpers ----------------------------------------------

#' Internal: basic structural checks for a single nuisance
#'
#' Performs basic validation and normalization for a single nuisance
#' specification. This ensures that the nuisance:
#' \itemize{
#'   \item is a list with \code{fit} and \code{predict} functions,
#'   \item has a valid positive integer \code{train_fold},
#'   \item has \code{fit_deps} and \code{pred_deps} either \code{NULL}
#'         or named character vectors (arg \eqn{\to} nuisance).
#' }
#'
#' This function does not infer dependencies or check cycles; that is
#' done at the method level by \code{validate_method()}.
#'
#' @param nf A list representing a nuisance specification.
#' @param nm Optional nuisance name for error messages.
#' @param mname Optional method name for error messages.
#'
#' @return A normalized nuisance list with guaranteed components
#'   \code{fit}, \code{predict}, \code{train_fold},
#'   \code{fit_deps}, and \code{pred_deps}.
#'
#' @keywords internal
validate_nuisance = function(nf, nm = NULL, mname = NULL) {
  prefix = paste0(
    if (!is.null(mname)) sprintf("In method '%s': ", mname) else "",
    if (!is.null(nm)) sprintf("nuisance '%s'", nm) else "nuisance"
  )

  if (!is.list(nf))
    stop(paste(prefix, "must be a list"))
  if (is.null(nf$fit) || !is.function(nf$fit))
    stop(paste(prefix, "must provide a function 'fit'"))
  if (is.null(nf$predict) || !is.function(nf$predict))
    stop(paste(prefix, "must provide a function 'predict'"))

  # train_fold: default to 1L if missing, otherwise must be a positive integer.
  tf = nf$train_fold
  tf = if (is.null(tf)) 1L else
    if (is.int(tf) && tf > 0L) as.integer(tf) else
      stop(paste(prefix, "'train_fold' must be a positive integer"))
  nf$train_fold = tf

  # fit_deps / pred_deps, if provided, must be named character vectors
  # (coerce atomic vectors to character).
  check_dep_vec = function(vec, dep_name) {
    if (is.null(vec)) return(NULL)
    if (!is.character(vec)) vec = as.character(vec)
    nmv = names(vec)
    if (is.null(nmv))
      stop(paste(prefix, dep_name, "must be named (arg -> nuisance)"))
    badn = is.na(nmv) | !nzchar(nmv)
    if (any(badn))
      stop(paste(prefix, dep_name, "has empty arg names"))
    vec
  }
  nf$fit_deps  = check_dep_vec(nf$fit_deps,  "fit_deps")
  nf$pred_deps = check_dep_vec(nf$pred_deps, "pred_deps")

  nf
}

#' Internal: validate and normalize a single method specification
#'
#' Validates a method specification and normalizes its nuisance graph.
#' This includes:
#' \itemize{
#'   \item checking \code{mode}, \code{eval_fold}, \code{folds},
#'         and \code{repeats},
#'   \item validating the nuisance container,
#'   \item checking that the target is a function and that its required
#'         arguments are nuisances,
#'   \item attaching a synthetic \code{"__TARGET__"} nuisance node,
#'   \item inferring \code{fit_deps} and \code{pred_deps} from required
#'         arguments,
#'   \item enforcing coverage of required arguments,
#'   \item and running a cycle check on the full graph.
#' }
#'
#' This function is normally called from \code{validate_batch()} and is
#' not intended to be used directly by end users.
#'
#' @param met A method specification list.
#' @param mname Optional method name (used for error messages).
#'
#' @return A normalized method specification with a fully validated
#'   nuisance graph and an added \code{"__TARGET__"} node.
#'
#' @importFrom stats setNames
#'
#' @keywords internal
validate_method = function(met, mname = NULL) {
  prefix_m = if (!is.null(mname)) sprintf("In method '%s': ", mname) else ""
  label_n = function(nm) {
    if (is.null(nm)) return("nuisance")
    if (identical(nm, "__TARGET__")) return("target()")
    sprintf("nuisance '%s'", nm)
  }

  mode_m = met$mode
  # Optional mode sanity: must be estimate/predict if provided.
  if (!is.null(mode_m) && !mode_m %in% c("estimate", "predict"))
    stop(paste0(prefix_m, "'mode' must be either 'estimate' or 'predict'"))

  # Method-level constraints on eval_fold depend on mode.
  ef = met$eval_fold
  if (identical(mode_m, "predict") && !(is.int(ef) && ef == 0L))
    stop(paste0(prefix_m, "in predict mode, 'eval_fold' must be 0"))

  if (identical(mode_m, "estimate") && !(is.int(ef) && ef > 0L))
    stop(paste0(prefix_m, "'eval_fold' must be a positive integer when mode = 'estimate'"))

  # Basic checks on folds / repeats config.
  k = met$folds
  if (!is.null(k) && !(is.int(k) && k > 0L))
    stop(paste0(prefix_m, "'folds' must be a positive integer or NULL"))

  r = met$repeats
  if (is.null(r))
    stop(paste0(prefix_m, "'repeats' must be provided and be a positive integer"))
  if (!(is.int(r) && r > 0L))
    stop(paste0(prefix_m, "'repeats' must be a positive integer"))
  met$repeats = as.integer(r)

  # Aggregation functions must be functions or NULL if provided.
  ap = met$aggregate_panels
  if (!is.null(ap) && !is.function(ap))
    stop(paste0(prefix_m, "'aggregate_panels' must be a function or NULL"))
  ar = met$aggregate_repeats
  if (!is.null(ar) && !is.function(ar))
    stop(paste0(prefix_m, "'aggregate_repeats' must be a function or NULL"))

  # Validate nuisance container (can be empty).
  nfs = met$nuisance
  if (is.null(nfs)) nfs = list()
  if (!is.list(nfs))
    stop(paste0(prefix_m, "'nuisance' must be a named list or NULL"))
  if (length(nfs)) {
    nfn = names(nfs)
    if (is.null(nfn) || any(is.na(nfn) | !nzchar(nfn)))
      stop(paste0(prefix_m, "all elements of 'nuisance' must be named with non-empty strings"))
    dup = unique(nfn[duplicated(nfn)])
    if (length(dup))
      stop(paste0(prefix_m, "duplicate nuisance names: ", paste(dup, collapse = ", ")))
    reserved = c("data", "model", "...", "__TARGET__")
    hit = intersect(nfn, reserved)
    if (length(hit))
      stop(paste0(prefix_m, "nuisance names must not use reserved names: ", paste(hit, collapse = ", ")))
    # Per-nuisance structural checks and normalization for user-specified nodes.
    for (nm in nfn) nfs[[nm]] = validate_nuisance(nfs[[nm]], nm, mname)
  }
  met$nuisance = nfs

  # Target must exist and be a function.
  if (is.null(met$target) || !is.function(met$target))
    stop(paste0(prefix_m, "target() must be a function"))

  # Required target args must be nuisances.
  req_target = setdiff(required_no_default(met$target), c("data", "..."))
  miss_t = setdiff(req_target, names(nfs))
  if (length(miss_t))
    stop(paste0(prefix_m, "target(): required args not in nuisance:", paste(miss_t, collapse = ", ")))

  # Attach synthetic '__TARGET__' nuisance node *before* graph-level checks.
  nf_tgt = list(
    fit        = function(data, ...) NULL,
    predict    = met$target,
    train_fold = 0L,
    eval_fold  = met$eval_fold,
    fit_deps   = NULL,
    pred_deps  = NULL
  )
  nfs$`__TARGET__` = nf_tgt

  # Graph-level normalization across all nuisances (including target()).
  if (length(nfs)) {
    for (nm in names(nfs)) {
      nf = nfs[[nm]]
      is_target = identical(nm, "__TARGET__")
      prefix = paste0(prefix_m, label_n(nm))

      # eval_fold: only target() uses it; for other nuisances we reset to 0.
      ne = nf$eval_fold
      if (!is_target && !is.null(ne))
        warning(paste(prefix, "defines 'eval_fold' but only target() uses it; ignoring."))
      nf$eval_fold = if (is_target) as.integer(ne) else 0L

      # Required fit()/predict() args (excluding data/model/...) used for coverage checks.
      req_fit  = setdiff(required_no_default(nf$fit), c("data", "..."))
      req_pred = setdiff(required_no_default(nf$predict), c("data", "model", "..."))

      fit_formals  = setdiff(names(formals(nf$fit)), c("data", "..."))
      pred_formals = setdiff(names(formals(nf$predict)), c("data", "model", "..."))

      # Base deps inferred only from required args whose names match nuisances.
      base_fit  = intersect(req_fit,  names(nfs)); base_fit  = setNames(base_fit,  base_fit)
      base_pred = intersect(req_pred, names(nfs)); base_pred = setNames(base_pred, base_pred)

      # User-specified fit_deps may override a subset of those.
      fd = nf$fit_deps
      if (!is.null(fd)) {
        nfd = names(fd)
        extra_keys = setdiff(nfd, fit_formals)
        if (length(extra_keys))
          stop(paste(prefix, "fit_deps contains unknown arg names:", paste(extra_keys, collapse = ", ")))
        bad_vals = setdiff(fd, names(nfs))
        if (length(bad_vals))
          stop(paste(prefix, "fit_deps references unknown nuisances:", paste(bad_vals, collapse = ", ")))
        base_fit[nfd] = fd
      }
      nf$fit_deps = base_fit[order(names(base_fit))]

      # User-specified pred_deps may override a subset of those.
      pd  = nf$pred_deps
      if (!is.null(pd)) {
        npd = names(pd)
        extra_keys = setdiff(npd, pred_formals)
        if (length(extra_keys))
          stop(paste(prefix, "pred_deps contains unknown arg names:", paste(extra_keys, collapse = ", ")))
        bad_vals = setdiff(pd, names(nfs))
        if (length(bad_vals))
          stop(paste(prefix, "pred_deps references unknown nuisances:", paste(bad_vals, collapse = ", ")))
        base_pred[npd] = pd
      }
      nf$pred_deps = base_pred[order(names(base_pred))]

      overlap = intersect(names(nf$fit_deps), names(nf$pred_deps))
      diff_map = overlap[nf$fit_deps[overlap] != nf$pred_deps[overlap]]
      if (length(diff_map)) warning(
        paste(prefix,
              "predict() inferred deps conflict with fit() ones for arg(s):",
              paste(diff_map, collapse = ", "),
              "- make sure this is intended for these arg(s)"))

      miss_cov_fit  = setdiff(req_fit, names(nf$fit_deps))
      if (length(miss_cov_fit))
        stop(paste(prefix,
                   "has required fit() args without a nuisance mapping:",
                   paste(miss_cov_fit, collapse = ", ")))
      miss_cov_pred = setdiff(req_pred, names(nf$pred_deps))
      if (length(miss_cov_pred))
        stop(paste(prefix,
                   "has required predict() args without a nuisance mapping:",
                   paste(miss_cov_pred, collapse = ", ")))

      nfs[[nm]] = nf
    }

    # Warn on nuisances declared but never used in any fit()/predict()/target() deps.
    nuis_names = setdiff(names(nfs), "__TARGET__")
    used_nuis = unique(unlist(
      lapply(nfs, function(nf) c(nf$fit_deps, nf$pred_deps)),
      use.names = FALSE
    ))
    unused = setdiff(nuis_names, used_nuis)
    if (length(unused))
      warning(paste0(
        prefix_m,
        "The following nuisances are declared but never used in any fit()/predict()/target() dependencies: ",
        paste(unused, collapse = ", ")
      ))

    # Per-method cycle check on the full nuisance graph (including target()).
    tryCatch({
      check_cycles(nfs)
    }, error = function(e) {
      stop(paste0(prefix_m, conditionMessage(e)), call. = FALSE)
    })
  }

  met$nuisance = nfs
  met
}

# Convenience constructors -----------------------------------------------

#' Create a nuisance specification
#'
#' Helper to create a nuisance specification with basic structural
#' checks. A nuisance is defined by a \code{fit} function, a
#' \code{predict} function, and optional dependency mappings.
#'
#' @param fit A function \code{fit(data, ...)} that trains the nuisance
#'   model on a subset of the data and returns a fitted model object.
#' @param predict A function \code{predict(model, data, ...)} that
#'   returns predictions for the nuisance on new data.
#' @param train_fold Positive integer giving the width (in folds) of the
#'   training window used for this nuisance. Defaults to \code{1L}.
#' @param fit_deps Optional named character vector mapping
#'   \code{fit()} argument names to nuisance names, used to specify
#'   nuisance inputs to the \code{fit} function. If \code{NULL}, the
#'   dependencies are inferred later from required arguments whose names
#'   match nuisance names.
#' @param pred_deps Optional named character vector mapping
#'   \code{predict()} argument names to nuisance names, used to specify
#'   nuisance inputs to the \code{predict} function.
#'
#' @return A list representing a nuisance specification, suitable for
#'   inclusion in the \code{list_nuisance} argument of
#'   \code{\link{create_method}}.
#'
#' @export
#' @examples
#' # Simple linear regression nuisance: E[Y | X]
#' set.seed(1)
#' n <- 50
#' x <- rnorm(n)
#' y <- x + rnorm(n)
#'
#' nuis <- create_nuisance(
#'   fit = function(data, ...) lm(y ~ x, data = data),
#'   predict = function(model, data, ...) predict(model, newdata = data)
#' )
#'
#' str(nuis)
create_nuisance = function(fit,
                           predict,
                           train_fold = 1L,
                           fit_deps = NULL,
                           pred_deps = NULL) {
  nf = list(
    fit        = fit,
    predict    = predict,
    train_fold = train_fold,
    eval_fold  = NULL,
    fit_deps   = fit_deps,
    pred_deps  = pred_deps
  )
  nf = validate_nuisance(nf, nm = NULL, mname = NULL)
  nf
}

#' Create a cross-fitting method specification
#'
#' Helper to create a method specification for
#' \code{\link{crossfit}} / \code{\link{crossfit_multi}}. A method
#' bundles together:
#' \itemize{
#'   \item a target functional \code{target()},
#'   \item a named list of nuisance specifications,
#'   \item cross-fitting geometry (\code{folds}, \code{repeats},
#'         \code{eval_fold}, \code{mode}, \code{fold_allocation}),
#'   \item and panel / repetition aggregation functions.
#' }
#'
#' The returned list is validated by \code{validate_method()} to ensure
#' structural soundness, but the validated object is not stored: you are
#' free to modify the returned method before passing it to
#' \code{\link{crossfit}} or \code{\link{crossfit_multi}}.
#'
#' By default, \code{eval_fold} is chosen to be \code{1L} when
#' \code{mode = "estimate"} and \code{0L} when \code{mode = "predict"}.
#' If you override \code{eval_fold}, it must satisfy these constraints:
#' positive integer for \code{"estimate"}, zero for \code{"predict"}.
#'
#' @param target A function representing the target functional. It must
#'   accept nuisance predictions as arguments (named after nuisances) and
#'   optionally a \code{data} argument.
#' @param list_nuisance Optional named list of nuisance specifications
#'   created by \code{\link{create_nuisance}}.
#' @param folds Positive integer giving the number of folds \eqn{K}. May
#'   be \code{NULL}, in which case \code{\link{crossfit_multi}} will
#'   infer a minimal feasible \code{K} from the dependency structure.
#' @param repeats Positive integer giving the number of repetitions.
#' @param mode Cross-fitting mode. Either \code{"estimate"} (target
#'   returns numeric estimates) or \code{"predict"} (target returns a
#'   cross-fitted predictor).
#' @param eval_fold Integer giving the width (in folds) of the
#'   evaluation window for the target. Must be \code{> 0} for
#'   \code{mode = "estimate"} and \code{0} for \code{mode = "predict"}.
#'   If omitted, the default is \code{1L} for \code{"estimate"} and
#'   \code{0L} for \code{"predict"}.
#' @param fold_allocation Fold allocation strategy; one of
#'   \code{"independence"}, \code{"overlap"}, or \code{"disjoint"}.
#' @param aggregate_panels Aggregation function for panel-level
#'   results, typically one of \code{\link{mean_estimate}},
#'   \code{\link{median_estimate}}, \code{\link{mean_predictor}},
#'   \code{\link{median_predictor}}, or a custom function. May be
#'   \code{NULL}, in which case a global default can be supplied via
#'   \code{\link{crossfit_multi}}.
#' @param aggregate_repeats Aggregation function for repetition-level
#'   results, typically one of \code{\link{mean_estimate}},
#'   \code{\link{median_estimate}}, \code{\link{mean_predictor}},
#'   \code{\link{median_predictor}}, or a custom function. May be
#'   \code{NULL}, in which case a global default can be supplied via
#'   \code{\link{crossfit_multi}}.
#'
#' @return A method specification list suitable for use in
#'   \code{\link{crossfit}} or \code{\link{crossfit_multi}}.
#'
#' @export
#' @examples
#' set.seed(1)
#' n <- 50
#' x <- rnorm(n)
#' y <- x + rnorm(n)
#'
#' # Nuisance: regression for E[Y | X]
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
#' m <- create_method(
#'   target = target_mse,
#'   list_nuisance = list(nuis_y = nuis_y),
#'   folds = 2,
#'   repeats = 1,
#'   eval_fold = 1L,
#'   mode = "estimate",
#'   fold_allocation = "independence",
#'   aggregate_panels  = mean_estimate,
#'   aggregate_repeats = mean_estimate
#' )
#'
#' str(m)
create_method = function(target,
                         list_nuisance = NULL,
                         folds,
                         repeats,
                         mode = c("estimate", "predict"),
                         eval_fold = if (mode == "estimate") 1L else 0L,
                         fold_allocation = c("independence", "overlap", "disjoint"),
                         aggregate_panels = NULL,
                         aggregate_repeats = NULL) {
  mode = match.arg(mode)
  fold_allocation = match.arg(fold_allocation)

  met = list(
    target            = target,
    nuisance          = list_nuisance,
    folds             = folds,
    repeats           = repeats,
    mode              = mode,
    eval_fold         = eval_fold,
    fold_allocation   = fold_allocation,
    aggregate_panels  = aggregate_panels,
    aggregate_repeats = aggregate_repeats
  )

  # Validate but do not store the validated result; this ensures the
  # created method is structurally sound even if the user later modifies it.
  validate_method(met, mname = NULL)
  met
}

# Normalization -----------------------------------------------------------

#' Internal: validate and normalize a batch of methods
#'
#' Standardizes a list of method specifications:
#' \itemize{
#'   \item ensures every method has a name,
#'   \item checks and sets \code{mode} and \code{fold_allocation},
#'   \item fills in per-method aggregation functions from global defaults,
#'   \item calls \code{validate_method()} on each method.
#' }
#'
#' This is the entry point used by \code{\link{crossfit_multi}} to
#' validate user-specified methods. It is not intended for direct use by
#' end users.
#'
#' @param methods A list of method specifications.
#' @param default_agg_panels Default aggregation function for panels
#'   (used when a method does not provide \code{aggregate_panels}).
#'   Must be a function.
#' @param default_agg_repeats Default aggregation function for
#'   repetitions (used when a method does not provide
#'   \code{aggregate_repeats}). Must be a function.
#'
#' @return A named list of fully validated and normalized methods.
#'
#' @importFrom stats setNames
#'
#' @keywords internal
validate_batch = function(methods, default_agg_panels, default_agg_repeats) {

  if (length(methods) == 0L) stop("'methods' must contain at least one method")

  if (!is.function(default_agg_panels))  stop("'default_agg_panels' must be a function")
  if (!is.function(default_agg_repeats)) stop("'default_agg_repeats' must be a function")

  # Ensure every method has a stable name (used in errors and output).
  mnames = names(methods)
  if (is.null(mnames)) mnames = rep("", length(methods))
  missing = is.na(mnames) | !nzchar(mnames)
  mnames = ifelse(missing, paste0("method_", seq_along(methods)), mnames)

  res = lapply(seq_along(methods), function(i) {
    met = methods[[i]]
    mname = mnames[i]

    # Per-method mode and fold allocation must be explicitly provided.
    m_mode = met$mode
    if (is.null(m_mode))
      stop(sprintf("In method '%s': 'mode' must be specified as 'estimate' or 'predict'", mname))
    m_mode = match.arg(m_mode, c("estimate", "predict"))
    met$mode = m_mode

    m_alloc = met$fold_allocation
    if (is.null(m_alloc))
      stop(sprintf(
        "In method '%s': 'fold_allocation' must be specified as 'independence', 'overlap' or 'disjoint'",
        mname
      ))
    m_alloc = match.arg(m_alloc, c("independence", "overlap", "disjoint"))
    met$fold_allocation = m_alloc

    # Per-method aggregation functions: use method-specific if provided,
    # otherwise fall back to the global defaults given to crossfit_multi.
    ap = met$aggregate_panels
    ar = met$aggregate_repeats

    ap = if (!is.null(ap)) ap else default_agg_panels
    ar = if (!is.null(ar)) ar else default_agg_repeats

    if (!is.function(ap))
      stop(sprintf("In method '%s': 'aggregate_panels' must be a function", mname))
    if (!is.function(ar))
      stop(sprintf("In method '%s': 'aggregate_repeats' must be a function", mname))

    met$aggregate_panels  = ap
    met$aggregate_repeats = ar

    # Full validation and graph normalization (adds __TARGET__, deps, cycles).
    validate_method(met, mname)
  })

  setNames(res, mnames)
}
