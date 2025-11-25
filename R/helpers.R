# log utilities ------------------------------------------------------------

show_log = FALSE

#' Internal logging utilities
#'
#' These helpers print training and prediction schedules for debugging
#' the cross-fitting plan. Logging is controlled by the global
#' \code{show_log} flag: when \code{show_log} is \code{TRUE}, both
#' \code{log_train()} and \code{log_pred()} write messages to the console.
#'
#' @param inst An instance descriptor from the internal plan object.
#' @param token A token created by \code{make_eval_token()} or
#'   \code{make_train_token()}, indicating the method index and panel.
#' @param folds Integer vector of fold indices involved in the
#'   operation.
#' @param insts List of all instances (for \code{log_pred()} only),
#'   used to resolve the goal of the prediction.
#'
#' @return These functions are called for their side-effect (printing);
#'   they return \code{NULL} invisibly.
#'
#' @keywords internal
log_train = function(inst, token, folds) {
  eval_msg = paste0("M", token$mi, " F", token$p)
  inst_msg = paste(inst$name, "training on folds", paste0(folds, collapse = ", "))
  cat(eval_msg, "|", inst_msg, "\n")
}

#' @rdname log_train
#' @keywords internal
log_pred = function(inst, token, folds, insts) {
  eval_msg = paste0("M", token$mi, " F", token$p)
  goal_msg = if (identical(token$type, "train"))
    paste0("training ", insts[[token$inst_key]]$name) else
      paste0(insts[[token$inst_key]]$name, " prediction")
  inst_msg = paste0(inst$name, " pred on folds ",
                    paste0(folds, collapse = ", "),
                    " for ", goal_msg)
  cat(eval_msg, "|", inst_msg, "\n")
}

# Internal helper utilities -----------------------------------------------

#' Internal helper utilities
#'
#' Small utilities used internally by the cross-fitting engine.
#'
#' \itemize{
#'   \item \code{idx_mod()} wraps panel indices modulo the number of folds,
#'     using 1-based indexing.
#'   \item \code{pass_named()} calls a function with only the arguments
#'     it declares, dropping unknown ones defensively.
#'   \item \code{is.int()} checks whether a value is a non-negative
#'     scalar integer (within floating-point tolerance).
#'   \item \code{required_no_default()} returns the names of required
#'     formals (arguments without defaults) of a function.
#' }
#'
#' These helpers are not intended for direct use by end users.
#'
#' @param n Integer index or count used in modulo arithmetic.
#' @param mod Modulus (number of folds).
#' @param fun A function to be called.
#' @param args A named list of arguments.
#'
#' @return Various small values used for internal control flow and
#'   argument handling.
#'
#' @keywords internal
idx_mod = function(n, mod) (n - 1) %% mod + 1

#' @rdname idx_mod
#' @keywords internal
pass_named = function(fun, args) {
  if (is.null(args) || length(args) == 0L) return(fun())
  keep = names(args) %in% names(formals(fun))
  do.call(fun, args[keep])
}

#' @rdname idx_mod
#' @keywords internal
is.int = function(n)
  is.numeric(n) && length(n) == 1L && is.finite(n) && n >= 0 && n == floor(n)

#' @rdname idx_mod
#' @keywords internal
required_no_default = function(fun) {
  fr = formals(fun); if (is.null(fr)) return(character(0))
  nm = names(fr)
  nodef = vapply(fr, function(x) identical(x, quote(expr = )), logical(1))
  nm[nodef]
}

# Layered predictors ------------------------------------------------------

#' Internal: wrap a model and its child predictors
#'
#' Wraps a fitted model and its prediction function into a layered
#' predictor that also calls dependency predictors for nuisance inputs.
#' This is used internally in \code{mode = "predict"} to build
#' cross-fitted predictors that compose multiple nuisance learners.
#'
#' @param pred Prediction function for the nuisance node (typically the
#'   \code{predict} component of a nuisance specification).
#' @param model Fitted model object returned by the corresponding
#'   \code{fit()}.
#' @param pred_deps_predict Optional list of child predictors, each of
#'   which will be called on \code{newdata} and passed into \code{pred}
#'   under the appropriate argument name.
#'
#' @return A function \code{f(newdata, ...)} calling
#'   \code{pred(model, data = newdata, ...)} with extra arguments coming
#'   from \code{pred_deps_predict}. Intended for internal use.
#'
#' @keywords internal
layer = function(pred, model, pred_deps_predict = NULL) {
  function(newdata, ...) {
    dots = list(...); nm = names(dots)
    if (length(dots) && !is.null(nm)) {
      dots = dots[!names(dots) %in% c("model", "data")]
    }
    extra = lapply(pred_deps_predict, function(f) f(newdata))
    pass_named(
      pred,
      c(list(model = model, data = newdata), extra, dots)
    )
  }
}

# Aggregators for numeric estimates ---------------------------------------

#' Aggregators for scalar estimates
#'
#' These helpers implement simple aggregation schemes for panel-level
#' and repetition-level estimates in \code{\link{crossfit}} and
#' \code{\link{crossfit_multi}}.
#'
#' In \code{mode = "estimate"}, each repetition typically produces a list
#' of numeric values (one per evaluation panel). The functions
#' \code{mean_estimate()} and \code{median_estimate()} aggregate such
#' lists into a single numeric value.
#'
#' @param xs A list of numeric values or numeric vectors. Elements are
#'   unlisted and concatenated prior to aggregation, so \code{xs} may
#'   contain scalars or length-\eqn{k} vectors.
#'
#' @return A single numeric value (the mean or median of all entries in
#'   \code{xs}.
#'
#' @export
#' @examples
#' xs <- list(c(1, 2, 3), 4, c(5, 6))
#' mean_estimate(xs)
mean_estimate = function(xs) mean(unlist(xs))

#' @rdname mean_estimate
#' @importFrom stats median
#' @export
#' @examples
#' xs <- list(c(1, 100), 10, 20)
#' median_estimate(xs)

median_estimate = function(xs) median(unlist(xs))

# Aggregators for predictors ----------------------------------------------

#' Aggregators for cross-fitted predictors
#'
#' These helpers aggregate several cross-fitted predictors into a single
#' ensemble predictor. They are designed for methods run with
#' \code{mode = "predict"} in \code{\link{crossfit}} and
#' \code{\link{crossfit_multi}}.
#'
#' @param fs A list of prediction functions. Each function must accept
#'   at least a \code{newdata} argument and return a numeric vector of
#'   predictions of the same length as \code{nrow(newdata)}.
#'
#' @return A function of the form \code{function(newdata, ...)}, which
#'   returns a numeric vector of predictions. If \code{fs} is empty, the
#'   returned function always returns \code{numeric(0)}.
#'
#' @export
#' @examples
#' # Two simple prediction functions of x
#' f1 <- function(newdata, ...) newdata$x
#' f2 <- function(newdata, ...) 2 * newdata$x
#'
#' ens_mean   <- mean_predictor(list(f1, f2))
#'
#' newdata <- data.frame(x = 1:5)
#' ens_mean(newdata)
mean_predictor = function(fs) {
  function(newdata, ...) {
    if (length(fs) == 0L) return(numeric(0))
    vals = lapply(fs, function(f) f(newdata, ...))
    if (length(vals) == 1L) return(vals[[1]])
    Reduce(`+`, vals) / length(vals)
  }
}

#' @rdname mean_predictor
#' @importFrom stats median
#' @export
#' @examples
#' # Two simple prediction functions of x
#' f1 <- function(newdata, ...) newdata$x
#' f2 <- function(newdata, ...) 2 * newdata$x
#'
#' ens_median <- median_predictor(list(f1, f2))
#'
#' newdata <- data.frame(x = 1:5)
#' ens_median(newdata)
median_predictor = function(fs) {
  function(newdata, ...) {
    if (length(fs) == 0L) return(numeric(0))
    vals = lapply(fs, function(f) f(newdata, ...))
    if (length(vals) == 1L) return(vals[[1]])
    mat = do.call(cbind, vals)
    apply(mat, 1L, median)
  }
}
