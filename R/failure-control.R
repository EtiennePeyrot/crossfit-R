#' Configure failure handling for a cross-fitting method
#'
#' Creates a named list of method-specific failure-handling options for
#' \code{\link{create_method}}.
#'
#' By default, an error in any panel fails the current repetition. If
#' \code{fail_repetition_on_error = FALSE}, the engine evaluates the remaining
#' panels, represents each failed panel by \code{NA_real_}, and passes the
#' complete panel list to the method's \code{aggregate_panels} function. The
#' repetition is successful when that aggregation succeeds.
#'
#' If \code{prune_on_shared_fit_failure = TRUE}, a method is skipped when a
#' structurally identical nuisance fit has already failed in the same
#' repetition. If it is \code{FALSE}, the method attempts the fit independently.
#'
#' @param fail_repetition_on_error Logical scalar. If \code{TRUE}, the first
#'   panel error fails the repetition. If \code{FALSE}, all panels are attempted and the
#'   panel aggregator decides whether the available results are sufficient.
#' @param prune_on_shared_fit_failure Logical scalar. If \code{TRUE}, a method is
#'   skipped within a repetition when an earlier method encountered a failure
#'   in a structurally identical nuisance fit. This option requires
#'   \code{fail_repetition_on_error = TRUE}.
#' @param max_failed_repetitions Non-negative integer or \code{Inf}. A method is
#'   disabled after it exceeds this number of failed repetitions.
#'
#' @return A named list suitable for the \code{failure_control} argument of
#'   \code{\link{create_method}}.
#'
#' @export
#' @examples
#' failure_control <- crossfit_failure_control(
#'   fail_repetition_on_error = FALSE,
#'   max_failed_repetitions = 2L
#' )
#'
#' str(failure_control)
crossfit_failure_control = function(
    fail_repetition_on_error = TRUE,
    prune_on_shared_fit_failure = FALSE,
    max_failed_repetitions = Inf
) {
  validate_failure_control(list(
    fail_repetition_on_error = fail_repetition_on_error,
    prune_on_shared_fit_failure = prune_on_shared_fit_failure,
    max_failed_repetitions = max_failed_repetitions
  ))
}

#' Internal: validate and normalize failure controls
#'
#' @param control A named list, or \code{NULL} for the defaults.
#' @param prefix Optional prefix for validation errors.
#'
#' @return A complete named failure-control list.
#'
#' @keywords internal
validate_failure_control = function(control = NULL, prefix = "") {
  defaults = list(
    fail_repetition_on_error = TRUE,
    prune_on_shared_fit_failure = FALSE,
    max_failed_repetitions = Inf
  )

  if (is.null(control)) return(defaults)
  if (!is.list(control))
    stop(paste0(prefix, "'failure_control' must be a named list or NULL"))

  control_names = names(control)
  if (length(control) &&
      (is.null(control_names) || any(is.na(control_names) | !nzchar(control_names))))
    stop(paste0(prefix, "all elements of 'failure_control' must be named"))

  duplicate_names = unique(control_names[duplicated(control_names)])
  if (length(duplicate_names))
    stop(paste0(
      prefix, "duplicate 'failure_control' entries: ",
      paste(duplicate_names, collapse = ", ")
    ))

  unknown = setdiff(control_names, names(defaults))
  if (length(unknown))
    stop(paste0(
      prefix, "unknown 'failure_control' entries: ",
      paste(unknown, collapse = ", ")
    ))

  defaults[control_names] = control
  control = defaults

  logical_fields = c(
    "fail_repetition_on_error",
    "prune_on_shared_fit_failure"
  )
  for (field in logical_fields) {
    value = control[[field]]
    if (!(is.logical(value) && length(value) == 1L && !is.na(value)))
      stop(paste0(prefix, "'failure_control$", field, "' must be TRUE or FALSE"))
  }

  max_failed = control$max_failed_repetitions
  valid_inf = is.numeric(max_failed) && length(max_failed) == 1L &&
    is.infinite(max_failed) && max_failed > 0
  valid_max = valid_inf || is.int(max_failed)
  if (!valid_max)
    stop(paste0(
      prefix,
      "'failure_control$max_failed_repetitions' must be a non-negative integer or Inf"
    ))

  if (control$prune_on_shared_fit_failure && !control$fail_repetition_on_error)
    stop(paste0(
      prefix,
      "'failure_control$prune_on_shared_fit_failure = TRUE' requires ",
      "'failure_control$fail_repetition_on_error = TRUE'"
    ))

  control$max_failed_repetitions = if (is.infinite(max_failed)) Inf else as.integer(max_failed)
  control
}
