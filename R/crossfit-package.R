#' crossfit: Cross-Fitting Engine for Double / Debiased Machine Learning
#'
#' Provides a general cross-fitting engine for double / debiased machine
#' learning and other meta-learners. The core functions implement flexible
#' graphs of nuisance models with per-node training fold widths, target-specific
#' evaluation windows, and several fold allocation schemes ("independence",
#' "overlap", "disjoint"). The engine supports both numeric estimators
#' (mode = "estimate") and cross-fitted prediction functions (mode = "predict"),
#' with configurable aggregation over panels and repetitions.
#'
#' @keywords package
"_PACKAGE"
