# Function signatures -----------------------------------------------------

#' Internal: create a function registry
#'
#' Constructs a small environment used to deduplicate identical
#' \code{fit()} / \code{predict()} functions across methods. Functions
#' are represented by string signatures (via \code{\link{fun_code_sig}})
#' stored in \code{$sigs}.
#'
#' @return An environment with a character vector component \code{sigs}
#'   used by \code{\link{fun_registry_id}}.
#'
#' @keywords internal
fun_registry_new = function() {
  e = new.env(parent = emptyenv())
  e$sigs = character(0)
  e
}

#' Internal: compute a code signature for a function
#'
#' Builds a crude but stable code signature for a function based on its
#' argument names and body. Argument names are sorted (to avoid
#' dependence on declaration order) and the body is \code{deparse()}d
#' into a single string. The result is used as a key for function
#' deduplication.
#'
#' @param f A function object.
#'
#' @return A single character string representing the function's code
#'   signature.
#'
#' @keywords internal
fun_code_sig = function(f) {
  fm = sort(names(formals(f))); bd = deparse(body(f))
  paste(c("F:", paste(fm, collapse = ","), "B:", paste(bd, collapse = "")), collapse = "")
}

#' Internal: assign an integer id to a function signature
#'
#' Maps a function code signature (as produced by
#' \code{\link{fun_code_sig}}) to a small integer id in a registry. If
#' the signature is new, it is appended to the registry; otherwise, the
#' existing id is returned. This is used to keep structural signatures
#' compact.
#'
#' @param reg A registry environment created by
#'   \code{\link{fun_registry_new}}.
#' @param sig A character string code signature.
#'
#' @return An integer id corresponding to \code{sig} within
#'   \code{reg}.
#'
#' @keywords internal
fun_registry_id = function(reg, sig) {
  idx = match(sig, reg$sigs, nomatch = 0L)
  if (idx == 0L) {
    reg$sigs = c(reg$sigs, sig)
    idx = length(reg$sigs)
  }
  idx
}

# Plan / instances --------------------------------------------------------

#' Internal: build per-method instance graph and fold geometry
#'
#' Given a list of validated methods (as returned by
#' \code{\link{validate_batch}}), constructs the global instance graph
#' used by the cross-fitting engine. Each nuisance (including the
#' synthetic \code{"__TARGET__"} node) is expanded into one or more
#' instances, depending on the fold allocation strategy.
#'
#' This function:
#' \itemize{
#'   \item builds a DAG of instances per method, starting from
#'         \code{"__TARGET__"},
#'   \item computes per-method evaluation width (\code{eval_width}),
#'   \item assigns training window offsets (\code{inst_offset}) according
#'         to \code{fold_allocation},
#'   \item determines the minimal required number of folds
#'         (\code{K_required}) per method and harmonizes a global
#'         \code{K},
#'   \item constructs child maps (via \code{\link{build_child_maps}}),
#'   \item computes structural signatures and marks instances that are
#'         worth caching.
#' }
#'
#' The resulting "plan" object is consumed by the core engine
#' (\code{crossfit_multi()}).
#'
#' @param methods A list of fully validated and normalized method
#'   specifications, typically the output of
#'   \code{\link{validate_batch}}.
#'
#' @return A list with components including (but not limited to):
#'   \describe{
#'     \item{\code{methods}}{The (possibly updated) methods list, with
#'           \code{folds} harmonized.}
#'     \item{\code{instances}}{Named list of instance nodes.}
#'     \item{\code{roots}}{Per-method root instance keys
#'           (corresponding to \code{"__TARGET__"}).}
#'     \item{\code{topo}}{Instance keys in topological order.}
#'     \item{\code{method_inst_keys}}{Per-method instance keys in
#'           topological order.}
#'     \item{\code{eval_width}}{Per-method evaluation window width.}
#'     \item{\code{inst_offset}}{Per-instance training window offset
#'           (in folds).}
#'     \item{\code{K_required}}{Per-method minimal required number of
#'           folds.}
#'     \item{\code{K}}{Global number of folds used in the plan.}
#'     \item{\code{child_maps}}{List of fit/predict child maps as
#'           returned by \code{\link{build_child_maps}}.}
#'     \item{\code{method_structs}}{Per-method set of structural
#'           signatures used by that method.}
#'     \item{\code{method_fold_allocation}}{Per-method
#'           \code{fold_allocation} values.}
#'   }
#'
#' @importFrom stats setNames
#' @importFrom utils head
#'
#' @keywords internal
build_instances = function(methods) {

  insts = list()
  roots = vector("list", length(methods))
  topo  = character(0)

  # Recursively register an instance, then its children (if any).
  add_inst = function(mi, nfs, node, ctx_vec) {
    alloc_mode = methods[[mi]]$fold_allocation
    key = if (identical(alloc_mode, "independence"))
      paste(c(paste0("M", mi), node, ctx_vec), collapse = "|") else
        paste0("M", mi, "|", node)

    if (!is.null(insts[[key]])) return(key)
    nf = nfs[[node]]
    child_names = unique(c(nf$pred_deps, nf$fit_deps))
    child_keys = vapply(child_names, function(ch) add_inst(mi, nfs, ch, c(node, ctx_vec)), character(1))
    insts[[key]] <<- list(
      key             = key,
      method_idx      = mi,
      name            = node,
      children        = child_keys,
      fit_deps_names  = nf$fit_deps,
      pred_deps_names = nf$pred_deps
    )
    topo <<- c(topo, key)
    key
  }

  # For each method, build the instance graph reachable from __TARGET__.
  for (mi in seq_along(methods)) {
    nfs = methods[[mi]]$nuisance
    roots[[mi]] = add_inst(mi, nfs, "__TARGET__", NULL)
  }
  topo = unique(topo)
  insts = insts[topo]

  # Instances belonging to each method, in topological order.
  method_inst_keys = lapply(seq_along(methods), function(mi)
    Filter(function(k) insts[[k]]$method_idx == mi, topo)
  )
  names(method_inst_keys) = names(methods)

  # Per-method eval width and per-instance training offsets.
  eval_width  = setNames(integer(length(methods)), names(methods))
  inst_offset = setNames(integer(length(insts)), names(insts))
  K_required  = setNames(integer(length(methods)), names(methods))

  for (mi in seq_along(methods)) {
    nfs  = methods[[mi]]$nuisance
    w_eval = nfs[["__TARGET__"]]$eval_fold
    eval_width[mi] = w_eval

    train_keys = Filter(function(k) insts[[k]]$name != "__TARGET__", method_inst_keys[[mi]])

    if (!length(train_keys)) { K_required[mi] = w_eval; next }

    # Width for each training node.
    w_vec = vapply(train_keys, function(k) nfs[[insts[[k]]$name]]$train_fold, integer(1))

    alloc_mode = methods[[mi]]$fold_allocation
    if (identical(alloc_mode, "independence")) {
      # Disjoint training blocks after eval (one per instance, including duplicates).
      offsets_m = w_eval + head(c(0L, cumsum(w_vec)), -1)
      for (j in seq_along(train_keys)) inst_offset[[train_keys[j]]] = offsets_m[j]
      K_required[mi] = w_eval + sum(w_vec)
    } else if (identical(alloc_mode, "overlap")) {
      # All nuisances start right after eval; blocks may overlap but stay
      # outside the evaluation window as long as K >= w_eval + max(train_fold).
      for (k in train_keys) inst_offset[[k]] = w_eval
      K_required[mi] = w_eval + max(w_vec)
    } else if (identical(alloc_mode, "disjoint")) {
      # Unique nuisances (no duplication via ctx_vec) get disjoint blocks
      # after eval, like in the independence schedule but with one instance
      # per nuisance name.
      offsets_m = w_eval + head(c(0L, cumsum(w_vec)), -1)
      for (j in seq_along(train_keys)) inst_offset[[train_keys[j]]] = offsets_m[j]
      K_required[mi] = w_eval + sum(w_vec)
    }
  }

  # Harmonize the global K across methods.
  K_min = max(K_required)
  fold_list = lapply(methods, `[[`, "folds")
  is_null = sapply(fold_list, is.null)
  if (any(is_null)) {
    msg_methods = names(methods)[is_null]
    message(sprintf(
      "methods without 'folds' (%s) set to minimal K=%d",
      paste(msg_methods, collapse = ", "), K_min
    ))
    for (i in which(is_null)) methods[[i]]$folds = K_min
  }

  k_vals = vapply(methods, function(m) m$folds, numeric(1))
  too_small = k_vals < K_required
  if (any(too_small)) {
    bad = names(methods)[too_small]
    bad_str = paste(sprintf(
      "%s: folds=%d, minimal_K=%d",
      bad, as.integer(k_vals[too_small]), as.integer(K_required[too_small])
    ), collapse = "; ")
    stop(sprintf(
      "Some methods have 'folds' smaller than required by their dependency structure: %s",
      bad_str
    ))
  }

  uniq_k = sort(unique(k_vals))
  if (length(uniq_k) > 1L)
    stop(sprintf("All methods must share the same 'folds'; got: %s", paste(uniq_k, collapse = ", ")))
  K = uniq_k[1L]

  # Build child maps once (fit and predict) based on the final instance graph.
  child_maps = build_child_maps(insts)

  fun_registry = fun_registry_new()
  struct_cache = new.env(parent = emptyenv())

  struct_sig_for = function(key) {
    sig = struct_cache[[key]]
    if (!is.null(sig)) return(sig)
    inst = insts[[key]]
    nf   = methods[[inst$method_idx]]$nuisance[[inst$name]]

    fit_sig = fun_code_sig(nf$fit)
    fit_id  = fun_registry_id(fun_registry, fit_sig)

    base = paste(
      paste0("FIT#", fit_id),
      paste0("TF:", nf$train_fold),
      paste0("EV:", nf$eval_fold),
      sep = "|"
    )

    fc = child_maps$fit[[key]]
    parent_off = inst_offset[[key]]

    child_parts = vapply(names(fc), function(arg) {
      ck = fc[[arg]]
      ch_inst = insts[[ck]]
      child_nf = methods[[ch_inst$method_idx]]$nuisance[[ch_inst$name]]
      child_pred_sig = fun_code_sig(child_nf$predict)
      child_pred_id  = fun_registry_id(fun_registry, child_pred_sig)
      child_struct_sig = struct_sig_for(ck)
      child_off = inst_offset[[ck]]
      rel_off   = child_off - parent_off
      paste0(arg, ":[", child_struct_sig, "][PRED#", child_pred_id, "][REL:", rel_off, "]")
    }, character(1))

    sig = paste(base, paste(child_parts, collapse = ";"), sep = "|CH:")
    struct_cache[[key]] <<- sig
    sig
  }
  # Ensure every instance has its structural signature computed and stored.
  for (k in topo) struct_sig_for(k)
  for (inst_key in names(insts)) insts[[inst_key]]$struct_sig = struct_cache[[inst_key]]

  # Structural signature per instance (used for sharing and diagnostics)
  struct_vec    = vapply(insts, function(inst) inst$struct_sig, character(1))
  struct_counts = table(struct_vec)

  # Structural footprint of each method (which structural nuisances it uses)
  method_structs = lapply(method_inst_keys, function(keys) unique(struct_vec[keys]))

  # Fan-in count per instance (how many parents reference this node)
  fan_in = setNames(integer(length(insts)), names(insts))
  for (inst_key in names(insts)) {
    for (child in insts[[inst_key]]$children) {
      fan_in[[child]] = fan_in[[child]] + 1L
    }
  }
  # Decide which instances are worth caching based on structural reuse
  for (inst_key in names(insts)) {
    sig = insts[[inst_key]]$struct_sig
    shared = struct_counts[[sig]] > 1L
    shared_parents = fan_in[[inst_key]] > 1L
    insts[[inst_key]]$cache_model = shared || shared_parents
  }

  list(
    methods               = methods,
    instances             = insts,
    roots                 = roots,
    topo                  = topo,
    method_inst_keys      = method_inst_keys,
    eval_width            = eval_width,
    inst_offset           = inst_offset,
    K_required            = K_required,
    K                     = K,
    child_maps            = child_maps,
    method_structs        = method_structs,
    method_fold_allocation = vapply(methods, `[[`, character(1), "fold_allocation")
  )
}

# Child maps --------------------------------------------------------------

#' Internal: map nuisance names to instance keys for each node
#'
#' Builds, for each instance, a mapping from nuisance names (as used in
#' \code{fit_deps_names} / \code{pred_deps_names}) to child instance
#' keys. This allows the engine to route predictions correctly from
#' child nuisances into parent \code{fit()} and \code{predict()}
#' calls.
#'
#' @param insts A named list of instances as produced by
#'   \code{\link{build_instances}}.
#'
#' @return A list with two components:
#'   \describe{
#'     \item{\code{fit}}{Named list mapping instance keys to named
#'           character vectors (args \eqn{\to} child instance keys) for
#'           \code{fit()} dependencies.}
#'     \item{\code{pred}}{Same for \code{predict()} dependencies.}
#'   }
#'
#' @importFrom stats setNames
#'
#' @keywords internal
build_child_maps = function(insts) {
  child_name_index = lapply(insts, function(inst) {
    cn = vapply(inst$children, function(k) insts[[k]]$name, character(1))
    setNames(inst$children, cn)
  })

  fit_child_map = mapply(insts, child_name_index, FUN = function(inst, idx)
    vapply(inst$fit_deps_names, function(dep_nm) {
      ck = idx[[dep_nm]]
      if (is.null(ck))
        stop(sprintf("Cannot resolve fit_dep '%s' for instance '%s'", dep_nm, inst$key))
      ck
    }, character(1)), SIMPLIFY = FALSE)

  pred_child_map = mapply(insts, child_name_index, FUN = function(inst, idx)
    vapply(inst$pred_deps_names, function(dep_nm) {
      ck = idx[[dep_nm]]
      if (is.null(ck))
        stop(sprintf("Cannot resolve pred_dep '%s' for instance '%s'", dep_nm, inst$key))
      ck
    }, character(1)), SIMPLIFY = FALSE)

  list(fit = fit_child_map, pred = pred_child_map)
}

# Model signature ---------------------------------------------------------

#' Internal: compute a model cache key for an instance
#'
#' Computes a unique model signature for a given instance under a given
#' evaluation token. The signature combines:
#' \itemize{
#'   \item the instance's structural signature (\code{inst$struct_sig}),
#'   \item the set of training folds used for this eval panel.
#' }
#'
#' This is used as a key in the model cache so that models are reused
#' exactly when both structure and training data coincide.
#'
#' @param inst_key Instance key (character) in \code{plan$instances}.
#' @param eval_token An evaluation token created by
#'   \code{make_eval_token()}.
#' @param methods The methods list used to build \code{plan}.
#' @param plan The cross-fitting plan as returned by
#'   \code{\link{build_instances}}.
#' @param K Number of folds.
#'
#' @return A character string uniquely identifying the model for this
#'   instance and training-fold configuration.
#'
#' @keywords internal
model_signature = function(inst_key, eval_token, methods, plan, K) {
  inst = plan$instances[[inst_key]]
  nf   = methods[[inst$method_idx]]$nuisance[[inst$name]]

  if (identical(inst$name, "__TARGET__")) {
    folds = idx_mod(eval_token$p, K)
  } else {
    offset = plan$inst_offset[[inst_key]]
    w = nf$train_fold
    folds = idx_mod(eval_token$p + offset + seq_len(w) - 1L, K)
  }
  folds = sort(unique(folds))
  fold_label = paste(folds, collapse = ",")

  paste(inst$struct_sig, fold_label, sep = "|F:")
}
