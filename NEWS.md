# crossfit 0.1.3

* `crossfit()` now returns a direct single-method result with `estimate` and
  `results` components instead of nesting outputs under an automatic method
  name. Method names remain part of `crossfit_multi()` output.
* Added `crossfit_failure_control()` and the method-level `failure_control`
  setting. Panel errors can optionally be passed to a missingness-aware panel
  aggregator, while failed-fit pruning is now opt-in and defaults to `FALSE`.
* Replaced the top-level `max_fail` argument with the method-specific
  `failure_control$max_failed_repetitions` setting.
* Function reuse now compares complete function objects, including closure
  environments, so learners with different captured state are not conflated.

# crossfit 0.1.2

# crossfit 0.1.0

* Initial CRAN submission.
* General cross-fitting engine for nested/meta learners.
* Supports estimate and predict modes, DAG of nuisances, multiple fold allocation strategies.
