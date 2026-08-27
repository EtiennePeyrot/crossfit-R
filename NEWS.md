# crossfit 0.1.4

* Simplified the single-method interface: `crossfit()` now returns `estimate`, `results`, and diagnostics directly. Method-indexed outputs remain available through `crossfit_multi()`.
* Added method-specific failure handling through `crossfit_failure_control()`, including control over whether panel errors fail a repetition, pruning after shared-fit failures, and the number of permitted failed repetitions.
* Improved nuisance-fit reuse by comparing complete function objects, including their closure environments, so learners with different captured states are not incorrectly treated as identical.
* Updated the documentation and examples to use the current output structure and clarified the terminology for evaluation windows, panels, and repetitions.


# crossfit 0.1.2

# crossfit 0.1.0

* Initial CRAN submission.
* General cross-fitting engine for nested/meta learners.
* Supports estimate and predict modes, DAG of nuisances, multiple fold allocation strategies.
