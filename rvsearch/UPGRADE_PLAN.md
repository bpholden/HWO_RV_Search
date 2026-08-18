# Upgrading rvsearch to radvel ≥ 1.4 (issue #196)

## 1. Why the `radvel<=1.3.8` pin exists

radvel **1.4.0** replaced the `Parameters`-dict evaluation path with a NumPy
**`Vector`** (`radvel.model.Vector`, introduced in commit `14c8395`, 2020-04-10,
first released in `v1.4.0`). From 1.4.0 onward:

* `RVModel` / `GeneralRVModel` own a `Vector` built once at construction
  (`radvel/model.py`, `GeneralRVModel.__init__`).
* Every `Likelihood` shares that same vector (`self.vector = model.vector`);
  `CompositeLikelihood` asserts `like.vector is vector`.
* All numerics read the vector, not the dict:
  * values → `vector.vector[idx][0]` (`_standard_rv_calc`, `RVLikelihood.residuals`,
    `errorbars`, `logprob`)
  * **vary flags** → `vector.vector[idx][1]`, cached in `self.vary_params` by
    `Likelihood.list_vary_params()`
  * **mcmcscale** → `vector.vector[idx][2]` (`radvel/mcmc.py`)
  * **linear** → `vector.vector[idx][3]`
* `Parameters` is now a *view* that is refreshed **from** the vector by
  `Vector.vector_to_dict()` (called at the end of `maxlike_fitting`). Writing to
  the dict pushes nothing back unless you explicitly call
  `Vector.dict_to_vector()`.

rvsearch drives radvel almost entirely by mutating `post.params[...]` **after**
the posterior has been built. Under radvel ≥ 1.4 every one of those writes is a
no-op. Nothing raises — the search just stops doing anything.

### Empirical confirmation

Environment: Python 3.13.6, radvel 1.6.1, numpy 2.5.1, rvsearch 0.3.3 (`master`),
`example_data/HD128311.csv`.

```
nvary before: 4 ['tc1', 'k1', "jit_b'j'", "jit_b'k'"]
nvary after params['k1'].vary = True / params['tc1'].vary = True: 4      # no-op
logprob unchanged after params['per1'].value = 300; params['k1'].value = 50: True
vector per1 = 100.0   params per1 = 300.0                                # desynced
after maxlike_fitting: per1 = 100.0, k1 = 0.0                            # write lost
# same writes via the vector API:
nvary via vector: 4   logprob: -676.38   (vs -44531.64)                  # works
```

End-to-end, `Periodogram.per_bic()` over 200 trial periods on HD 128311:

```
n unique ΔBIC values: 1 of 200
ΔBIC min/max: 1.82e-12  2.27e-12
```

**The ΔBIC periodogram is identically zero at every trial period.** Every grid
point is fit from the same unchanged vector, so `baseline_bic - post.bic()`
collapses to numerical noise. `run_search()` still exits cleanly and reports
`num_planets found: 1` (the placeholder planet with `k=0`) for a system with two
large, well-known planets. The only visible hints are a
`divide by zero encountered in log10` warning and
`Polyfit may be poorly conditioned` from the eFAP fit on the degenerate array.

This is the worst failure mode available: **silent, wrong science**, not a
traceback. That is almost certainly why the pin was added rather than the API
being migrated.

### The pin is now a hard blocker, not just stale

`pip install radvel==1.3.8` fails outright on modern Python/setuptools:

```
AttributeError: 'dict' object has no attribute '__NUMPY_SETUP__'
    and no __dict__ for setting new attributes
error: metadata-generation-failed
```

The pre-1.5 `setup.py` uses the old `__NUMPY_SETUP__` build hack. `1.4.9` fails
the same way. So on Python ≥ 3.11 rvsearch is currently **uninstallable with a
satisfiable dependency set** — `requirements.txt` names a version of radvel that
cannot be built. radvel ≥ 1.5.7 ships cibuildwheel binary wheels and installs
cleanly; radvel 1.6.1 imports and runs fine alongside rvsearch's other deps
(astroML, PyAstronomy, pathos, tqdm all install on 3.13 / numpy 2.5).

---

## 2. Inventory of what has to change

### 2a. Broken: post-construction `params` writes (the whole bug)

The rule: **writes to `params` before the `Posterior` is constructed are fine**
(`Vector.dict_to_vector()` runs at construction and picks them up). Writes
*after* construction are lost.

Safe today (pre-construction, leave alone):
* [utils.py:87-91](rvsearch/utils.py#L87-L91) — `initialize_default_pars`
* [search.py:218-226](rvsearch/search.py#L218-L226), [search.py:266-269](rvsearch/search.py#L266-L269) — `add_planet` / `sub_planet` operate on `new_params` before `initialize_post`

Broken (post-construction), by call site:

| File | Lines | What it tries to do | Consequence under ≥1.4 |
|---|---|---|---|
| [periodogram.py:170-184](rvsearch/periodogram.py#L170-L184) | 15 `.vary` writes | fix Keplerian params for the baseline fit | baseline fit varies the wrong set |
| [periodogram.py:193-204](rvsearch/periodogram.py#L193-L204) | 8 `.vary` writes | free K/tc, fix period, set eccentricity mode | the search never frees K or tc |
| [periodogram.py:219-221](rvsearch/periodogram.py#L219-L221), [228-230](rvsearch/periodogram.py#L228-L230), [237-240](rvsearch/periodogram.py#L237-L240) | `.value` writes in `_fit_period` | step the trial period, reset to defaults, retry seeds | **flat periodogram** — the headline failure |
| [search.py:134-158](rvsearch/search.py#L134-L158) | `trend_test` | three nested dvdt/curv models | all three BICs identical → verdict is always "Flat" |
| [search.py:165-180](rvsearch/search.py#L165-L180) | apply trend verdict | writes chosen dvdt/curv back | trend never enters the model |
| [search.py:286-300](rvsearch/search.py#L286-L300) | `fit_orbit` vary set | free all planet params (respect setup fixes) | final fit varies the wrong set |
| [search.py:320-322](rvsearch/search.py#L320-L322), [335-346](rvsearch/search.py#L335-L346) | `fit_orbit` polish grid | reset + step period, restore best fit | polish step is a no-op |
| [search.py:459-481](rvsearch/search.py#L459-L481) | `run_search` | seed new planet from periodogram best fit, set fix/free | new planet starts at defaults, not the peak |
| [search.py:495-498](rvsearch/search.py#L495-L498) | jitter sign flip | flip negative jitter | no-op; harmless numerically (`jit**2`) but leaves dict/vector inconsistent |
| [search.py:527-529](rvsearch/search.py#L527-L529) | e/w `mcmcscale = 0.005` | tighten MCMC proposal scale | ignored; `mcmc.py` reads `vector[...][2]` |
| [search.py:673-677](rvsearch/search.py#L673-L677) | injection setup | fix planet params before injection | injected-signal recovery tests are invalid |

Stale `.vary` **reads** — these read the dict, which `vector_to_dict()` never
refreshes for `vary`, so after the migration they will read whatever rvsearch
last wrote and can drift from the vector:
[search.py:218](rvsearch/search.py#L218), [220](rvsearch/search.py#L220),
[266](rvsearch/search.py#L266), [268](rvsearch/search.py#L268),
[518](rvsearch/search.py#L518), [588](rvsearch/search.py#L588),
[616-618](rvsearch/search.py#L616-L618),
[370-373](rvsearch/search.py#L370-L373).

### 2b. Plotting: `PeriodModelPlot` is a copy of radvel 1.3.8's `MultipanelPlot.__init__`

[plots.py:73-190](rvsearch/plots.py#L73-L190) subclasses
`radvel.plot.orbit_plots.MultipanelPlot` but never calls `super().__init__()` —
it reimplements the 1.3.8 body. radvel HEAD's version changed in ways that now
diverge:

* `self.post = copy.deepcopy(post)` upstream vs. [plots.py:83](rvsearch/plots.py#L83) `self.post = self.search.post` — rvsearch mutates the live search posterior.
* Upstream does `self.post.params = synthparams` **followed by `self.post.vector.dict_to_vector()`**; [plots.py:130-131](rvsearch/plots.py#L130-L131) does `self.post.params.update(synthparams)` with no vector refresh. (`update()` on the `Parameters` OrderedDict does not change `.basis`, so the model curve happens to still come out right — but the dict and vector are inconsistent and this is fragile.)
* `phase_nrows`/`phase_ncols` handling was buggy in 1.3.8 (the `else` branches were missing) and was fixed upstream; [plots.py:95-98](rvsearch/plots.py#L95-L98) still carries the 1.3.8 bug, so explicit values are ignored.
* Upstream `plot_residuals` now plots `self.rawresid` against a zero line, where 1.3.8 plotted `self.resid` against `self.slope`. rvsearch sets both `rawresid` and `resid` ([plots.py:138-142](rvsearch/plots.py#L138-L142)) so inherited methods won't crash, but **the residual panel's meaning changes** once inherited code runs.
* Upstream `plot_phasefold` guards the `hasattr(self.post, 'derived')` block on `self.status` being present; rvsearch never sets `self.status`.

No crashes here — this is a "re-sync and re-review the figures" task, not a hard break.

### 2c. Unaffected (verified compatible)

* `radvel.utils`: `initialize_posterior`, `working_directory`, `Msini`,
  `semi_major_axis`, `round_sig`, `sigfig`, `bintels` — signatures unchanged.
* `radvel.fitting.maxlike_fitting(post, verbose=, method=)` — unchanged.
* `radvel.mcmc(...)` — all rvsearch kwargs still present; changes are purely
  additive (`headless`, `progress_callback`).
* `radvel.kepler.rv_drive(t, orbel)` — unchanged.
* `radvel.basis._copy_params` — still present (still private).
* `radvel.prior.{PositiveKPrior, EccentricityPrior, UserDefinedPrior}` —
  constructors unchanged. `Prior.__call__` gained a `vector` argument, but
  rvsearch only *constructs* priors; `UserDefinedPrior` still calls the
  user function with a plain list of values, so
  [`GaussianDiffFunc`](rvsearch/utils.py#L18) needs no change.
* `radvel.likelihood.RVLikelihood` / `CompositeLikelihood` / `Posterior`
  constructors — unchanged as called from
  [utils.py:127-136](rvsearch/utils.py#L127-L136).
* `Vector` index layout is positional (`5*n` blocks + `dvdt`/`curv` +
  gamma/jit), so `add_planet`'s unusual key ordering is safe.
* `basis.v_to_synth` returns a copy — not destructive.
* Pre-existing bug, unrelated to radvel: [utils.py:110](rvsearch/utils.py#L110)
  `initialize_post` calls `data.tel` / `data.time` as attributes, which fails
  for a DataFrame read with a `jd` column even though lines 114-115 try to
  handle that case (the rename happens *after* the attribute access).
* [utils.py:31-35](rvsearch/utils.py#L31-L35) `reset_params` has no callers —
  dead code.

---

## 3. Plan

Goal: `requirements.txt` says `radvel>=1.6.1`, and rvsearch reproduces its
radvel-1.3.8 scientific results.

The migration is mechanical but there is **no test suite** (`rvsearch/tests/`
does not exist), and the failure mode is silent. So step 1 is a baseline, not
code.

### Phase 0 — capture a golden baseline under radvel 1.3.8 *(do this first)*

Without this there is no way to tell a correct migration from a plausible-looking
wrong one.

1. Build a reference env pinned to radvel 1.3.8. Modern pip can't build it, so
   use one of: an existing working conda/virtualenv on the old stack;
   Python 3.9 + `pip install "setuptools<60" "numpy<1.24"` then
   `pip install --no-build-isolation radvel==1.3.8`; or a Docker image on
   `python:3.9`.
2. Run and archive, as JSON/CSV, for at least `example_data/HD128311.csv` plus
   one multi-instrument and one trend-dominated dataset:
   * the full `pers` / ΔBIC periodogram array for each search iteration
   * `search.bic_threshes`, `search.best_bics`, `search.eFAPs`
   * final `post.params` values and `num_planets`
   * `trend_test()`'s three BICs and its verdict
   * one `inject.py` recovery grid (small, e.g. 8×8) — this is the end-to-end
     integrity check
   Use `mcmc=False` and a fixed `num_pers` throughout so runs are deterministic
   and fast.
3. Commit these as fixtures under `rvsearch/tests/data/`.

**Verifiable target:** a `pytest` test that loads each fixture, reruns the
search, and asserts periodogram arrays match to ~1e-6 relative and
`num_planets` matches exactly. It must **pass on 1.3.8** before any migration
work starts.

### Phase 1 — introduce a sync helper

Add to `rvsearch/utils.py`:

```python
def sync_to_vector(post):
    """Push post.params (value/vary/mcmcscale/linear) into post.vector and
    refresh the cached vary-parameter index lists.

    Required for radvel >= 1.4, where all numerics read the Vector and writes
    to post.params are otherwise ignored.
    """
    post.vector.dict_to_vector()
    post.vector.vector_names()
    post.list_vary_params()
    post.likelihood.list_vary_params()
    return post
```

Two reasons to prefer this over rewriting ~90 call sites into
`post.vector.vector[post.vector.indices['per1']][0] = per`:

* it keeps the diff small and the intent readable — the existing
  `params[...].vary = False` lines stay as documentation of what the search is
  doing;
* it is one place to audit, so "did we forget a sync?" is answerable.

Note both `post.list_vary_params()` **and**
`post.likelihood.list_vary_params()` are required — radvel's own
`fitting.model_comp` does exactly this pair, because `Posterior` and its
`Likelihood` cache `vary_params` independently.

For the hot inner loop only ([periodogram.py:216-248](rvsearch/periodogram.py#L216-L248),
~4600 iterations × workers), measure `sync_to_vector` cost. `dict_to_vector`
is a Python loop over all keys; if it dominates, specialise that one loop to
direct vector writes for the two parameters that actually change (`per`, `k`,
`tc`) and hoist the `.vary` setup out of the loop, where it already is.

### Phase 2 — apply the sync at each broken site

Work file by file, running the Phase 0 tests against the 1.3.8 baseline after
each file.

1. **`periodogram.py`** — highest value, fixes the flat periodogram.
   * after the `.vary` block at [170-184](rvsearch/periodogram.py#L170-L184), sync before `maxlike_fitting`/`bic()`
   * after the `.vary` block at [193-204](rvsearch/periodogram.py#L193-L204), sync once (outside the loop)
   * inside `_fit_period`, sync after each of the three reset/seed blocks
     ([219-221](rvsearch/periodogram.py#L219-L221), [227-230](rvsearch/periodogram.py#L227-L230), [236-240](rvsearch/periodogram.py#L236-L240)) before the corresponding `maxlike_fitting`
   * **Verifiable target:** ΔBIC array on HD 128311 matches the 1.3.8 fixture;
     `len(np.unique(bic)) > 1` as a cheap smoke assertion.
2. **`search.py`**
   * `trend_test` — sync after each of the post1/post2/post3 vary blocks and
     before each `bic()`; sync after the verdict block at [165-180](rvsearch/search.py#L165-L180).
     **Target:** the three BICs differ, and the verdict matches the fixture on a
     dataset with a real trend.
   * `fit_orbit` — sync after [286-300](rvsearch/search.py#L286-L300), after the polish
     reset/seed at [320-322](rvsearch/search.py#L320-L322), after [335-346](rvsearch/search.py#L335-L346).
   * `run_search` — sync after the seed/fix block at [459-481](rvsearch/search.py#L459-L481).
   * `inject_recover` — sync after [673-677](rvsearch/search.py#L673-L677).
     **Target:** the recovery grid fixture matches.
   * mcmcscale at [527-529](rvsearch/search.py#L527-L529): these values are set on
     `self.post` but MCMC runs on `logpost`, rebuilt from
     `to_any_basis(...)` at [533-535](rvsearch/search.py#L533-L535). Set the scales on
     `logpost` (or on `logparams` before `initialize_post`) instead, and sync.
     Confirm with `logpost.vector.vector[idx][2] == 0.005`.
   * Replace the stale `.vary` reads listed in §2a with reads through the
     vector, e.g.
     `bool(post.vector.vector[post.vector.indices['dvdt']][1])`, so dict drift
     can't change control flow.
3. **`utils.py`** — `initialize_post` currently assigns
   `likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]`
   ([131-132](rvsearch/utils.py#L131-L132)) *after* `RVLikelihood.__init__` has
   already run `dict_to_vector()`. Add a sync after `Posterior` construction
   (before `post.priors = priors`) so the `linear=True, vary=False` gamma flags
   reach `vector[...][3]` and `[1]` — the analytic-gamma path in
   `RVLikelihood.residuals`/`logprob` gates on exactly those two slots.
   Also fix the `data.tel`/`data.time` attribute-access bug at
   [110](rvsearch/utils.py#L110) (move the `jd` → `time` rename above it) and
   delete the dead `reset_params`.

### Phase 3 — re-sync `PeriodModelPlot` with upstream `MultipanelPlot`

1. Diff [plots.py:73-190](rvsearch/plots.py#L73-L190) against
   `radvel/plot/orbit_plots.py::MultipanelPlot.__init__` at 1.6.1 and adopt:
   `copy.deepcopy(post)`; `self.post.params = synthparams` +
   `self.post.vector.dict_to_vector()`; the `phase_nrows`/`phase_ncols` `else`
   branches; `self.status = None`.
2. Better: call `super().__init__(...)` and override only what
   `PeriodModelPlot` genuinely adds (`search`, `pers`, `periodograms`,
   `bic_threshes`, `fap`, `runners`, `summary_ncols`, the summary-panel
   geometry). That removes the copy-paste drift permanently. Check first whether
   any of the ~15 duplicated attribute assignments differ in value from
   upstream's defaults — if several do, keep the explicit body and just sync it.
3. Decide deliberately whether rvsearch's residual panel should follow
   upstream's new `rawresid`-vs-zero convention or keep the
   `resid`-vs-`slope` convention. They are different plots; pick one and say so
   in the docstring.
4. **Verifiable target:** visual review of `orbit_plot*.pdf` and
   `summary_plot.pdf` for HD 128311 against 1.3.8 output. Not automatable;
   budget a human look.

### Phase 4 — dependency metadata and CI

1. `requirements.txt`: `radvel<=1.3.8` → `radvel>=1.6.1`. Pin the *floor* at
   1.6.1, not a range — 1.4.x/1.5.x cannot be built on modern Python, so
   allowing them is a trap.
2. `setup.py` reads `requirements.txt` verbatim into `install_requires`, which
   also drags in `sphinx_rtd_theme` as a runtime dep. Move the docs deps to a
   `docs/requirements.txt` or an extra while you're in there.
3. Add `python_requires=">=3.10"`.
4. Add CI (there is none) running the Phase 0 fixture tests on the lowest and
   highest supported Python.
5. Update the ReadTheDocs install instructions, which currently imply the old
   radvel.
6. Close #196 referencing the ΔBIC evidence above, so the next person doesn't
   have to re-derive why the pin existed.

### Phase 5 — recalibration check (do not skip)

rvsearch's detection thresholds (`fap=0.001`, the ΔBIC floor, the eFAP
power-law extrapolation in [plots.py](rvsearch/plots.py) /
[periodogram.py](rvsearch/periodogram.py)) were tuned against radvel 1.3.8's
optimiser behaviour. I checked the one thing most likely to shift absolute BIC —
the `log(sqrt(2*pi*sigz))` normalisation on the analytically-marginalised gamma
— and it is present identically in both 1.3.8 and 1.6.1 (it landed around
v1.2.7), so that is **not** a concern.

Still worth confirming once the fixtures pass: run the injection/recovery grid
at production resolution on one or two well-characterised systems and check the
50% completeness contour against a published rvsearch result. If it moves, the
thresholds — not the migration — need revisiting.

---

## 4. Effort and risk

* Phase 0 is the real cost: ~1-2 days, mostly wrestling a radvel 1.3.8 env into
  existence. Do it in Docker.
* Phases 1-2 are ~1 day of mechanical edits across three files, but only
  because Phase 0 makes them checkable.
* Phase 3 is a few hours plus a human figure review.
* Phases 4-5 are a few hours plus one long compute run.

Main risk: skipping Phase 0. Every symptom of this bug class is silent, so
"it runs and produces a plot" is not evidence of anything. If a 1.3.8 baseline
truly cannot be built, the fallback is to validate against **published** rvsearch
results (the California Legacy Survey planet catalogue) rather than against
nothing — slower and coarser, but still a real check.

Second risk: `sync_to_vector` in the periodogram inner loop is O(n_params) per
call on a ~4600-iteration loop. Measure before and after; if the search gets
meaningfully slower, specialise that one loop to direct vector indexing.
