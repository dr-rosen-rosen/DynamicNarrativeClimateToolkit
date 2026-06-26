# Environment, Reproducibility & Packaging Plan

> Scope: the Python toolkit (`DynamicNarrativeClimateToolkit/dynclim`) and the R
> analysis repo (`data_anlaysis/dyn_nar_clim_mv_analysis`). Goal: a setup that
> (a) makes day-to-day work reliable, (b) lets the codebase be *shared and
> reused*, and (c) supports *freezing a known-good version per publication*.
> Written 2026-06-21.

---

## 0. Executive summary

The environment is **not actually broken** — there is a working canonical Python
venv (`.dynClimVenv`). The recurring pain is a *discipline and tooling* gap, not
a missing dependency:

- **Python**: a working venv exists but reproducibility rests on a hand-curated
  `requirements.txt`, an editable install, **no lock file**, **no
  `pyproject.toml`**, and an `__init__.py` that **eagerly imports the heavy ML
  stack** (torch / sentence-transformers) for every consumer — including the
  lightweight feature work that doesn't need it. Running the *system* Python
  3.12 (NumPy 2.3 / SciPy 1.12 mismatch) is what produces the import crash.
- **R**: **no `renv`, no lock file, no version pinning at all**, and
  **hard-coded absolute `/Users/michaelrosen/...` paths in 10+ scripts**. This
  is the larger reproducibility liability of the two — the compiled
  `glmmTMB`/`TMB`/`Matrix`/`lme4` chain is version-sensitive and will fit
  *differently* across versions.
- **No per-publication freeze mechanism** on either side.

The plan below is phased: **Phase 0** stabilizes today's workflow with near-zero
risk; **Phase 1** introduces lock-based reproducibility (uv + renv); **Phase 2**
adds the per-publication freezing protocol and optional containerization.

---

## 1. Current state (as inventoried)

### 1.1 Python toolkit

| Aspect | Finding |
|---|---|
| Canonical env | **`./.dynClimVenv`** — Python 3.9.13, numpy 1.24.3, scipy 1.10.1, spacy 3.6.1, en_core_web_sm 3.6.0, torch 2.0.1, sentence-transformers 3.0.1. `import dynclim` works; `python -m …` works. |
| Broken env | **System** `/Library/Frameworks/Python.framework/Versions/3.12` — numpy 2.3.1 + scipy 1.12.0 (compiled for numpy<1.29) → `numpy.core.multiarray failed to import`. Do not use it for this project. |
| Dep declaration | `setup.py` (v0.2.0, `python_requires>=3.9`) + heavily pinned `requirements.txt` (173 lines). **No `pyproject.toml`, no lock file.** |
| Install mode | Editable via `.egg-link` (not distributable as-is). |
| Coupling problem | `dynclim/__init__.py:12` eagerly does `from dynclim.features.safety import Processor` → `unified_processor.py:26` `from sentence_transformers import SentenceTransformer`. **Every `import dynclim` drags in torch + transformers + scipy**, which is exactly what makes the NumPy-2 mismatch fatal and forces light users onto the heavy stack. |
| Dependency shape | Two layers: a **heavy ML stack** (torch, transformers, sentence-transformers, sklearn) used only by the *climate scoring* path, and a **light analysis stack** (spacy, pandas, numpy, pyarrow, nltk, vaderSentiment) used by the new linguistic/temporal feature work. They are currently undivided. |
| spaCy model | en_core_web_sm 3.6.0 pinned by URL in requirements (good), venv-local. |

### 1.2 R analysis repo

| Aspect | Finding |
|---|---|
| Reproducibility tooling | **None.** No `renv.lock`, no `renv/`, no `DESCRIPTION`, no `.Rprofile` at repo root, no `install.R`. |
| R version | 4.5.1 installed (pinnable, not pinned). |
| Critical compiled deps | `glmmTMB` (→ TMB, Rcpp), `lme4` (→ Rcpp, Matrix), `ordinal`, `arrow`. Optimizer defaults / likelihood computations **change across versions** → different model fits. |
| Other deps | tidyverse, broom.mixed, furrr, slider, pROC, QCA (optional), patchwork, here, lubridate, jsonlite, quarto. |
| Path hardcoding | **Critical.** Absolute `/Users/michaelrosen/.../dynclim/...` paths to `events.parquet`, ops/covariate parquets, and date-stamped checkpoint dirs (`rail_04-14-2026`, `nrc_04-14-2026`, `asrs_05-01-2026`, `msha_06-01-2026`) embedded in `5_manuscript_plots.R` and others. Breaks on any other machine. |
| Partial manifest | `common/feature_layers.R` *can* write an R+package-version JSON manifest, but it's optional and not run automatically. |
| Data contract | Parquet keyed by event id (`eid`), read via `arrow`. Stable and good — this is the clean seam between the two repos. |

---

## 2. The cross-language seam

The two repos meet at **parquet files on disk keyed by `eid`** (and the
checkpoint dirs with climate configs). This is a good, stable contract — keep
it. The only fix needed at the seam is **path indirection**: both sides should
resolve a single root from the environment, not hard-code `/Users/michaelrosen`.

Recommended single source of truth:

```
DYN_CLIM_ROOT=/Users/michaelrosen/Documents/dev/DynamicNarrativeClimateToolkit
```

- Python: already runs relative to the repo; keep using repo-relative paths.
- R: replace the hard-coded `CFG_DIR_*` / `*_EVENTS` constants with
  `file.path(Sys.getenv("DYN_CLIM_ROOT"), "dynclim/data/processed/…")`, set in
  a committed `.Renviron` template + a local `.Renviron` (gitignored).

---

## 3. Recommended target architecture

### 3.1 Python — migrate to **uv** + `pyproject.toml` + dependency groups

`uv` gives fast, fully-resolved, **hash-pinned `uv.lock`** files and one-command
reproducible installs (`uv sync`), with Python-version pinning. It replaces the
manual `requirements.txt` + venv juggling and is the right modern default.

**Split dependencies so light work doesn't require the ML stack** (this is the
structural fix for the import crash and for sharing):

```toml
# pyproject.toml (sketch)
[project]
name = "dynclim"
version = "0.3.0"
requires-python = ">=3.9,<3.13"
dependencies = [          # CORE — light, always needed
  "numpy<2", "pandas", "pyarrow", "spacy==3.6.1",
  "nltk", "vaderSentiment", "blingfire", "python-dotenv", "tqdm",
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl",
]

[project.optional-dependencies]
climate = ["torch==2.0.1", "sentence-transformers==3.0.1", "transformers==4.44.0", "scikit-learn", "scipy<1.13"]
db      = ["sqlalchemy", "pymongo", "psycopg2-binary"]
viz     = ["matplotlib", "seaborn", "plotly", "streamlit"]
dev     = ["pytest", "pytest-cov", "black", "ruff", "mypy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Then:
- `uv sync` → core only (linguistic/temporal feature work, the new driver).
- `uv sync --extra climate` → adds the scoring stack.
- `uv lock` → commit `uv.lock` for exact reproducibility.

**Make the heavy imports lazy** so `import dynclim` (or, better, importing a
subpackage) never forces torch:
- Remove the eager `from dynclim.features.safety import Processor` from
  `dynclim/__init__.py`; expose `Processor` via a lazy `__getattr__` (PEP 562)
  or require `from dynclim.features.safety import Processor` explicitly.
- Net effect: `from dynclim.features.linguistic import ...` works with the
  **core** install alone — no NumPy-2 trap, no 2 GB of torch for feature work.
- Keep `numpy<2` pinned until `scipy>=1.13` (numpy-2-compatible) is adopted
  across the climate stack; that pin is what actually prevents the crash.

### 3.2 R — adopt **renv** + path indirection

- `renv::init()` then `renv::snapshot()` → commit `renv.lock` (pins R 4.5.1 + all
  package versions, incl. the glmmTMB/TMB/Matrix chain). Reproduce elsewhere with
  `renv::restore()`.
- Add a committed `.Rprofile` that activates renv and a `setup.R` bootstrap that
  checks packages + validates `DYN_CLIM_ROOT`.
- Replace hard-coded absolute paths with `Sys.getenv("DYN_CLIM_ROOT")`-based
  `file.path(...)`. Ship a `.Renviron.example`.
- Auto-write the existing version manifest (`common/feature_layers.R`) next to
  every multiverse/CV output.

> Caveat: `renv` pins *package* versions but not the compiler/BLAS. For the
> compiled `glmmTMB`/`TMB` chain, exact cross-machine fit reproducibility may
> still need a container (see Phase 2). On the same machine, renv is sufficient.

### 3.3 Per-publication freezing protocol

Treat a publication as a **tagged, lockfile-bearing commit on both repos**:

1. Tag both repos: `git tag paper2-submission` (and `-revision`, `-final`).
2. Ensure `uv.lock` (Python) and `renv.lock` (R) are committed at that tag.
3. Record the exact spaCy model + checkpoint-dir dates used (the toolkit already
   date-stamps checkpoints — capture which ones the paper used).
4. Optionally deposit a snapshot (tagged source + locks + a data manifest) to
   Zenodo/OSF and add a `CITATION.cff`.
5. To reproduce paper N: `git checkout paperN-final` in both repos →
   `uv sync --extra climate` / `renv::restore()` → set `DYN_CLIM_ROOT` → run.

A frozen branch per paper (`freeze/paper2`) is an alternative to tags if you
expect to apply reviewer-driven patches to a frozen line independently of `main`.

---

## 4. Phased rollout

### Phase 0 — Stabilize now (hours, near-zero risk, no version changes)
- [ ] Document the canonical venv in README: *use `./.dynClimVenv`, never system
      python3.12*. Add a one-line activate snippet.
- [ ] Pin the interpreter expectation: add `.python-version` (3.9) to the repo.
- [ ] R: run `renv::snapshot()` to **capture today's working versions** into
      `renv.lock` immediately (even before the full renv adoption) — this is the
      single highest-value, lowest-risk action; it freezes the model-fitting
      stack as it currently is.
- [ ] R: write down the current `DYN_CLIM_ROOT` and the checkpoint-dir dates the
      paper-1 analysis used, so paper 1 stays reproducible.

### Phase 1 — Lock-based reproducibility (1–2 days)
- [ ] Python: author `pyproject.toml` with core vs. `climate`/`db`/`viz`/`dev`
      groups; generate and commit `uv.lock`; verify `uv sync` reproduces the
      feature pipeline and `uv sync --extra climate` reproduces scoring.
- [ ] Python: make `dynclim/__init__.py` lazy so core install runs the
      linguistic/temporal drivers without torch. Add a CI/smoke test:
      `uv run python -m dynclim.features.linguistic.compute_linguistic_features
      --industry aviation --limit 50 --no-save`.
- [ ] R: full `renv` adoption (`.Rprofile`, `setup.R`), externalize all absolute
      paths to `DYN_CLIM_ROOT`, ship `.Renviron.example`.

### Phase 2 — Publication freezing & portability (as needed per paper)
- [ ] Adopt the tag + lock protocol in §3.3; add `CITATION.cff` to both repos.
- [ ] Optional container for the compiled R chain (rocker/r-ver pinned to 4.5.1 +
      renv) and/or a Python image, for true cross-machine reproduction.
- [ ] Optional: a thin top-level repo or Makefile that documents the two-repo
      workflow and the data contract in one place.

---

## 5. Decisions (recorded 2026-06-21)

1. **Python packaging → adopt uv.** Migrate to `pyproject.toml` + `uv.lock` with
   the dependency groups in §3.1. The Python toolkit is **production-bound**
   (see #5), so the `core` (+ `climate` if scoring runs in prod) groups are the
   production install; `viz`/notebook/paper deps stay optional/dev.
2. **NumPy → status quo for now.** Pin `numpy<2` (with `scipy<1.13`); revisit a
   NumPy-2 upgrade of the whole climate stack after the first series of papers.
3. **R containerization → deferred.** Same-machine `renv` is enough for now.
   Pin this for later, to be done together with the NumPy-2 major update (#2).
4. **Freeze granularity → asymmetric (leverages #5).**
   - **Python toolkit: tags only** (don't fork production history per paper). Tag
     the toolkit commit each paper used so a paper can pin a
     `(toolkit_tag, analysis_tag)` pair.
   - **R repo: `freeze/paperN` branch per paper**, with `-submission` /
     `-revision` / `-final` tags inside the branch. (R is papers-only, so there's
     no production line to protect and per-paper isolation is the natural fit.)
   - Universal rule: the committed lock file at the tagged point is what
     guarantees reproducibility; the branch only decides where future commits land.
   - Lighter fallback if branch upkeep isn't wanted: tags-by-default, cut a
     `freeze/paperN` branch lazily off the submission tag only if a revision needs
     new commits.
5. **Repos stay separate.** No umbrella/submodule repo. Rationale: moving toward
   **production the R side won't be needed** (papers only); the Python toolkit is
   the production artifact. The `(toolkit_tag, analysis_tag)` pairing from #4
   covers the freeze-protocol need that an umbrella repo would otherwise serve.
