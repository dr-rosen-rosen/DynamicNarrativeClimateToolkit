# Dynamic Narrative Climate Toolkit (`dynclim`)

Python toolkit that turns free-text **safety-event narratives** into
organizational **safety-climate measures**, and builds the analysis-ready
event and operational datasets used in the accompanying study. Covers rail,
nuclear power, coal mining, and aviation (with exploratory PHMSA / healthcare
pipelines).

## What it does

1. **Data pipelines** (`dynclim/data/`) — download and standardize public U.S.
   federal safety data into per-industry event and operational parquet files:
   - MSHA (mining) accidents / production / violations — `msha_*_pipeline.py`
   - FRA (rail) accident + operational data — `rail_data_pipeline.py`
   - NRC (nuclear) Licensee Event Reports — `nrc_data_pipeline.py`
   - Aviation: NASA ASRS narratives, NTSB accidents, FAA AIDS, BTS T-100 —
     `asrs_raw_pipeline.py`, `aviation_data_pipeline.py`, `aids_data_pipeline.py`

2. **Safety-climate scoring** (`dynclim/features/safety/`, `dynclim/experiments/runner.py`)
   — the core capability. Each narrative is segmented, embedded
   (sentence-transformers), scored for sentiment (RoBERTa / SiEBERT / VADER) and
   semantic similarity to a safety-climate item bank, then composited and
   thresholded into a per-report climate score. The runner sweeps a **multiverse
   of scoring configurations** (embedding × sentiment × composite × thresholding),
   writing one `results.parquet` per configuration plus a `config_registry.csv`,
   to support specification-curve robustness analysis downstream.

3. **Feature layers** (`dynclim/features/`) — `linguistic/` (lexical/syntactic
   features) and `temporal/` (early-warning-signal features).

Processed outputs feed the R analysis repo (`dyn_nar_clim_mv_analysis`), which
runs the multiverse / cross-validation modeling and produces the manuscript
figures.

## Setup

Requires **Python 3.9**.

```bash
python3.9 -m venv .dynClimVenv
source .dynClimVenv/bin/activate
pip install -r requirements.txt
pip install -e .                       # editable install of the dynclim package
python -m spacy download en_core_web_sm
```

Key dependencies: `torch`, `sentence-transformers`, `transformers`, `spacy`,
`vaderSentiment`, `nltk`, `scikit-learn`, `pandas`, `pyarrow`.

Optional — Jupyter kernel and database-backed storage:
```bash
python -m ipykernel install --user --name=dynClimKern --display-name="Dynamic Climate Kernel"
# for DB options, create a .env with DB_TYPE / DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME
```

## Layout

```
dynclim/
  data/         # per-industry + source data pipelines
  features/     # safety-climate scoring, linguistic, temporal feature layers
  experiments/  # multiverse scoring runner
  utils/        # diagnostics, config helpers
```

## Data & reproducibility

Raw and processed data (`dynclim/data/`) and scoring checkpoints
(`dynclim/notebooks/checkpoints/`) are **not** tracked in git — they are shared
via the Zenodo deposit (see `DATA_MANIFEST.md` in the analysis repo for sources
and coverage). The state used for the published study is tagged
`paper1-submission`.
