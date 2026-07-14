"""
NRC LER re-scrape (2026-07): separator + case fix
=================================================

Re-scrapes the known-good NRC event set with the fixed pipeline
(`get_text(separator=' ')` + preserved case — see nrc_data_pipeline.py and
project memory nrc-text-quality). The original 2026-02 scrape produced
word-glued, and the cleaning step lowercased, `event_text`; both defects
degrade sentiment and every spaCy-based lexical/syntactic feature.

Scrapes into a FRESH output dir (own checkpoint; resumable by rerunning
this script) and finishes with one retry pass over transient failures.
The original raw csv / checkpoint under data/raw/nrc/ are left untouched
as the as-scraped record. QA + the events.parquet swap into
data/processed/nrc/ happen separately after inspection.

Run (by file path):
    .dynClimVenv/bin/python dynclim/data/rescrape_nrc_events.py \\
        --out dynclim/data/raw/nrc/rescrape_2026-07
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

log = logging.getLogger("nrc_rescrape")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dynclim/data/raw/nrc/rescrape_2026-07"))
    ap.add_argument("--event-list", default=None,
                    help="csv/xlsx of event numbers; default = the event_num "
                         "set of the original raw scrape (known-good 14,110)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nrc_pipe", Path(__file__).parent / "nrc_data_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.event_list:
        ev_list = Path(args.event_list)
    else:
        # Known-good set = events the 2026-02 scrape succeeded on
        src = REPO / "dynclim/data/raw/nrc/nrc_events_raw.csv"
        nums = pd.read_csv(src, usecols=["event_num"])["event_num"]
        nums = nums.dropna().astype(str).drop_duplicates()
        if args.limit:
            nums = nums.head(args.limit)
        ev_list = out / "event_list.csv"
        pd.DataFrame({"event_num": nums}).to_csv(ev_list, index=False)
        log.info("event list: %d known-good events -> %s", len(nums), ev_list)

    pipeline = mod.NRCDataPipeline(output_dir=out)
    pipeline.scrape_events(ev_list)                      # main pass
    df = pipeline.scrape_events(ev_list, retry_failed=True)  # transient retry
    df = pipeline.clean_events(df)
    pipeline.save(df, "events_rescraped")
    log.info("DONE: %d events in %s/events_rescraped.parquet", len(df), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    main()
