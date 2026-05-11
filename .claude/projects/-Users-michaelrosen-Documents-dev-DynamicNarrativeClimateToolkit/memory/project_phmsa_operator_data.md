---
name: PHMSA Operator Data — Structure and Integration Roadmap
description: How PHMSA annual operator data links to the incident/accident event data; what metrics are available; implementation notes for future operator data integration
type: project
---

PHMSA operator data is the operational covariate layer for the event pipeline — analogous to FRA operational data in the rail pipeline. Not yet integrated; this note describes what's there and how to do it.

**Why:** Needed to compute incident rates as a denominator (incidents per 1,000 pipeline miles), and to attach operator-level exposure variables for modeling.

**How to apply:** When building operator data integration into `phmsa_data_pipeline.py`, follow this note for join logic, file structure, and gotchas.

---

## Event Data (already built)

Pipeline: `dynclim/data/phmsa_data_pipeline.py` → `PHMSADataPipeline`
Output columns include: `OPERATOR_ID` (as `operator_raw_id`), `event_date`, `pipeline_type`, `total_fatalities`, `total_injuries`, `total_cost`, `narrative`
Join key into operator data: `OPERATOR_ID` (integer) + `year(event_date)`

---

## Operator Data — Directory Overview

Location: `dynclim/data/raw/phmsa/operator_data/`

| Directory | Pipeline type | Year coverage | File structure |
|---|---|---|---|
| `annual_gas_distribution_2010_present 2/` | Gas distribution | 2017–2023 | One CSV per year (`GD AR 2023.csv`) |
| `annual_gas_transmission_gathering_2010_present/` | Gas transmission | 2017–2025 | ~14 CSVs per year, one per operator segment (must concat) |
| `annual_hazardous_liquid_2010_present/` | Hazardous liquid | 2017–2024 | Multiple part-files per year (`Part A to E`, `Part F to G`, etc.) — Part A has total pipeline mileage |
| `annual_liquefied_natural_gas_2010_present/` | LNG | 2017+ | One CSV per year; minimal columns (contact info only, no useful metrics) |
| `annual_underground_natural_gas_storage_2017_present/` | Underground storage | 2017+ | One CSV per year; minimal columns |

Also present but require older file formats (xlrd): `annual_gas_distribution_1970_2003/`, `annual_gas_distribution_2004_2009/`, `annual_gas_transmission_gathering_1970_2000/`, `annual_gas_transmission_gathering_2001_2009/`, `annual_hazardous_liquid_2004_2009/` — skip for now.

---

## Join Key

`OPERATOR_ID` (integer) is the primary key across **all** operator files and **all** event files.
Join pattern: `event.OPERATOR_ID == operator.OPERATOR_ID AND operator.REPORT_YEAR == year(event.event_date)`

Validated match rate: ~77% of hazardous liquid event operators appear in the 2022 annual operator file. Unmatched operators are likely: acquired/merged companies, new entrants, or operators that only appear in other pipeline type data.

Use a **year-lagged join** if computing rates as exposure denominator: match each event to the operator's annual report for the same year or prior year (annual reports reflect prior-year operations).

---

## Available Metrics by Pipeline Type

### Gas Distribution (`GD AR {year}.csv`)
- Columns: 274 | Key join: `OPERATOR_ID` + `REPORT_YEAR`
- **101 mileage columns** broken down by material (steel bare/coated, plastic, cast iron, copper, etc.) and pressure class
- Leak counts by cause category (corrosion, third-party, etc.)
- Service counts (residential, commercial, industrial)
- Useful denominator: sum of steel + plastic mileage for total system miles

### Gas Transmission (`GT AR {year} Part *.csv` — ~14 files/year, one per operator segment)
- Columns: 154 per file | Key join: `OPERATOR_ID` + `REPORT_YEAR`
- **Must concat all part-files for a given year, then group/sum by OPERATOR_ID**
- Key mileage cols: `PARTLTRON192MILES` (onshore), `PARTLTROFF192MILES` (offshore), `PARTLTR192MILESTOTAL`
- Also has `REPORT_NUMBER` and `SUPPLEMENTAL_NUMBER` (can join directly to event records)

### Hazardous Liquid (`HL AR {year} Part A to E.csv`, etc.)
- Part A has pipeline totals (88 cols): `PARTDTOTALMILES` is the key denominator column
- Parts B–J have commodity-specific, material, and inspection detail
- Key join: `OPERATOR_ID` + `REPORT_YEAR`

---

## Implementation Plan for Future Session

1. Write `load_operator_data(base_dir, pipeline_type, year_range)`:
   - GD: read single CSV per year, stack years
   - GT: glob all part-files per year, concat, groupby/sum OPERATOR_ID per REPORT_YEAR
   - HL: read Part A file per year (has total mileage), stack years
2. Add `join_events_with_ops(lag_months=0)` to `PHMSADataPipeline` — same pattern as `RailDataPipeline.join_events_with_ops()`
3. Output adds: `system_miles`, `report_year_matched`, `incidents_per_1000_miles` (computed post-join)
4. Handle unmatched operators gracefully (left join, NaN fill)
