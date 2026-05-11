---
name: PHMSA Historical Data — Schema Differences and Comparability Gaps
description: Column-level mapping of what can and cannot be standardized across the 4 historical eras of PHMSA incident/accident data; critical issues for NLP-based safety climate analysis
type: project
---

## Overview: Four Schema Eras

| Era | Files | Records | Pipeline types |
|---|---|---|---|
| **Era 1** — 1970–mid-1984 | GD_1970_1984, GT_1970_1984 | 22,475 | Gas distribution, gas transmission |
| **Era 2** — mid-1984–2002 | HL_pre1986 (1968–1985), HL_1986_2002, GD_1984_2004, GT_1984_2001 | 10,270 | All three types |
| **Era 3** — 2002–2009 | HL_2002_2009, GD_2004_2009, GT_2002_2009 | 4,954 | All three types |
| **Era 4** — 2010–present | HL_2010+, GD_2010+, GT_2010+, gravity_regulated | 9,992 | All four types (already built) |
| **Total historical** | | ~37,699 | |

---

## Narrative Fields — The Core Issue for NLP

This is the most consequential gap for safety climate analysis.

### Era 1 (1970–1984): NO free-text narrative
- GD and GT have only structured `_TEXT` fields: `ORIGIN_LEAK`, `DESCRIPTION_CORROSION`, `CAUSE_CORROSION`, `RUPTURE_OCCURED`, etc.
- These are **encoded categorical labels** in text form, not free-text descriptions. A typical value: `"STEEL BARE"` or `"EXTERNAL CORROSION"`.
- Can be concatenated into a pseudo-narrative string, but the resulting text is categorically different from a natural-language safety narrative — unsuitable for NLP-based climate scoring without separate calibration.

### Era 2 (mid-1984–2002): Mixed
- `GD_1984_2004` and `GT_1984_2001`: **Have `NARRATIVE` column** — comparable to modern data
- `HL_pre1986` (1968–1985): **No `NARRATIVE`** — 11 `_TEXT` fields (same limitation as Era 1)
- `HL_1986_2002`: **Has `NARRATIVE`** plus `CORR`/`CORRO` fields

### Era 3 (2002–2009): Richest historical narrative data
- All files have a `NARRATIVE` column
- Plus 33–38 additional `_TEXT` columns (`CAUSE_TEXT`, `CAUSE_DETAILS_TEXT`, `LEAK_TEXT`, `EVAC_REASON_TEXT`, etc.)
- `NARRATIVE` field is directly comparable to modern; the supplementary `_TEXT` fields could optionally be concatenated to enrich it (analogous to combining description + suggestions in healthcare pipeline)

### Era 4 (2010–present): Modern — already built

**Bottom line for NLP:**
- Era 3 (2002–2009): Merge with modern data with confidence — `NARRATIVE` field is equivalent
- Era 2 gas files (1984–2001): Merge with caution — `NARRATIVE` present but may be less detailed/consistent
- Era 1 + Era 2 HL pre-1986: Do not merge for NLP analysis — no comparable free-text narrative

---

## Event ID

| Era | Field | Format | Compatible with OPERATOR_ID join? |
|---|---|---|---|
| Era 1 | `REPORT_ID` | Integer | Yes — `OPERATOR_CODE` / `OPERATOR_NAME` |
| Era 2 HL | `RPTID` | Integer | Yes — `OPERATOR_ID` |
| Era 2 Gas | `RPTID` | Integer | Yes — `OPID` (may differ from modern OPERATOR_ID) |
| Era 3 | `RPTID` | Integer | Yes — `OPERATOR_ID` |
| Era 4 | `REPORT_NUMBER` | Integer | Yes — `OPERATOR_ID` |

**Gotcha:** `OPID` in Era 2 gas files may not be the same coding as `OPERATOR_ID` in Era 3/4. Verify by matching on `NAME`/`PNAME` if joining across eras.

---

## Fatality & Injury Categorization

Can compute a total, but the breakdown changes across eras:

| Era | Available categories | Total computable? |
|---|---|---|
| Era 1 GD/GT | Employee, non-employee only | Yes (sum 2 fields) |
| Era 1 HL_pre1986 | Employee, non-employee, total | Yes (use TOTAL field) |
| Era 2 gas | Single `FAT` / `INJ` field (total) | Yes |
| Era 2 HL | `EFAT`, `NFAT`, `TFAT` (employee, non-emp, total) | Yes |
| Era 3 | `FATAL`/`INJURE` (total) + `EFAT`, `NFAT`, `GPFAT` | Yes |
| Era 4 | `NUM_EMP_`, `NUM_CONTR_`, `NUM_ER_`, `NUM_WORKER_`, `NUM_GP_` | Yes |

**What cannot be standardized:** The employee / contractor / general public breakdown only exists in Era 4. Comparing "general public fatalities" across eras is not possible — the category did not exist until 2010.

---

## Cost Data

Not comparable across eras, even after renaming:

| Era | Field(s) | What it covers | Issue |
|---|---|---|---|
| Era 1 GD/GT | `OPERATORS_PROPERTY_DAMAGE` only | Operator property damage only | Single category, 1970s dollars |
| Era 2 gas | `PRPTY` only | Property damage (total) | Single category, no emergency/env costs |
| Era 2 HL | `PRPTY_DAMAGE`, `TOTAL_PRPTY_DAMAGE` | Property damage items | Separate line items, pre-1986 dollars |
| Era 3 | `PRPTY`, `PIPE_DAMAGE`, `PROPLN` | Pipeline/property damage | Three partially overlapping fields |
| Era 4 | `EST_COST_OPER_PAID`, `EST_COST_PROP_DAMAGE`, `EST_COST_EMERGENCY`, `EST_COST_ENVIRONMENTAL`, etc. | Full cost breakdown | Estimated costs, current dollars |

**Cannot standardize:** Emergency response costs and environmental remediation costs do not exist as categories before 2010. Inflation adjustment required for any dollar comparisons. The `EST_COST_` prefix in Era 4 signals estimated cost; earlier fields may be actuals or different definitions entirely.

---

## Cause Coding

Cause taxonomy is completely different across eras — not mappable:

| Era | Approach |
|---|---|
| Era 1 | Coded categorical fields (integer codes) for cause category — small number of top-level categories |
| Era 2 | Coded categoricals with some `_TEXT` elaboration |
| Era 3 | Coded categoricals + 30+ `_TEXT` fields expanding each cause |
| Era 4 | Binary indicator columns for 200+ specific sub-cause factors (corrosion type, material defect specifics, operator error type, etc.) |

A "corrosion" cause in 1975 is a single code. In 2020, it triggers 15+ binary indicators (galvanic, atmospheric, microbiological, etc.). There is no principled way to map these to a common scheme without a hand-built crosswalk.

---

## What CAN Be Standardized (reliable across all eras)

| Field | Notes |
|---|---|
| `OPERATOR_ID` | Present in most files; Era 2 gas uses `OPID` — verify alignment |
| Incident year | All files have some year field (`IYEAR`, `IYEAR` from `IDATE`, `IYEAR` column) |
| State | All files have state — field name varies (`INCIDENT_STATE`, `ACSTATE`, `OPSTATE`, `OPERATOR_STATE_ABBREVIATION`) |
| Pipeline type label | Add as metadata during load |
| Total fatalities | Computable from whatever fields exist |
| Total injuries | Computable from whatever fields exist |
| Narrative (limited) | Era 2 gas + Era 3: `NARRATIVE` field comparable to Era 4. Era 1 + Era 2 HL pre-1986: structured text only |

---

## Recommended Integration Strategy

**Tier 1 — Integrate now (high confidence):**
Era 3 files (2002–2009): GD_2004_2009, GT_2002_2009, HL_2002_2009
- Have `NARRATIVE` column directly comparable to Era 4
- Have `RPTID` (rename to `raw_event_id`)
- Have `OPERATOR_ID`
- Casualty totals computable
- Adds ~4,954 events, extends coverage to 2002
- Optional: concatenate the 30+ `_TEXT` fields after `NARRATIVE` to enrich text

**Tier 2 — Integrate with caution:**
Era 2 gas files (1984–2001): GD_1984_2004, GT_1984_2001
- Have `NARRATIVE` but likely shorter/less standardized
- Use `OPID` not `OPERATOR_ID` — verify alignment
- Single `FAT`/`INJ` field only (no breakdown)
- Adds ~4,343 events, extends coverage to 1984
- Flag these with a `narrative_quality = 'historical'` indicator

**Tier 3 — NLP analysis not viable, exclude or use for counts only:**
Era 1 (1970–1984): GD_1970_1984, GT_1970_1984
Era 2 HL pre-1986: HL_pre1986, HL_1986_2002 (pre-1986 portion)
- No free-text narrative comparable to modern data
- Can be used for incident count / casualty time-series analysis only, not NLP

---

## Implementation Notes for Future Session

When building historical integration into `phmsa_data_pipeline.py`:

1. Add `era` and `narrative_quality` columns (`'modern'`, `'era3'`, `'era2_gas'`, `'structured_text_only'`) so downstream analysis can filter or weight appropriately
2. Era 3 narrative concatenation: `NARRATIVE + ' ' + CAUSE_TEXT + ' ' + CAUSE_DETAILS_TEXT + ' ' + EVAC_REASON_TEXT` (prioritize these 3 supplementary fields)
3. Date parsing: Era 2–3 use `IDATE` (format YYYYMMDD integer or similar) not `LOCAL_DATETIME` — add flexible date parser
4. Event ID: rename `RPTID` → `raw_event_id` for Era 2–3; confirm `OPID` ↔ `OPERATOR_ID` alignment before joining to operator data
5. Cost: expose raw cost fields per era; do not attempt cross-era `total_cost` aggregation
