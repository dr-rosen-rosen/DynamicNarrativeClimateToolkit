"""Draw the aviation calibration check for the 2026-11-09 meeting.

15 of the 64 aviation narratives that are NOT on the stage-2 discordant
worklist, simple random, seed 20261109. Each rater gets an identical blank
sheet; earlier (guide v1.0) ratings go to a separate key. No indicator score
is read or written.

    .dynClimVenv/bin/python validity_study/stage2_consolidated_2026-10-26/draw_aviation_check.py
"""
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font

HERE = Path(__file__).resolve().parent
OUT = HERE / "aviation_check_2026-11-09"
SEED, N = 20261109, 15


def main():
    m = pd.read_csv(HERE / "stage2_ratings_merged.csv", dtype=str,
                    usecols=["rating_id", "industry", "r_A", "cj_A", "r_B", "cj_B"]).fillna("")
    work = pd.read_csv(HERE / "stage2_for_consensus_tagged.csv", dtype=str, usecols=["rating_id"])
    txt = pd.read_csv(HERE / "rating_sheet_raterA_consolidated.csv", dtype=str,
                      usecols=["rating_id", "text"]).fillna("")
    pool = m[(m.industry == "aviation") & ~m.rating_id.isin(work.rating_id)]
    pool = pool.sort_values("rating_id").merge(txt, on="rating_id")
    assert len(pool) == 64
    s = pool.sample(n=N, random_state=SEED).reset_index(drop=True)
    s.insert(0, "check_id", range(1, N + 1))

    sheet = pd.DataFrame({
        "check_id": s.check_id, "rating_id": s.rating_id,
        "text": s.text.map(lambda x: " ".join(x.split())),
        "event_occurred_YN": "", "rating_1to5": "", "cant_judge": "", "notes": "",
    })
    OUT.mkdir(exist_ok=True)
    s[["check_id", "rating_id", "r_A", "cj_A", "r_B", "cj_B"]].rename(
        columns={"r_A": "v1.0_rating_A", "cj_A": "v1.0_cj_A",
                 "r_B": "v1.0_rating_B", "cj_B": "v1.0_cj_B"}
    ).to_csv(OUT / "aviation_check_key_v1.0_ratings.csv", index=False)
    widths = {"check_id": 9, "rating_id": 16, "text": 120, "event_occurred_YN": 14,
              "rating_1to5": 12, "cant_judge": 11, "notes": 40}
    for rater in "AB":
        path = OUT / f"aviation_check_rater{rater}.xlsx"
        sheet.to_excel(path, index=False, sheet_name="Aviation check")
        wb = load_workbook(path)
        ws = wb.active
        for i, name in enumerate(sheet.columns, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = widths[name]
            ws.cell(1, i).font = Font(bold=True)
        for row in ws.iter_rows():
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        wb.save(path)
    print(s[["check_id", "rating_id", "r_A", "r_B"]].to_string(index=False))
    print("words:", int(sheet.text.str.split().str.len().sum()))


if __name__ == "__main__":
    main()
