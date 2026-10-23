CAI 8103 Capstone Project II - CAT 1 submission
Tony Blair Ochieng, JAN26/MAIDS/0077U

1. JAN26_MAIDS_0077U_CAT1_Progress_Report.pdf   Progress report (7 pages, all five components)
2. JAN26_MAIDS_0077U_Updated_Work_Plan.xlsx     Completed and outstanding activities with dates
3. source_code_capstone_electricity/            Repository snapshot: Capstone I notebook + Capstone II package (capstone2/)
     GitHub: https://github.com/TBlair24/capstone_electricity  (Capstone II on branch "capstone2")
4. evidence/                                     Figures, 23 result tables (CSV) and the full console log of a clean run

To reproduce: pip install -r requirements.txt pytest ; python capstone2/run_all.py
The 85 MB distribution-line layer is omitted to keep the ZIP small; download it as described in
README.md (energydata.info "distribution-lines-2023") into data/gis/distribution_lines.geojson.
