#!/usr/bin/env bash
set -euo pipefail

# World Bank Global Economic Prospects, June 2026: one-domain public demo corpus.
# Run from anywhere; downloaded files live under this repository's data directory.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$repo_root/data/gep_2026_june/raw"
base='https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026'
mkdir -p "$target"

curl -fLsS --retry 2 --max-time 180 "$base/original/GEP-Jun-2026.pdf" -o "$target/report.pdf"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Chapter-1-Highlights.pdf" -o "$target/chapter1_highlights.pdf"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Chapter-2-Highlights.pdf" -o "$target/chapter2_highlights.pdf"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Chapter-3-highlights.pdf" -o "$target/chapter3_highlights.pdf"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Chapter-4-highlights.pdf" -o "$target/chapter4_highlights.pdf"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Regional-Highlights-SAR.pdf" -o "$target/regional_south_asia.pdf"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Regional-Highlights-LAC.pdf" -o "$target/regional_latin_america.pdf"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Regional-Highlights-SSA.pdf" -o "$target/regional_sub_saharan_africa.pdf"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-June-2026-Chapter1-Fig1-1.xlsx" -o "$target/figure_1_1.xlsx"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-June-2026-Chapter1-Fig1-5.xlsx" -o "$target/figure_1_5.xlsx"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-June-2026-Chapter2-Fig2-1.xlsx" -o "$target/figure_2_1.xlsx"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-June-2026-Chapter4-Fig4-13.xlsx" -o "$target/figure_4_13.xlsx"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Chapter-2-SA-data.xlsx" -o "$target/chapter2_south_asia.xlsx"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Table-Statistical-Appendix.xlsx" -o "$target/statistical_appendix.xlsx"
curl -fLsS --retry 2 --max-time 180 "$base/related/GEP-Jun-2026-Table-1-1.xlsx" -o "$target/table_1_1.xlsx"
curl -fLsS --retry 2 --max-time 180 'https://databankfiles.worldbank.org/public/ddpext_download/GEP_CSV.zip' -o "$target/gep_csv.zip"
unzip -o -j "$target/gep_csv.zip" GEPCSV.csv GEPSeries.csv -d "$target"

for workbook in "$target"/*.xlsx; do
  unzip -tq "$workbook" >/dev/null
done
file "$target"/*
shasum -a 256 "$target"/*
