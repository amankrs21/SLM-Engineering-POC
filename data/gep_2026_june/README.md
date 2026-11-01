# Sample corpus: Global Economic Prospects, June 2026

One-domain public test corpus for the RAG POC. The World Bank [report page](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/) provides PDFs and Excel files; its [data catalog](https://datacatalog.worldbank.org/search/dataset/0037888/global-economic-prospects) provides the CSV archive. There are **20 ingestible files: 8 PDFs, 7 XLSX, 2 CSV, and 3 derived TXT notes** (~7.7 MB total on disk, including one retained ZIP archive). The TXT notes paraphrase facts from the named primary sources.

| File | Type | Purpose | Source |
| --- | --- | --- | --- |
| `raw/report.pdf` | PDF, 200 pages | Long document retrieval and page citations | [Full report](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/original/GEP-Jun-2026.pdf) |
| `raw/chapter1_highlights.pdf` | PDF | Short summary retrieval | [Chapter 1 highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Chapter-1-Highlights.pdf) |
| `raw/chapter2_highlights.pdf` | PDF | Regional outlook narrative | [Chapter 2 highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Chapter-2-Highlights.pdf) |
| `raw/chapter3_highlights.pdf` | PDF | Sovereign debt narrative | [Chapter 3 highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Chapter-3-highlights.pdf) |
| `raw/chapter4_highlights.pdf` | PDF | Commodity/fiscal narrative | [Chapter 4 highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Chapter-4-highlights.pdf) |
| `raw/regional_south_asia.pdf` | PDF | South Asia regional follow-ups | [South Asia highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Regional-Highlights-SAR.pdf) |
| `raw/regional_latin_america.pdf` | PDF | Cross-region retrieval | [Latin America highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Regional-Highlights-LAC.pdf) |
| `raw/regional_sub_saharan_africa.pdf` | PDF | Cross-region retrieval | [Sub-Saharan Africa highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Regional-Highlights-SSA.pdf) |
| `raw/figure_1_1.xlsx` | XLSX | Excel sheet inspection | [Figure 1.1 data](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-June-2026-Chapter1-Fig1-1.xlsx) |
| `raw/figure_1_5.xlsx` | XLSX | Chapter 1 figure data | [Figure 1.5 data](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-June-2026-Chapter1-Fig1-5.xlsx) |
| `raw/figure_2_1.xlsx` | XLSX | Chapter 2 figure data | [Figure 2.1 data](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-June-2026-Chapter2-Fig2-1.xlsx) |
| `raw/figure_4_13.xlsx` | XLSX | Chapter 4 figure data | [Figure 4.13 data](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-June-2026-Chapter4-Fig4-13.xlsx) |
| `raw/chapter2_south_asia.xlsx` | XLSX | South Asia structured table | [Chapter 2 South Asia data](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Chapter-2-SA-data.xlsx) |
| `raw/statistical_appendix.xlsx` | XLSX | Wider statistical table | [Statistical appendix](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Table-Statistical-Appendix.xlsx) |
| `raw/table_1_1.xlsx` | XLSX | Structured report table | [Table 1.1 data](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Table-1-1.xlsx) |
| `raw/GEPCSV.csv` | CSV | Country/region forecasts and exact calculations | [GEP CSV archive](https://databankfiles.worldbank.org/public/ddpext_download/GEP_CSV.zip) |
| `raw/GEPSeries.csv` | CSV | Indicator definitions | Same archive |
| `derived/press_release_summary.txt` | TXT | Plain-text ingestion and source attribution | [Press release](https://www.worldbank.org/en/news/press-release/2026/06/11/global-economic-prospects-june-2026-press-release) |
| `derived/south_asia_notes.txt` | TXT | Regional text ingestion | [South Asia highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Regional-Highlights-SAR.pdf) and CSV archive |
| `derived/debt_notes.txt` | TXT | Topic-specific text ingestion | [Chapter 3 highlights](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/related/GEP-Jun-2026-Chapter-3-highlights.pdf) |

`raw/gep_csv.zip` is the downloaded archive, retained for provenance. Ingest the extracted files; skip the ZIP to avoid duplicates. Files were downloaded on 31 October 2026. Run `bash scripts/fetch_gep_data.sh` from the repository root to fetch them again. The [source manifest](source_manifest.json) maps every ingestible file to its original URL/title; the [SHA-256 manifest](checksums.sha256) records downloaded files, derived notes and source metadata. Run `shasum -a 256 -c data/gep_2026_june/checksums.sha256` from the repository root to verify them. Do not silently replace an indexed corpus: create a new corpus version after refresh.

The 200-page report and short highlights overlap heavily. Preserve distinct file citations, but deduplicate near-identical chunks at retrieval time so top 5 does not contain five copies of one paragraph. The TXT notes are **derived test fixtures**, not primary evidence for production answers.

The [report](https://thedocs.worldbank.org/en/doc/2b672b3b0415d6b66c45b66579db4ef5-0050012026/original/GEP-Jun-2026.pdf) states CC BY 3.0 IGO attribution terms; the [data catalog](https://datacatalog.worldbank.org/search/dataset/0037888/global-economic-prospects) labels its downloadable dataset CC BY 4.0. Keep source and attribution with any redistributed demo data. No World Bank endorsement is implied.

Example gold questions:

1. “Report ke hisaab se 2026 global growth forecast kya hai?” → 2.5%, with report/press-release evidence.
2. “Aur 2027 mein?” → follow-up should resolve global growth from branch-specific chat history; 2.8%.
3. “CSV mein South Asia ka 2026 aur 2027 GDP growth kya hai?” → 6.3% and 6.9%, with `GEPCSV.csv` row citation.
4. “Report aur CSV dono 2026 South Asia forecast par agree karte hain?” → compare separately retrieved PDF and CSV evidence; do not assume agreement without checking.
5. “Which chapter discusses debt levels and borrowing costs?” → Chapter 3 highlights and report pages, with citation.
6. “South Asia aur Latin America ka 2026 forecast compare karo.” → independent regional/CSV evidence, with both sources cited.
