# ParentCare Medical Reference Corpus

This folder contains structured factual reference material for interpreting
common lab tests, intended as the retrieval corpus for a RAG-based lab
parsing system. Content is organized one panel per file. Each entry follows
a consistent template:

- **What it measures** — what the test is and what biological quantity it tracks
- **When it's ordered** — typical clinical reasons to request the test
- **Indian adult reference range** — typical ranges (defer to the printed range on the actual report)
- **Low values can indicate** — factual differential considerations
- **High values can indicate** — factual differential considerations
- **Important context** — interpretation caveats, related tests, population notes
- **Critical thresholds** — values that may warrant urgent attention

This corpus is **factual reference content only**. It does not provide
treatment, medication, diet, or lifestyle advice. It is intended to be used
inside an automated explanation system that retrieves relevant passages and
asks a language model to answer questions grounded in them, with citations.
Always interpret lab results with a qualified clinician.

## File layout

- `01_cbc.md` — Complete Blood Count (Hb, MCV, MCH, RBC, WBC and differential, platelets, reticulocytes)
- `02_metabolic_panel.md` — Lipid Profile, Glucose / HbA1c, Liver Function (LFT), Kidney Function (KFT)
- `03_endocrine_and_nutrition.md` — Thyroid, Iron Studies, Vitamins D / B12 / Folate / Homocysteine
- `04_minerals_and_inflammation.md` — Electrolytes, Calcium / Phosphorus / Magnesium, CRP / ESR, basic cardiac markers
- `05_clinical_patterns.md` — Named clinical patterns (microcytic anemia, fatty liver, prediabetes, subclinical hypothyroidism, etc.)

## Source

This is internal reference material compiled from authoritative clinical
sources including MedlinePlus (US National Library of Medicine), Testing.com
(AACC), Mayo Clinic patient pages, and WHO fact sheets. It is intended for
personal use in a non-commercial setting.
