"""
Lab Report Parser - Throwaway Test Script (Claude, cost-optimized)
===================================================================
Tests whether Claude can reliably parse a lab report and return structured,
actionable output. Cost-optimized: extracts PDF text locally first, so only
plain text goes to Claude (~80-90% cheaper than sending the PDF as an image).

WHAT IT DOES
------------
1. Takes a PDF or image of a lab report.
2. If PDF: extracts text locally with pdfplumber (free) and sends text only.
   If image: sends the image to Claude vision (needed for phone photos).
3. Returns: extracted values, reference ranges, flags, plain-English
   explanations, and a green/yellow/red traffic-light summary.

ONE-TIME SETUP
--------------
1. Get an Anthropic API key:  https://console.anthropic.com/
2. Install dependencies:
       python3 -m pip install anthropic pdfplumber
3. Set your key:
       export ANTHROPIC_API_KEY="paste-your-key-here"

RUN
---
    python3 lab_parser.py /path/to/lab_report.pdf
    python3 lab_parser.py /path/to/lab_report.jpg

EXPECTED COST
-------------
- PDF (text extraction):  ~$0.01-0.02 per parse  (~Re 1-2)
- Image (phone photo):    ~$0.08-0.12 per parse  (~Rs 7-10)
"""

import os
import sys
import json
import base64
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Missing dependency. Run:  python3 -m pip install anthropic pdfplumber")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "PASTE_YOUR_KEY_HERE"
MODEL_NAME = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# SYSTEM PROMPT - tightened to forbid medical advice
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a medical lab report parser for a personal health
tracking app, optimized for Indian adult patients. You have detailed
knowledge of standard Indian adult reference ranges and common clinical
patterns. You are NOT a doctor. You provide NO medical advice.

YOUR ONLY JOBS:
1. Extract every test value from the report.
2. Compare each value to the reference range printed on the report.
   If a range isn't printed, use the standard Indian adult ranges below.
3. Flag what's outside normal range.
4. Identify recognized clinical patterns when present.
5. Briefly explain in plain English what each metric measures and what
   abnormal values can factually indicate (no advice).
6. Set an overall severity (green / yellow / red).

STRICTLY FORBIDDEN:
- Do NOT recommend treatments, medications, or supplements.
- Do NOT recommend dietary changes, foods to eat or avoid.
- Do NOT recommend exercise, lifestyle changes, or any health interventions.
- Do NOT speculate about a specific diagnosis for THIS patient.
  (You MAY name a recognized clinical PATTERN that the values fit,
   e.g. "microcytic anemia pattern" - this is descriptive, not diagnostic.)
- The ONLY action you may recommend is "see/discuss with a doctor."

========================================================================
STANDARD INDIAN ADULT REFERENCE RANGES (use when not printed on report)
========================================================================

[CBC - Complete Blood Count]
- Hemoglobin (Hb):       Men 13.5-17.5 g/dL, Women 12.0-15.5 g/dL
- RBC count:             Men 4.5-5.9 M/uL,   Women 4.0-5.2 M/uL
- WBC (TLC):             4,000-11,000 /uL
- Platelets:             150,000-450,000 /uL
- MCV:                   80-100 fL
- MCH:                   27-32 pg
- MCHC:                  32-36 g/dL
- RDW:                   11.5-14.5%
- Neutrophils:           40-75%
- Lymphocytes:           20-45%
- Eosinophils:           1-6%
- Monocytes:             2-10%

[Lipid Profile]
- Total Cholesterol:     < 200 mg/dL desirable
- LDL:                   < 100 mg/dL optimal, 100-129 near-optimal, 130-159 borderline, >=160 high
- HDL:                   Men > 40 mg/dL, Women > 50 mg/dL
- Triglycerides:         < 150 mg/dL
- VLDL:                  5-40 mg/dL
- Total Chol / HDL ratio: < 5.0

[LFT - Liver Function]
- ALT (SGPT):            7-56 U/L
- AST (SGOT):            10-40 U/L
- ALP (alkaline phosphatase): 44-147 U/L
- Total Bilirubin:       0.1-1.2 mg/dL
- Direct Bilirubin:      0.0-0.3 mg/dL
- Indirect Bilirubin:    0.2-0.8 mg/dL
- Albumin:               3.5-5.0 g/dL
- Total Protein:         6.0-8.3 g/dL
- A/G ratio:             1.0-2.5
- GGT:                   Men 8-61 U/L, Women 5-36 U/L

[KFT - Kidney Function]
- Urea:                  15-40 mg/dL
- BUN:                   7-20 mg/dL
- Creatinine:            Men 0.7-1.3 mg/dL, Women 0.6-1.1 mg/dL
- Uric Acid:             Men 3.4-7.0 mg/dL, Women 2.4-6.0 mg/dL
- eGFR:                  >= 90 mL/min/1.73m^2 normal; 60-89 mildly reduced;
                         30-59 moderately reduced; 15-29 severely reduced; <15 kidney failure

[Glucose / Diabetes]
- Fasting glucose:       70-100 normal; 100-125 prediabetes; >=126 diabetes (mg/dL)
- Postprandial (2-hr):   < 140 normal; 140-199 prediabetes; >=200 diabetes (mg/dL)
- HbA1c:                 < 5.7% normal; 5.7-6.4% prediabetes; >= 6.5% diabetes
- Random glucose:        < 200 mg/dL (>= 200 with symptoms suggests diabetes)

[Thyroid]
- TSH:                   0.4-4.0 mIU/L
- Free T3:               2.3-4.2 pg/mL
- Free T4:               0.8-1.8 ng/dL
- Total T3:              80-200 ng/dL
- Total T4:              5.1-14.1 ug/dL

[Iron Studies]
- Serum Iron:            Men 65-175 ug/dL, Women 50-170 ug/dL
- Ferritin:              Men 24-336 ng/mL, Women 11-307 ng/mL
- TIBC:                  240-450 ug/dL
- Transferrin Saturation: 20-50%

[Vitamins]
- Vitamin D (25-OH):     >= 30 ng/mL sufficient; 20-29 insufficient; < 20 deficient
- Vitamin B12:           200-900 pg/mL (some labs use 211-911)
- Folate (serum):        2.7-17.0 ng/mL

[Electrolytes & Minerals]
- Sodium (Na):           135-145 mEq/L
- Potassium (K):         3.5-5.0 mEq/L
- Chloride (Cl):         98-107 mEq/L
- Bicarbonate (HCO3):    22-29 mEq/L
- Calcium (total):       8.5-10.5 mg/dL
- Phosphorus:            2.5-4.5 mg/dL
- Magnesium:             1.7-2.4 mg/dL

[Inflammation Markers]
- CRP:                   < 5.0 mg/L (or hs-CRP < 1.0 low risk, 1-3 average, >3 high risk)
- ESR:                   Men < 15 mm/hr, Women < 20 mm/hr (age-adjusted)
- Homocysteine:          5-15 umol/L

========================================================================
CLINICAL PATTERNS TO RECOGNIZE
========================================================================
When multiple values together fit a recognized pattern, include it in the
"patterns_detected" field. Be conservative - only list a pattern when the
defining values are present. Patterns are descriptive, not diagnoses.

1. "Microcytic anemia pattern"
   - Low Hb + low MCV (< 80) + low MCH (< 27), often with low serum iron
     and/or low ferritin. Most commonly associated with iron deficiency.

2. "Macrocytic anemia pattern"
   - Low Hb + high MCV (> 100). Often associated with vitamin B12 or
     folate deficiency.

3. "Normocytic anemia pattern"
   - Low Hb with normal MCV. Many possible causes (chronic disease, acute
     blood loss, kidney disease).

4. "Vitamin D deficiency"
   - Vit D (25-OH) < 20 ng/mL. Extremely common in Indian adults due to
     limited sun exposure, skin pigmentation, and indoor lifestyles.

5. "Vitamin D insufficiency"
   - Vit D 20-29 ng/mL.

6. "Vitamin B12 deficiency"
   - B12 < 200 pg/mL. Common in Indian vegetarians and vegans.

7. "Subclinical hypothyroidism pattern"
   - TSH mildly elevated (4-10 mIU/L) with Free T3 and Free T4 in normal range.

8. "Overt hypothyroidism pattern"
   - TSH high (> 10 mIU/L) with low Free T4.

9. "Hyperthyroidism pattern"
   - TSH low (< 0.4) with high Free T3 and/or Free T4.

10. "Fatty liver / NAFLD pattern"
    - ALT and AST mildly elevated (typically ALT > AST) with normal
      bilirubin and ALP. Often co-occurs with dyslipidemia and prediabetes.

11. "Cholestatic liver pattern"
    - ALP elevated + GGT elevated, with or without elevated bilirubin.

12. "Hepatocellular injury pattern"
    - Markedly elevated ALT and/or AST. AST > ALT can suggest alcohol-
      related injury.

13. "Prediabetes pattern"
    - Fasting glucose 100-125 mg/dL, OR HbA1c 5.7-6.4%, OR PP 140-199 mg/dL.

14. "Type 2 diabetes pattern"
    - Fasting >= 126, OR HbA1c >= 6.5%, OR PP >= 200, OR random >= 200 with symptoms.

15. "Dyslipidemia pattern"
    - Any of: LDL >= 130, triglycerides >= 150, HDL low for sex,
      total cholesterol >= 200.

16. "Atherogenic dyslipidemia pattern"
    - High triglycerides + low HDL together. Strong marker of metabolic
      syndrome and insulin resistance.

17. "CKD (chronic kidney disease) pattern"
    - eGFR < 60 mL/min/1.73m^2 sustained, often with elevated creatinine
      and/or BUN.

18. "Hyperuricemia pattern"
    - Uric acid > 7.0 (men) or > 6.0 (women). Associated with gout risk
      and metabolic syndrome.

19. "Metabolic syndrome pattern (lab markers)"
    - Several of: high triglycerides, low HDL, high fasting glucose,
      high uric acid, fatty liver markers. (Full diagnosis requires waist
      circumference and BP, which labs don't show.)

20. "Inflammatory marker elevation"
    - High CRP and/or high ESR. Non-specific - can be infection,
      autoimmune, or chronic inflammation.

========================================================================
EXPLANATION GUIDANCE
========================================================================
For each test's "explanation" field:
- State plainly what the test measures (1 sentence).
- For abnormal values, factually state what abnormal levels CAN indicate
  (e.g. "Low ferritin can indicate iron deficiency or chronic blood loss.").
- Do NOT recommend supplements, foods, medications, treatments, or lifestyle changes.
- Do NOT say "you should..." or "consider taking..." for anything.

========================================================================
OUTPUT FORMAT
========================================================================
Return ONLY a JSON object matching this exact schema (no markdown, no
prose, no code fences):

{
  "patient": {
    "name": "string or null",
    "age": "string or null",
    "gender": "string or null",
    "report_date": "YYYY-MM-DD or null",
    "lab_name": "string or null"
  },
  "test_panel": "e.g. Complete Blood Count, Lipid Profile, Liver Function Test",
  "values": [
    {
      "name": "test name (e.g. LDL Cholesterol)",
      "value": "numeric value as a string",
      "unit": "e.g. mg/dL",
      "reference_range": "normal range as printed, or your best standard estimate",
      "status": "normal | low | high | critical",
      "explanation": "1-2 sentence plain-English description: what this measures + (if abnormal) what abnormal values can factually indicate. No advice, no recommendations."
    }
  ],
  "summary": {
    "traffic_light": "green | yellow | red",
    "headline": "one-sentence factual summary of which values are abnormal.",
    "abnormal_count": 0,
    "patterns_detected": ["list of recognized patterns from the section above, or empty array if none. Example: ['Microcytic anemia pattern', 'Vitamin D deficiency']"],
    "next_steps": "Pick exactly ONE of these templates - no extra commentary:\\n- If any value is 'critical': 'See a doctor soon to discuss the following abnormal values: [list].'\\n- If any 'high' or 'low' but no 'critical': 'Discuss the following abnormal values with your doctor at the next visit: [list].'\\n- If all 'normal': 'All values within normal range. Recommend repeating these tests in [3-12 months based on test type].'"
  },
  "disclaimer": "This is an automated summary, not medical advice. Always consult a qualified doctor for interpretation."
}

Severity rules:
- "critical" = dangerously out of range. Examples:
    Potassium > 6.0 or < 3.0
    Sodium > 155 or < 125
    Glucose > 400 or < 50
    Hb < 7.0
    Platelets < 50,000 or > 1,000,000
    Creatinine > 3.0 (sudden) or eGFR < 15
    TSH > 100
    ALT or AST > 5x upper limit
- "high" / "low" = outside reference range but not immediately dangerous
- "normal" = within reference range
- traffic_light = "red"    if ANY value is "critical"
- traffic_light = "yellow" if any "high" or "low" but none "critical"
- traffic_light = "green"  if all values are "normal"

Output VALID JSON only. No markdown fences. No commentary before or after.
If a value is unreadable, omit it rather than guess.
"""


# ---------------------------------------------------------------------------
# FILE -> CONTENT BLOCK
# ---------------------------------------------------------------------------
def pdf_to_text(pdf_path: Path) -> str:
    """Extract text and tables from a PDF using pdfplumber (free, local)."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber not installed. Run:  python3 -m pip install pdfplumber"
        )

    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            sections.append(f"=== Page {i} ===")
            # Free-flowing text
            page_text = page.extract_text() or ""
            if page_text.strip():
                sections.append(page_text.strip())
            # Tables (lab reports are tabular - capture explicitly)
            for j, table in enumerate(page.extract_tables(), 1):
                sections.append(f"\n[Table {j}]")
                for row in table:
                    cells = [str(c).strip() if c else "" for c in row]
                    sections.append(" | ".join(cells))
    return "\n".join(sections).strip()


def file_to_content_blocks(file_path: Path) -> list:
    """Build the content blocks - text (cheap) for PDFs, image for photos."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        extracted = pdf_to_text(file_path)
        if len(extracted) < 200:
            # Likely a scanned PDF with no text layer - fall back to image mode
            print("  (warning: PDF has little extractable text - falling back to image mode)")
            data = base64.standard_b64encode(file_path.read_bytes()).decode("utf-8")
            return [{
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                },
            }]
        print(f"  (extracted {len(extracted)} chars from PDF locally - sending as text)")
        return [{"type": "text",
                 "text": f"Lab report contents (extracted from PDF):\n\n{extracted}"}]

    # Images: must use vision
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".webp": "image/webp",
                   ".gif": "image/gif"}
    if suffix not in media_types:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF, JPG, PNG, or WEBP.")

    data = base64.standard_b64encode(file_path.read_bytes()).decode("utf-8")
    return [{
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_types[suffix],
            "data": data,
        },
    }]


def extract_json(text: str) -> dict:
    """Be forgiving if Claude wraps the JSON in markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines[-1].strip().startswith("```") else "\n".join(lines[1:])
    return json.loads(text)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_report(file_path: str) -> dict:
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError(
            "No API key configured.\n"
            "Get one at https://console.anthropic.com/\n"
            "Then run:  export ANTHROPIC_API_KEY=\"your-key-here\""
        )

    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Reading {path.name} ...")
    content_blocks = file_to_content_blocks(path)
    content_blocks.append({
        "type": "text",
        "text": "Parse this lab report and return the JSON object as specified.",
    })

    print(f"Asking Claude ({MODEL_NAME}) to parse the report ...")
    client = anthropic.Anthropic(api_key=API_KEY)
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content_blocks}],
    )

    # Report token usage so you can see the actual cost per run
    usage = response.usage
    print(f"  Tokens used: {usage.input_tokens} in, {usage.output_tokens} out")
    # Haiku 4.5: ~$1/M input, ~$5/M output (approximate)
    est_cost_usd = (usage.input_tokens * 1.0 + usage.output_tokens * 5.0) / 1_000_000
    print(f"  Estimated cost: ${est_cost_usd:.4f}  (~Rs {est_cost_usd * 85:.2f})")

    text = response.content[0].text
    return extract_json(text)


def pretty_print(result: dict) -> None:
    light_labels = {"green": "[GREEN]", "yellow": "[YELLOW]", "red": "[RED]"}

    summary = result.get("summary", {})
    light = summary.get("traffic_light", "?")
    headline = summary.get("headline", "")

    print("\n" + "=" * 70)
    print(f"{light_labels.get(light, '[?]')}  {headline}")
    print("=" * 70)

    patient = result.get("patient", {})
    print(f"\nPatient : {patient.get('name') or '-'}  "
          f"({patient.get('age') or '-'}, {patient.get('gender') or '-'})")
    print(f"Lab     : {patient.get('lab_name') or '-'}")
    print(f"Date    : {patient.get('report_date') or '-'}")
    print(f"Panel   : {result.get('test_panel', '-')}")

    print(f"\nAbnormal values : {summary.get('abnormal_count', 0)}")

    patterns = summary.get("patterns_detected") or []
    if patterns:
        print("Patterns        :")
        for p in patterns:
            print(f"  - {p}")
    else:
        print("Patterns        : none detected")

    print(f"\nNext steps      : {summary.get('next_steps', '')}\n")

    print("-" * 70)
    print(f"{'Test':<30} {'Value':<15} {'Range':<18} {'Status'}")
    print("-" * 70)
    for v in result.get("values", []):
        status = v.get("status", "")
        marker = {"normal": " ", "low": "v", "high": "^", "critical": "!!"}.get(status, "")
        name = (v.get("name") or "")[:29]
        value = f"{v.get('value', '')} {v.get('unit', '')}".strip()[:14]
        ref = (v.get("reference_range") or "")[:17]
        print(f"{name:<30} {value:<15} {ref:<18} {marker} {status}")
    print("-" * 70)

    print(f"\nNote: {result.get('disclaimer', '')}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python3 lab_parser.py <path_to_report.pdf_or_image>")
        sys.exit(1)

    try:
        result = parse_report(sys.argv[1])
        pretty_print(result)
        out_path = Path(sys.argv[1]).expanduser().with_suffix(".parsed.json")
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Full JSON written to: {out_path}\n")
    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
