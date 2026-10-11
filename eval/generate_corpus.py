#!/usr/bin/env python3
"""Generate clean synthetic PDFs with KNOWN needle facts for RAG eval.

Each fact is on its own line so chunk attribution (500/100 splitter in
app/tasks.py) is unambiguous. Output -> eval/synthetic_corpus/*.pdf

Usage:
    .venv/bin/pip install fpdf2
    .venv/bin/python eval/generate_corpus.py
    # then upload each PDF via POST /documents/upload and poll status
"""
from pathlib import Path
from fpdf import FPDF

OUT = Path(__file__).parent / "synthetic_corpus"
OUT.mkdir(parents=True, exist_ok=True)


def write_pdf(name: str, title: str, lines: list[str]) -> Path:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    for line in lines:
        # multi_cell handles long lines; each fact stays a separate paragraph
        # so the RecursiveCharacterTextSplitter keeps it intact.
        pdf.multi_cell(0, 7, line)
        pdf.ln(2)
    path = OUT / name
    pdf.output(str(path))
    print(f"wrote {path} ({len(lines)} facts)")
    return path


COMPANY_POLICY = [
    "Acme Corp Employee Handbook, effective 2025-01-15.",
    "FACT-001: The Project Phoenix budget is $42,500.",
    "FACT-002: Full-time employees receive 22 days of annual leave per year.",
    "FACT-003: The Chief Technology Officer is Dr. Amara Singh.",
    "FACT-004: Employees may work remotely up to 3 days per week.",
    "FACT-005: Expense reports above $1,000 require manager approval.",
    "FACT-006: Company headquarters is on the 4th floor of Meridian Tower, Karachi.",
    "FACT-007: The Q3 security training deadline is 2025-09-30.",
    "FACT-008: New hires must complete onboarding within 14 days of joining.",
    "FACT-009: The IT helpdesk email is helpdesk@acme.example.",
    "FACT-010: Travel bookings must use the Concur portal.",
]

PRODUCT_SPECS = [
    "Acme Widget Product Specifications, version 3.2.1, dated 2025-06-01.",
    "FACT-101: The Widget X200 maximum load capacity is 750 kg.",
    "FACT-102: The Widget X200 battery lasts 48 hours under continuous use.",
    "FACT-103: The Widget X200 firmware version is v3.2.1.",
    "FACT-104: The Widget X200 operating temperature range is -10C to 50C.",
    "FACT-105: The Widget X200 warranty period is 24 months.",
    "FACT-106: The Widget X200 weighs 12.4 kg.",
    "FACT-107: The Widget X200 charging time is 90 minutes with fast charger.",
    "FACT-108: The Widget X200 is certified to IP67 for dust and water resistance.",
]

# Same keywords, DIFFERENT values -> tests disambiguation + MMR diversity.
DISTRACTOR = [
    "Archived notes and unrelated memos. Do not confuse with current policy.",
    "DISTRACT-001: The Project Atlas budget is $19,000. This is NOT Project Phoenix.",
    "DISTRACT-002: Interns receive 10 days of annual leave. This is NOT the full-time policy.",
    "DISTRACT-003: The former CTO was Mr. John Doe (retired 2023). The current CTO is different.",
    "DISTRACT-004: The Widget X100 (old model) maximum load was 300 kg. The X200 is different.",
    "DISTRACT-005: Office parking is in basement B2. Unrelated to headquarters floor.",
    "A table-like section to test PyPDFLoader on structured text:",
    "Model | Max load | Battery | Warranty",
    "X100 | 300 kg | 20 hours | 12 months",
    "X200 | 750 kg | 48 hours | 24 months",
    "End of archived notes.",
]

if __name__ == "__main__":
    write_pdf("company_policy.pdf", "Acme Corp - Employee Handbook (Synthetic)", COMPANY_POLICY)
    write_pdf("product_specs.pdf", "Acme Widget - Product Specs (Synthetic)", PRODUCT_SPECS)
    write_pdf("distractor.pdf", "Archived Notes - Distractors (Synthetic)", DISTRACTOR)
    print(f"\nDone. Upload with:\n  POST /documents/upload (each PDF) -> poll GET /documents/{{id}} until COMPLETED")
