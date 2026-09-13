import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from . import resume_structure

ACCENT = colors.HexColor("#1d4ed8")
DARK = colors.HexColor("#111111")
GRAY = colors.HexColor("#555555")

SECTION_TITLES = {
    "summary": "Professional Summary",
    "skills": "Skills",
    "experience": "Experience",
    "projects": "Projects",
    "achievements": "Achievements",
    "certifications": "Certifications",
    "education": "Education",
    "other": "Additional Information",
}

# Canonical resume ordering, applied at render time regardless of the order
# sections happened to appear in the candidate's original document. Recruiter
# convention (and the mentor feedback this project is following) is to lead
# with what's most decision-relevant -- summary, skills, then the actual
# experience/projects -- and push Education toward the end once a candidate
# has real work/project history to show. Anything not in this list (a custom
# heading we didn't recognize) sorts after everything we do recognize.
CANONICAL_SECTION_ORDER = [
    "summary", "skills", "experience", "projects",
    "achievements", "certifications", "education", "other",
]

# Recognized contact-line labels we'll try to turn into real hyperlinks, and
# the URL-substring each one is normally associated with (used only to guess
# a human-friendly link label back from a URL if needed -- the actual URL
# always comes from the resume itself, never invented here).
LINK_LABEL_PATTERNS = {
    "linkedin": re.compile(r"\blinkedin\b", re.I),
    "github": re.compile(r"\bgithub\b", re.I),
    "portfolio": re.compile(r"\bportfolio\b", re.I),
    "website": re.compile(r"\bwebsite\b", re.I),
    "twitter": re.compile(r"\btwitter\b|\bx\.com\b", re.I),
    "behance": re.compile(r"\bbehance\b", re.I),
    "medium": re.compile(r"\bmedium\b", re.I),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
}


def _styles():
    styles = getSampleStyleSheet()
    return {
        "name": ParagraphStyle("Name", parent=styles["Title"], fontSize=19, leading=22,
                                alignment=TA_CENTER, textColor=DARK, spaceAfter=2),
        "contact": ParagraphStyle("Contact", parent=styles["Normal"], fontSize=9.5,
                                   alignment=TA_CENTER, textColor=GRAY, spaceAfter=4),
        "section": ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12,
                                   textColor=ACCENT, spaceBefore=6, spaceAfter=2,
                                   leading=14),
        "body": ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=12.5),
        "job_title": ParagraphStyle("JobTitle", parent=styles["Normal"], fontSize=10,
                                     leading=12, fontName="Helvetica-Bold"),
        "job_date": ParagraphStyle("JobDate", parent=styles["Normal"], fontSize=9,
                                    leading=12, alignment=2, textColor=GRAY),
        "bullet": ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=9.5,
                                  leading=12, leftIndent=14, spaceAfter=1),
        "flat_item": ParagraphStyle("FlatItem", parent=styles["Normal"], fontSize=9.5,
                                     leading=12, leftIndent=14, spaceAfter=1),
    }


def _hr():
    return HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#d0d0d0"),
                       spaceBefore=1, spaceAfter=2)


def _ordered_sections(sections: list[dict]) -> list[dict]:
    """Stable-sort sections into CANONICAL_SECTION_ORDER, preamble excluded."""
    def sort_key(indexed):
        idx, section = indexed
        sec_type = section["type"]
        try:
            rank = CANONICAL_SECTION_ORDER.index(sec_type)
        except ValueError:
            rank = len(CANONICAL_SECTION_ORDER)
        return (rank, idx)

    non_preamble = [s for s in sections if s["type"] != "preamble"]
    indexed = list(enumerate(non_preamble))
    indexed.sort(key=sort_key)
    return [s for _, s in indexed]


def _linkify_contact(contact_text: str, links: dict) -> tuple[str, list[str]]:
    """
    Replace recognizable labels (LinkedIn, GitHub, ...) in the contact line
    with real hyperlinks, using ONLY urls the candidate's own resume actually
    contained (extracted PDF/DOCX hyperlink annotations, or a raw URL typed
    into the text). Never invents a URL. Returns (markup, unresolved_labels)
    so the caller can note in the change log when a label like "GitHub" was
    present in the text but no URL could be found for it anywhere.
    """
    if not contact_text:
        return contact_text, []

    unresolved = []
    result = contact_text
    for label, pattern in LINK_LABEL_PATTERNS.items():
        if label == "email":
            continue  # email already appears as visible text; linkify separately below
        match = pattern.search(result)
        if not match:
            continue
        url = links.get(label)
        if url:
            display = match.group(0)
            markup = f'<link href="{url}"><font color="#1d4ed8"><u>{display}</u></font></link>'
            result = result[:match.start()] + markup + result[match.end():]
        else:
            unresolved.append(match.group(0))

    email_match = LINK_LABEL_PATTERNS["email"].search(result)
    if email_match and "<link" not in result[max(0, email_match.start() - 6):email_match.start()]:
        addr = email_match.group(0)
        markup = f'<link href="mailto:{addr}"><font color="#1d4ed8"><u>{addr}</u></font></link>'
        result = result[:email_match.start()] + markup + result[email_match.end():]

    return result, unresolved


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def render_resume_pdf(state: dict, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.4 * inch, bottomMargin=0.4 * inch,
                             leftMargin=0.5 * inch, rightMargin=0.5 * inch)
    S = _styles()
    candidate = state["candidate"]
    sections = _ordered_sections(candidate.get("sections", []))
    excluded_ids = set(state.get("excluded_ids", []))
    latest_draft = state["draft_history"][-1] if state["draft_history"] else {"bullets": []}
    selected_ids = {b["source_id"] for b in latest_draft["bullets"] if b.get("source_id")} - excluded_ids

    # Precompute the normalized text of every bullet that will actually be
    # shown, so the summary section can be skipped if it's a verbatim
    # duplicate of one of them -- a summary is meant to add something the
    # bullets don't, not repeat one of them word-for-word.
    kept_bullet_texts = set()
    for section in sections:
        if section["type"] in resume_structure.GROUPED_SECTION_TYPES:
            for entry in section.get("entries", []):
                bullet_ids = entry.get("bullet_ids", [])
                for b_text, b_id in zip(entry["bullets"], bullet_ids):
                    if b_id in selected_ids:
                        kept_bullet_texts.add(_normalize(b_text))

    story = [
        Paragraph(candidate["name"], S["name"]),
    ]
    if candidate.get("contact"):
        contact_markup, _unresolved = _linkify_contact(candidate["contact"], candidate.get("links", {}))
        story.append(Paragraph(contact_markup, S["contact"]))
    story.append(_hr())

    summary_done = False
    for section in sections:
        sec_type = section["type"]

        if sec_type == "summary":
            text = " ".join(section.get("items", [])) or candidate.get("summary_raw", "")
            if not text:
                continue
            if _normalize(text) in kept_bullet_texts:
                # The summary is a verbatim copy of a bullet shown elsewhere
                # in the resume -- skip it rather than showing the same
                # sentence twice. summary_done stays False so the fallback
                # block below won't re-insert it either.
                continue
            story.append(Paragraph(SECTION_TITLES["summary"], S["section"]))
            story.append(Paragraph(text, S["body"]))
            summary_done = True

        elif sec_type == "skills":
            skills_found = sorted({s for ev in candidate["evidence"] for s in ev["skills"]})
            if not skills_found:
                continue
            story.append(Paragraph(SECTION_TITLES["skills"], S["section"]))
            pretty = ", ".join(s.title() if s.islower() else s for s in skills_found)
            story.append(Paragraph(pretty, S["body"]))

        elif sec_type in resume_structure.GROUPED_SECTION_TYPES:
            rendered_any = False
            block = []
            for entry in section.get("entries", []):
                bullet_ids = entry.get("bullet_ids", [])
                kept_bullets = [
                    b_text for b_text, b_id in zip(entry["bullets"], bullet_ids)
                    if b_id in selected_ids
                ]
                if not kept_bullets:
                    continue
                rendered_any = True
                header = entry.get("header") or ""
                title, date_str = resume_structure.split_header_and_date(header)
                if date_str:
                    row = Table(
                        [[Paragraph(title, S["job_title"]), Paragraph(date_str, S["job_date"])]],
                        colWidths=[doc.width * 0.72, doc.width * 0.28],
                    )
                    row.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ]))
                    block.append(row)
                elif title:
                    block.append(Paragraph(title, S["job_title"]))
                for b in kept_bullets:
                    block.append(Paragraph(f"&bull;&nbsp;&nbsp;{b}", S["bullet"]))
                block.append(Spacer(1, 2))
            if rendered_any:
                story.append(Paragraph(SECTION_TITLES.get(sec_type, sec_type.title()), S["section"]))
                story.extend(block)

        elif sec_type in ("achievements", "certifications"):
            # Trimmable flat sections: page-fit may have excluded specific
            # low-relevance lines here, unlike education/skills/summary which
            # are never filtered.
            item_ids = section.get("item_ids", [])
            items = section.get("items", [])
            pairs = [
                (line, iid) for line, iid in zip(items, item_ids)
                if iid not in excluded_ids
            ] if item_ids else [(line, None) for line in items]
            if not pairs:
                continue
            req_coverage = state.get("requirement_coverage", {})

            def _relevance(iid):
                if iid is None:
                    return 0
                return sum(1 for info in req_coverage.values() if iid in info["evidence_ids"])

            pairs.sort(key=lambda p: -_relevance(p[1]))
            story.append(Paragraph(SECTION_TITLES.get(sec_type, sec_type.title()), S["section"]))
            for line, _iid in pairs:
                story.append(Paragraph(f"&bull;&nbsp;&nbsp;{line}", S["flat_item"]))

        else:  # education / other: always shown in full, never filtered
            items = section.get("items", [])
            if not items:
                continue
            story.append(Paragraph(SECTION_TITLES.get(sec_type, sec_type.title()), S["section"]))
            for line in items:
                story.append(Paragraph(f"&bull;&nbsp;&nbsp;{line}", S["flat_item"]))

    fallback_summary = candidate.get("summary_raw", "")
    if (not summary_done and fallback_summary
            and _normalize(fallback_summary) not in kept_bullet_texts):
        story.insert(3, Paragraph(fallback_summary, S["body"]))
        story.insert(3, Paragraph(SECTION_TITLES["summary"], S["section"]))

    doc.build(story)
    return out_path