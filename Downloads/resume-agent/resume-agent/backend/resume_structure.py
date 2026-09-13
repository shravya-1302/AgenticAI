"""
Parses raw resume text into a structured outline: section headers, and
within "experience"/"projects" sections, job/project titles with their own
bullet points grouped underneath -- instead of treating every line as an
independent, disconnected bullet. This is what lets the final PDF look like
an actual resume (job title + dates, then its bullets) rather than a flat
dump.
"""
import re

BULLET_CHARS = "-•*■♦▪●○▶►‣"
FLAT_SECTION_TYPES = {"summary", "skills", "education", "certifications", "achievements", "other"}
GROUPED_SECTION_TYPES = {"experience", "projects"}

SECTION_KEYWORDS = {
    "summary": ["summary", "objective", "profile"],
    "skills": ["skill", "competenc", "technical", "technolog"],
    "experience": ["experience", "employment", "work history", "internship"],
    "projects": ["project"],
    "education": ["education", "qualification", "academic"],
    "certifications": ["certification", "certificate", "license"],
    "achievements": ["achievement", "award", "honor", "honour", "accomplishment"],
}

DATE_RANGE_PATTERN = re.compile(
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}.{0,15}"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}"
    r"|\b\d{4}\s*[-–—]\s*(?:\d{4}|present|current)\b",
    re.IGNORECASE,
)

# Generic column-header words that show up in tabular Education sections
# (e.g. "Degree | Institution | CGPA | Year") when a PDF table's header row
# gets flattened into plain text alongside the data rows. A line built
# almost entirely out of these labels, with no digit anywhere in it, is a
# leftover header row, not a real education entry -- skip it. This is a
# generic label set, not tied to any one resume's wording.
EDUCATION_HEADER_LABELS = {
    "degree", "institution", "university", "college", "school", "board",
    "course", "stream", "branch", "cgpa", "gpa", "marks", "percentage",
    "score", "year", "duration", "grade", "result",
}

# Common resume action verbs. A bulleted line that starts with one of these
# is almost certainly a real accomplishment sentence, not a role/sub-entry
# title -- used to disambiguate below.
ACTION_VERB_PATTERN = re.compile(
    r"^(built|developed|created|designed|implemented|led|managed|wrote|gained|"
    r"practiced|worked|collaborated|analyzed|automated|improved|increased|reduced|"
    r"optimized|launched|deployed|maintained|architected|delivered|coordinated|"
    r"conducted|executed|drove|owned|spearheaded|streamlined|enhanced|established|"
    r"initiated|resolved|debugged|tested|integrated|migrated|scaled|mentored|"
    r"trained|presented|researched|authored|published|achieved|earned|completed|"
    r"contributed|participated|assisted|supported|handled|performed|utilized|"
    r"leveraged|configured)\b",
    re.IGNORECASE,
)


def _is_bullet_line(line: str) -> bool:
    return line.strip()[:1] in BULLET_CHARS


def _strip_bullet(line: str) -> str:
    return line.strip().lstrip(BULLET_CHARS).strip()


def _is_education_header_row(line: str) -> bool:
    """
    True if `line` is a leftover table header row (e.g. "Degree Institution
    CGPA/Marks Year") rather than a real education entry: no digits
    anywhere, and every token is a generic column-label word.
    """
    if any(ch.isdigit() for ch in line):
        return False
    tokens = [t for t in re.split(r"[\s/|,]+", line.strip().lower()) if t]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in EDUCATION_HEADER_LABELS)
    return hits >= 2 and hits == len(tokens)


DANGLING_ENDING_PATTERN = re.compile(
    r"\b(using|with|for|and|of|to|in|on|by|from|via|through|including|as)$",
    re.IGNORECASE,
)


def _is_dangling(text: str) -> bool:
    """
    True if `text` reads as an incomplete clause -- ends on a bare
    preposition/conjunction with no terminal punctuation. This almost
    always means a bullet's sentence was hard-wrapped across lines in the
    source document and its tail landed on its own line (sometimes even
    picking up a stray bullet glyph from the source PDF's own list
    formatting, which is why this check doesn't rely on the next line
    lacking a bullet character).
    """
    stripped = text.strip()
    if stripped.endswith((".", ",", ";", ":")):
        return False
    return bool(DANGLING_ENDING_PATTERN.search(stripped))


def _looks_like_subentry_title(text: str) -> bool:
    """
    A bulleted line inside Experience/Projects is normally an accomplishment
    sentence ("Built X", "Reduced Y by Z%"). Some resumes instead put a
    second job/internship title under the same bullet-character style as
    its own accomplishments, with no separate heading line. Treat a bulleted
    line as a new sub-entry header -- not content of the current one --
    when it reads like a title rather than an action: no terminal sentence
    punctuation, reasonably short, contains an organization separator, and
    does not start with a resume action verb.
    """
    stripped = text.strip()
    if not stripped or stripped.endswith((".", ",", ";")):
        return False
    if ACTION_VERB_PATTERN.match(stripped):
        return False
    # Separator between a role and its organization: dash variants, "@",
    # comma-before-org ("... Intern, ExcelR"), pipe, or "at Org". Widened
    # from just dash/@ because real resumes format this many different ways.
    has_org_separator = bool(re.search(r"[-–—@|]|,\s*[A-Z]|\bat\s+[A-Z]", stripped))
    # A role+org title with a couple of parenthetical qualifiers (e.g.
    # "(AWS & Azure)", "(APSCHE)") can easily run past 10 words -- widened
    # so those don't get misread as an accomplishment sentence.
    is_short = len(stripped.split()) <= 14
    return has_org_separator and is_short


def _classify_section_title(line: str):
    """Return a section type if this line looks like a section heading, else None."""
    stripped = line.strip().strip(":").strip()
    if not stripped or len(stripped) > 45:
        return None
    if stripped[:1] in BULLET_CHARS:
        return None  # bullet content, never a heading
    if stripped.endswith((".", ",", ";")):
        return None  # headings don't end in sentence punctuation
    word_count = len(stripped.split())
    if word_count > 6:
        return None
    lower = stripped.lower()
    for sec_type, keywords in SECTION_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return sec_type
    # Fallback: an unlabeled but clearly shouty ALL-CAPS line (e.g. a
    # custom heading like "CERTIFICATIONS & AWARDS" not in our keyword
    # list) is still treated as a generic section heading.
    letters = [c for c in stripped if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.8:
        return "other"
    return None


def parse_resume_sections(resume_text: str) -> list[dict]:
    """
    Returns an ordered list of sections:
      {"type": "summary"|"skills"|"experience"|"projects"|"education"|
                "certifications"|"achievements"|"other"|"preamble",
       "title": "<original heading text>",
       "items": [...]}   # for FLAT_SECTION_TYPES: list of plain line strings
       "entries": [...]  # for GROUPED_SECTION_TYPES: list of
                          # {"header": str|None, "bullets": [str, ...]}
    The first section (before any recognizable heading) is type "preamble"
    and holds the name/contact lines.
    """
    lines = [l.strip() for l in resume_text.splitlines()]
    sections = []
    current = {"type": "preamble", "title": "", "items": []}
    sections.append(current)

    seen_first_line = False
    for line in lines:
        if not line:
            continue
        # The very first non-blank line of a resume is virtually always the
        # candidate's name -- never treat it as a section heading, even if
        # it happens to be short and all-caps.
        if not seen_first_line:
            seen_first_line = True
            current["items"].append(line)
            continue

        sec_type = _classify_section_title(line)
        if sec_type:
            current = {"type": sec_type, "title": line}
            if sec_type in GROUPED_SECTION_TYPES:
                current["entries"] = []
            else:
                current["items"] = []
            sections.append(current)
            continue

        if current["type"] in GROUPED_SECTION_TYPES:
            current_entry = current["entries"][-1] if current["entries"] else None
            prev_bullet_dangling = bool(
                current_entry and current_entry["bullets"]
                and _is_dangling(current_entry["bullets"][-1])
            )

            if _is_bullet_line(line):
                text = _strip_bullet(line)
                if not text:
                    continue
                # If the previous bullet was cut off mid-clause (a wrapped
                # line in the source PDF), this line is its tail, not a new
                # bullet of its own -- even if it happens to carry its own
                # bullet glyph (some source PDFs put one on every wrapped
                # line of a hanging-indent list).
                if prev_bullet_dangling and not _looks_like_subentry_title(text):
                    current_entry["bullets"][-1] = (
                        current_entry["bullets"][-1].rstrip() + " " + text
                    ).strip()
                    continue
                # A bulleted line can still be a second job/project title
                # rather than an accomplishment -- some resumes format a
                # sub-role with the same bullet character as its own
                # bullets, with no separate heading line above it.
                if current["entries"] and _looks_like_subentry_title(text):
                    current["entries"].append({"header": text, "bullets": []})
                    continue
                if not current["entries"]:
                    current["entries"].append({"header": None, "bullets": []})
                current["entries"][-1]["bullets"].append(text)
            else:
                # A dangling previous bullet takes priority over title
                # detection -- an incomplete sentence's continuation should
                # never be mistaken for a new job/project heading.
                if prev_bullet_dangling:
                    current_entry["bullets"][-1] = (
                        current_entry["bullets"][-1].rstrip() + " " + line.strip()
                    ).strip()
                    continue
                # No bullet character on this line -- decide whether it's a
                # new job/project title or a plain-sentence achievement line
                # (some PDF/text extractions strip bullet glyphs entirely).
                # A line is treated as a new title if it contains a date
                # range (job headers usually do), reads like a role/org
                # title (see _looks_like_subentry_title), or reads short and
                # title-like (no terminal sentence punctuation); otherwise
                # it's content under the current entry.
                is_title_like = (
                    bool(DATE_RANGE_PATTERN.search(line))
                    or _looks_like_subentry_title(line)
                    or (len(line.split()) <= 8 and not line.rstrip().endswith((".", ",")))
                )
                if is_title_like or not current["entries"]:
                    current["entries"].append({"header": line, "bullets": []})
                else:
                    current["entries"][-1]["bullets"].append(line)
        else:
            text = _strip_bullet(line) if _is_bullet_line(line) else line
            if text:
                # Skip a leftover table-header row at the top of Education
                # (e.g. "Degree Institution CGPA/Marks Year") -- it's a
                # flattened table label row, not a real education entry.
                if (current["type"] == "education" and not current["items"]
                        and _is_education_header_row(text)):
                    continue
                current["items"].append(text)

    return sections


def split_header_and_date(header_line: str) -> tuple[str, str]:
    """Pull a trailing date range out of a job/project header line, if present."""
    match = DATE_RANGE_PATTERN.search(header_line)
    if not match:
        return header_line.strip(), ""
    date_str = match.group(0).strip()
    title = (header_line[:match.start()] + header_line[match.end():]).strip(" -–—|,")
    return title or header_line.strip(), date_str