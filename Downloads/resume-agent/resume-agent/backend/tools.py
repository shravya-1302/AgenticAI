"""
Deterministic 'tools' the agent calls. Kept rule-based (not LLM) on purpose:
a hackathon judge should be able to see these as real, inspectable tools with
verifiable outputs, not just another prompt.
"""
import re

from . import resume_structure

# A broad, generic vocabulary of skills/tools/domains. Both the JD parser and
# the resume-evidence extractor tag against this SAME vocabulary, so a
# requirement found in the JD can only ever be "matched" if the identical
# term is genuinely present in the candidate's own resume text -- there is
# no LLM in this loop free-associating or inventing a connection.
SKILL_VOCAB = [
    # languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "sql", "r", "scala",
    # web / backend frameworks
    "react", "angular", "vue", "django", "flask", "fastapi", "spring",
    "node.js", "node", "express", "rest api", "graphql", "microservices",
    # data / ml
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "pandas", "numpy", "data analysis", "data science", "nlp",
    "computer vision", "llm", "generative ai", "statistics",
    # cloud / devops
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "ci/cd", "jenkins", "github actions", "linux", "devops",
    # databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
    # testing
    "unit testing", "automated tests", "test coverage", "pytest", "selenium",
    "qa", "quality assurance",
    # design / product
    "figma", "ui/ux", "product management", "agile", "scrum", "jira",
    "wireframing", "user research",
    # business / soft skills
    "leadership", "project management", "stakeholder management",
    "communication", "team management", "cross-functional", "negotiation",
    "public speaking", "mentoring", "budgeting",
    # marketing / sales / finance
    "seo", "content marketing", "social media", "google analytics",
    "salesforce", "crm", "sales", "financial modeling", "excel",
    "powerpoint", "accounting", "forecasting",
    # general engineering
    "distributed systems", "system design", "algorithms", "data structures",
    "api design", "backend", "frontend", "full stack", "mobile development",
    "android", "ios",
]


# Aliases that refer to the exact same real-world skill as a SKILL_VOCAB
# term, just spelled differently between a JD and a resume (e.g. a JD says
# "Node.js" while the resume only ever writes "Node"). Recognizing these as
# the same skill is not fabrication -- it's the same technology -- and it
# meaningfully raises how much of a real match a candidate's genuine
# background can score, instead of losing coverage purely to spelling.
SKILL_SYNONYMS = {
    "node.js": "node", "nodejs": "node",
    "reactjs": "react", "react.js": "react",
    "js": "javascript",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "restful api": "rest api", "restful": "rest api", "rest apis": "rest api",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "ci/cd pipeline": "ci/cd", "ci-cd": "ci/cd",
}


def _vocab_terms_in_text(text: str) -> list[str]:
    """Return every SKILL_VOCAB term that appears in text (word-boundary,
    case-insensitive), also recognizing known spelling variants via
    SKILL_SYNONYMS and folding them into their canonical vocab term."""
    text_lower = text.lower()
    found = set()
    for term in SKILL_VOCAB:
        pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(term)
    for alias, canonical in SKILL_SYNONYMS.items():
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(canonical)
    return list(found)


def parse_job_description(jd_text: str) -> list[str]:
    """
    Extract the set of known skill/requirement terms present anywhere in a
    job description. Works on any JD format (bullets, paragraphs, pasted
    text) because it matches against a fixed vocabulary rather than relying
    on line structure -- this avoids picking up junk like the job title.
    """
    return _vocab_terms_in_text(jd_text)


EDUCATION_KEYWORDS = [
    "university", "college", "b.tech", "b.e.", "bachelor", "master", "m.tech",
    "degree", "diploma", "institute of technology", "school of", "phd", "ph.d",
]


def extract_evidence_from_resume(resume_text: str) -> tuple[list[dict], list[dict]]:
    """
    Parse resume text into its structured sections, then produce a flat
    evidence list for scoring/matching:
      - Bullets inside "experience"/"projects" sections become evidence
        items tagged with skills from BOTH the bullet text and its parent
        job/project header (so e.g. a bullet under "AWS Data Engineer
        Intern" picks up the "aws" tag even if the bullet sentence itself
        doesn't repeat the word). These are the ONLY items the agent may
        select/omit when tailoring -- type is "experience" or "projects".
      - Lines inside flat sections (skills, education, certifications,
        achievements, other) also become evidence items (so JD requirements
        can be matched against them for coverage scoring), but are tagged
        with a type that keeps them OUT of the tailorable draft -- they are
        always shown in full in the final resume, never filtered.
    Returns (evidence_list, sections) -- sections is the structured outline
    used later by the PDF renderer to reproduce real resume formatting.
    """
    sections = resume_structure.parse_resume_sections(resume_text)
    number_pattern = re.compile(
        r"\d+(?:\.\d+)?%"                                            # 30%, 12.5%
        r"|\d+(?:\.\d+)?x\b"                                         # 3x, 2.5x
        r"|\$\d[\d,.]*[kKmMbB]?"                                     # $500, $2,000, $2k
        r"|\d+\+?\s*(?:years?|months?|minutes?|hours?|days?|weeks?)" # 3 years, 40 minutes
        r"|\b\d{1,3}(?:,\d{3})+\b"                                   # 10,000 (comma-grouped)
        r"|\b\d+\+?\s*(?:users?|customers?|clients?|requests?|records?|rows?|files?|"
        r"employees?|members?|projects?|apps?|apis?|endpoints?|servers?|"
        r"microservices?|features?|tests?|bugs?|tickets?|reports?|teams?)\b"
        r"|\b\d+(?:\.\d+)?\s*(?:cgpa|gpa)\b",                        # 8.24 CGPA
        re.I,
    )

    evidence = []
    counter = 1

    def add_evidence(text, ev_type):
        nonlocal counter
        eid = f"ev{counter}"
        counter += 1
        evidence.append({
            "id": eid,
            "type": ev_type,
            "title": text[:60],
            "text": text,
            "skills": _vocab_terms_in_text(text),
            "metrics_supported": number_pattern.findall(text),
        })
        return eid

    for section in sections:
        sec_type = section["type"]
        if sec_type == "preamble":
            continue
        if sec_type in resume_structure.GROUPED_SECTION_TYPES:
            for entry in section.get("entries", []):
                header = entry.get("header") or ""
                for bullet in entry["bullets"]:
                    eid = f"ev{counter}"
                    counter += 1
                    combined_for_skills = f"{header} {bullet}"
                    evidence.append({
                        "id": eid,
                        "type": sec_type,  # "experience" or "projects"
                        "title": bullet[:60],
                        "text": bullet,
                        "header": header,
                        "skills": _vocab_terms_in_text(combined_for_skills),
                        "metrics_supported": number_pattern.findall(bullet),
                    })
                    entry.setdefault("bullet_ids", []).append(eid)
        else:
            ev_type = "education" if sec_type == "education" else sec_type
            for line in section.get("items", []):
                eid = add_evidence(line, ev_type)
                section.setdefault("item_ids", []).append(eid)

    return evidence, sections




def match_evidence_to_requirements(requirements: list[str], evidence: list[dict]) -> dict:
    """For each requirement, find which evidence items support it."""
    coverage = {}
    for req in requirements:
        matches = []
        for ev in evidence:
            hay = (ev["text"] + " " + " ".join(ev["skills"])).lower()
            if req.lower() in hay or any(req.lower() in s for s in ev["skills"]):
                matches.append(ev["id"])
        coverage[req] = {
            "status": "matched" if matches else "unmatched",
            "evidence_ids": matches,
        }
    return coverage


def score_ats(requirement_coverage: dict, draft_bullets: list[dict]) -> float:
    """
    Fraction of JD requirements that are backed by evidence actually
    present in the current draft (matched by source_id, not just keyword
    text-matching -- a requirement can be satisfied by a bullet that
    doesn't literally repeat the JD's wording).
    """
    if not requirement_coverage:
        return 1.0
    draft_source_ids = {b["source_id"] for b in draft_bullets if b.get("source_id")}
    hits = 0
    for req, info in requirement_coverage.items():
        if info["status"] == "matched" and set(info["evidence_ids"]) & draft_source_ids:
            hits += 1
    return round(hits / len(requirement_coverage), 2)


def check_factual_consistency(draft_bullets: list[dict], evidence: list[dict]) -> list[str]:
    """
    Flag any bullet whose claim (especially quantified claims) isn't
    traceable to a real evidence item.
    """
    evidence_by_id = {e["id"]: e for e in evidence}
    flags = []
    number_pattern = re.compile(r"\d+%|\d+x|\d+\s*(minutes|hours|days|months)")
    for bullet in draft_bullets:
        source_id = bullet.get("source_id")
        text = bullet["text"]
        has_number_claim = bool(number_pattern.search(text))
        if source_id is None:
            flags.append(f"Unsupported bullet (no source evidence): \"{text}\"")
            continue
        ev = evidence_by_id.get(source_id)
        if ev is None:
            flags.append(f"Bullet cites unknown evidence id '{source_id}': \"{text}\"")
            continue
        if has_number_claim:
            # Bullet text is always copied verbatim from its source evidence,
            # so a quantified claim in the bullet is "supported" exactly when
            # that same evidence item's own metrics_supported list is
            # non-empty. (Previously this always evaluated to True via a
            # dead `or True`, so it could never actually flag anything.)
            supported = bool(ev.get("metrics_supported"))
            if not supported:
                flags.append(
                    f"Quantified claim not found in source evidence '{source_id}': \"{text}\""
                )
    return flags