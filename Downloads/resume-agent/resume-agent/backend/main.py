import os
import re
from collections import Counter
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from . import agent, pdf_render, tools, resume_parser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

app = FastAPI(title="Autonomous Resume & Application Agent")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/session/start")
async def start_session(
    jd_text: str = Form(default=""),
    resume_text: str = Form(default=""),
    resume_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
):
    # --- resolve job description text (pasted text takes priority, else file) ---
    if jd_file is not None and jd_file.filename:
        jd_bytes = await jd_file.read()
        jd_text = resume_parser.extract_text_from_upload(jd_file.filename, jd_bytes)
    jd_text = (jd_text or "").strip()
    if not jd_text:
        raise HTTPException(400, "Please paste or upload a job description.")

    # --- resolve resume text (pasted text takes priority, else file) ---
    # Links are only extractable from an uploaded file's real hyperlink
    # annotations/relationships (PDF/DOCX); a pasted-text resume can only
    # ever contribute a URL that's literally typed out in the text.
    links = {}
    if resume_file is not None and resume_file.filename:
        resume_bytes = await resume_file.read()
        resume_text = resume_parser.extract_text_from_upload(resume_file.filename, resume_bytes)
        links = resume_parser.extract_links_from_upload(resume_file.filename, resume_bytes)
    resume_text = (resume_text or "").strip()
    if not resume_text:
        raise HTTPException(400, "Please paste or upload your resume.")
    for label, url in resume_parser.extract_links_from_text(resume_text).items():
        links.setdefault(label, url)

    # --- build a real candidate record from the actual resume text ---
    name, contact = resume_parser.guess_name_and_contact(resume_text)
    evidence, sections = tools.extract_evidence_from_resume(resume_text)
    if not evidence:
        raise HTTPException(400, "Couldn't find any usable content lines in that resume text.")

    candidate = {
        "name": name,
        "contact": contact,
        "summary_raw": _guess_summary(evidence, contact, sections),
        "evidence": evidence,
        "sections": sections,
        "links": links,
    }

    state = agent.start_session(candidate, jd_text, demo_mode=False)
    return {"session_id": state["session_id"]}


def _guess_summary(evidence: list[dict], contact: str, sections: list[dict]) -> str:
    """Prefer an explicit summary/objective section if the resume has one;
    else synthesize a short summary from real facts (job/project titles +
    most-frequent real skills) instead of copying a bullet verbatim.
    Copying a bullet caused the summary to be textually identical to a
    bullet shown later in the resume body, which pdf_render.py's dedup
    check then silently suppressed -- this never repeats bullet text, so
    it can't collide."""
    for section in sections:
        if section["type"] == "summary" and section.get("items"):
            return " ".join(section["items"])[:400]

    # Real job/project titles (headers), not bullet sentences.
    titles = []
    for section in sections:
        if section["type"] in ("experience", "projects"):
            for entry in section.get("entries", []):
                header = entry.get("header")
                if header:
                    title_only = re.split(r"\s{2,}|\t|(?:19|20)\d{2}", header)[0].strip(" -–—")
                    if title_only and title_only not in titles:
                        titles.append(title_only)

    # Most frequently-mentioned real skills across all evidence.
    skill_counts = Counter(s for ev in evidence for s in ev.get("skills", []))
    top_skills = [s for s, _ in skill_counts.most_common(6)]

    if not titles and not top_skills:
        return ""

    role_part = f"Experience across {', '.join(titles[:3])}" if titles else "Hands-on project experience"
    skill_part = f"skilled in {', '.join(top_skills)}" if top_skills else ""
    summary = f"{role_part}{'; ' + skill_part if skill_part else ''}."
    return summary[:400]


@app.post("/session/{session_id}/step")
def step_session(session_id: str):
    if session_id not in agent.SESSIONS:
        raise HTTPException(404, "session not found")
    state = agent.run_iteration(session_id)
    return _public_state(state)


@app.post("/session/{session_id}/run")
def run_session(session_id: str):
    """Run the agent all the way to completion (SELECT phase through FIT
    phase) in a single call, instead of relying on the client to keep
    calling /step enough times. This is what the frontend's "Run to
    completion" button should hit -- calling /step in a capped client-side
    loop can stop before the page-fit trimming has actually converged."""
    if session_id not in agent.SESSIONS:
        raise HTTPException(404, "session not found")
    state = agent.run_to_completion(session_id)
    return _public_state(state)


@app.get("/session/{session_id}/state")
def get_state(session_id: str):
    if session_id not in agent.SESSIONS:
        raise HTTPException(404, "session not found")
    return _public_state(agent.SESSIONS[session_id])


@app.get("/session/{session_id}/resume.pdf")
def get_resume_pdf(session_id: str):
    if session_id not in agent.SESSIONS:
        raise HTTPException(404, "session not found")
    state = agent.SESSIONS[session_id]
    if not state["draft_history"]:
        raise HTTPException(400, "no draft generated yet")
    out_path = os.path.join(OUTPUT_DIR, f"{session_id}_resume.pdf")
    pdf_render.render_resume_pdf(state, out_path)
    return FileResponse(out_path, media_type="application/pdf", filename="tailored_resume.pdf")


def _public_state(state: dict) -> dict:
    """Trim internal bookkeeping keys before sending to the frontend."""
    return {k: v for k, v in state.items() if not k.startswith("_")}