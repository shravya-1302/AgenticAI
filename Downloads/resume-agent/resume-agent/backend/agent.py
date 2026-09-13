"""
Core agent state machine for PS11: Autonomous Resume & Application Agent.

Two-phase loop:

  PHASE 1 -- SELECT  (DRAFT -> EVALUATE -> DECIDE)
    Grow the set of Experience/Projects bullets pulled into the draft,
    ranked so that bullets covering more JD requirements -- and bullets
    that carry a real quantified metric -- are preferred. Stops once ATS
    coverage clears the threshold, or once every genuinely relevant piece
    of evidence has been used (a real coverage gap, not a length problem).

  PHASE 2 -- FIT  (RENDER -> VERIFY -> ADAPT)
    Actually render the PDF and count its pages with the same library a
    human would open it in (pdfplumber) -- no character-count guessing.
    If it's more than one page, remove the single least-relevant piece of
    *optional* content (a non-academic achievement or certification not
    tied to anything in the JD; only as a last resort, the weakest bullet)
    and render again. Repeat until it fits one page or nothing left is
    safe to cut. Education, skills, and summary are never touched -- only
    achievements/certifications/bullets are ever trimmable.

Design principle (unchanged): the agent may only ever use evidence the
candidate actually wrote. If a JD requirement has zero supporting evidence
anywhere in the resume, the agent does not invent anything to cover it --
it says so plainly in the change log and still delivers a resume. This
mirrors the hackathon guardrail: "must never fabricate skills, projects,
credentials, employment, or achievements." A real gap in the candidate's
background is not a reason to block delivery -- it's a fact the human
reading the change log deserves to see stated clearly.
"""
import os
import tempfile
import uuid
import time

from . import tools

MAX_SELECT_ITERATIONS = 8
MAX_FIT_ITERATIONS = 30
# score_ats is (JD requirements covered by evidence actually in the draft) /
# (ALL requirement terms our fixed vocabulary found in the JD text) -- and
# that denominator includes "nice-to-have" terms no single real candidate
# will ever fully cover. Requiring 90% of that raw set was effectively
# unreachable for most real JD/resume pairs; run_iteration's all_evidence_used
# fallback always delivers a resume regardless, but the score itself is more
# useful as a genuine, attainable target. 0.75 still means "clearly tailored
# to this JD" without demanding coverage of every bonus/optional line item.
ATS_PASS_THRESHOLD = 0.75

INITIAL_STEP = 2   # how many evidence items to add per replanning step

# Evidence types that are allowed to be cut, in the order the agent will
# reach for them, when the rendered resume overflows one page. Achievements
# and certifications go first because they are the least load-bearing part
# of a resume next to actual experience/education; bullets are the very
# last resort, and even then only the single weakest one at a time.
TRIMMABLE_FLAT_TYPES = ["achievements", "certifications"]

SESSIONS: dict[str, dict] = {}  # in-memory store; fine for a demo/hackathon


def start_session(candidate: dict, jd_text: str, demo_mode: bool = False) -> dict:
    session_id = str(uuid.uuid4())[:8]
    requirements = tools.parse_job_description(jd_text)
    coverage = tools.match_evidence_to_requirements(requirements, candidate["evidence"])
    unmatched = [r for r, info in coverage.items() if info["status"] == "unmatched"]

    state = {
        "session_id": session_id,
        "candidate": candidate,
        "jd_text": jd_text,
        "requirements": requirements,
        "requirement_coverage": coverage,
        "draft_version": 0,
        "draft_history": [],
        "eval_history": [],
        "change_log": [],
        "status": "in_progress",
        "demo_mode": demo_mode,
        "_k": INITIAL_STEP,
        "_phase": "select",
        "_fit_iterations": 0,
        "excluded_ids": [],   # evidence ids trimmed for page-fit; kept as a
                               # list (not a set) so the public state stays
                               # JSON-serializable for the frontend.
        "log": [],
    }
    _log(state, f"Session started. Parsed {len(requirements)} requirements from JD.")
    if unmatched:
        _log(state, f"No evidence found yet for: {', '.join(unmatched)} (may still be covered once more evidence is drafted in).")
    SESSIONS[session_id] = state
    return state


def _log(state: dict, message: str):
    state["log"].append({"t": round(time.time(), 2), "msg": message})


def _req_count_for_evidence(state: dict, eid: str) -> int:
    coverage = state["requirement_coverage"]
    return sum(1 for info in coverage.values() if eid in info["evidence_ids"])


def _rank_evidence_by_relevance(state: dict) -> list[str]:
    """
    Rank Experience/Projects evidence best-first using three tiers, in
    priority order:
      1. More JD requirements covered wins.
      2. Among ties, a bullet with a quantified metric (a %, a $ figure,
         a count, a time saved, etc.) wins over one without.
      3. Among further ties, "projects" is placed ahead of "experience".
         Projects usually demonstrate the actual JD-relevant tech stack
         more concretely than a generic internship bullet does, so they
         get priority when everything else about two bullets is equal.
    Evidence with zero requirement coverage is dropped entirely -- this
    naturally excludes irrelevant projects/roles from ever being drafted.
    """
    evidence_by_id = {e["id"]: e for e in state["candidate"]["evidence"]}
    experience_ids = {
        e["id"] for e in state["candidate"]["evidence"]
        if e["type"] in ("experience", "projects")
    }
    req_count = {eid: _req_count_for_evidence(state, eid) for eid in experience_ids}
    relevant = [eid for eid, c in req_count.items() if c > 0]

    def sort_key(eid):
        ev = evidence_by_id[eid]
        has_metric = bool(ev.get("metrics_supported"))
        is_project = ev["type"] == "projects"
        return (-req_count[eid], 0 if has_metric else 1, 0 if is_project else 1)

    relevant.sort(key=sort_key)
    return relevant


def _draft_bullets(state: dict, selected_ids: list[str]) -> list[dict]:
    evidence_by_id = {e["id"]: e for e in state["candidate"]["evidence"]}
    return [{"text": evidence_by_id[eid]["text"], "source_id": eid} for eid in selected_ids]


def evidence_has_metric(state: dict, evidence_id: str) -> bool:
    for ev in state["candidate"]["evidence"]:
        if ev["id"] == evidence_id:
            return bool(ev.get("metrics_supported"))
    return False


def get_state(session_id: str) -> dict:
    return SESSIONS[session_id]


def run_to_completion(session_id: str, max_steps: int = MAX_SELECT_ITERATIONS + MAX_FIT_ITERATIONS + 5) -> dict:
    state = SESSIONS[session_id]
    steps = 0
    while state["status"] == "in_progress" and steps < max_steps:
        run_iteration(session_id)
        steps += 1
    if state["status"] == "in_progress":
        # Safety net: never leave a session hanging with no deliverable.
        state["status"] = "ready"
        state["change_log"].append(
            "Reached the internal step limit before fully converging. Delivering the best draft "
            "produced so far rather than leaving the session stuck."
        )
        _log(state, "Step limit reached; delivering best-effort draft instead of stalling.")
    return state


def run_iteration(session_id: str) -> dict:
    """Advance the agent one step, in whichever phase it's currently in."""
    state = SESSIONS[session_id]
    if state["status"] != "in_progress":
        _log(state, "Session already finished; no further iterations.")
        return state

    if state["_phase"] == "select":
        return _run_select_iteration(state)
    return _run_fit_iteration(state)


# ---------------------------------------------------------------------------
# PHASE 1: content selection (which bullets belong in the draft at all)
# ---------------------------------------------------------------------------

def _run_select_iteration(state: dict) -> dict:
    if state["draft_version"] >= MAX_SELECT_ITERATIONS:
        _log(state, "Selection step limit reached; moving to page-fit verification with current draft.")
        state["_phase"] = "fit"
        return state

    k = state["_k"]
    ranked = _rank_evidence_by_relevance(state)
    selected_ids = ranked[:k]
    bullets = _draft_bullets(state, selected_ids)
    state["draft_version"] += 1
    state["draft_history"].append({"version": state["draft_version"], "bullets": bullets})
    _log(state, f"Drafted resume v{state['draft_version']} using top {len(bullets)} most relevant evidence item(s).")

    ats_score = tools.score_ats(state["requirement_coverage"], bullets)
    flags = tools.check_factual_consistency(bullets, state["candidate"]["evidence"])
    quantified = [b for b in bullets if evidence_has_metric(state, b["source_id"])]
    impact_ratio = round(len(quantified) / len(bullets), 2) if bullets else 0.0
    state["eval_history"].append({
        "version": state["draft_version"],
        "ats_score": ats_score,
        "factual_flags": flags,
        "quantified_impact_ratio": impact_ratio,
    })
    _log(state, f"Evaluated v{state['draft_version']}: ATS coverage={ats_score}, "
                f"quantified-impact ratio={impact_ratio}, factual flags={len(flags)}.")

    if quantified:
        quoted = "; ".join(f"\"{b['text'][:70]}\"" for b in quantified[:3])
        state["change_log"].append(
            f"v{state['draft_version']}: {len(quantified)} of {len(bullets)} selected bullet(s) "
            f"carry a quantified metric and were prioritized for that reason, e.g. {quoted}."
        )

    if flags:
        # The one genuine escalation case left: an unverifiable claim would
        # mean fabricating or misrepresenting evidence, which the agent
        # will not silently patch over.
        state["status"] = "escalated"
        state["change_log"].append(
            f"v{state['draft_version']}: factual-consistency check found unverifiable claims. "
            "Escalated for human review rather than auto-editing or dropping the claim silently."
        )
        _log(state, "Factual-consistency flags found. Escalating rather than fabricating a fix.")
        return state

    total_relevant_evidence = len(ranked)
    all_evidence_used = len(selected_ids) >= total_relevant_evidence
    passed = ats_score >= ATS_PASS_THRESHOLD

    if passed and not all_evidence_used:
        # ATS coverage already clears the threshold, but there is still
        # genuine, relevant evidence sitting unused -- drafting only "just
        # enough to pass" leaves real content out and blank space on the
        # page. Pull in everything relevant; the FIT phase already knows
        # how to trim back down if that overflows one page, so this never
        # risks a worse outcome than stopping early did.
        state["_k"] = total_relevant_evidence
        _log(state, "ATS threshold already met, but relevant evidence remains unused -- "
                     "expanding draft to use all of it before checking page fit.")
        return state

    if passed or all_evidence_used:
        if not passed:
            unmet = [
                req for req, info in state["requirement_coverage"].items()
                if info["status"] == "unmatched"
            ]
            note = (
                f"All available evidence is included (ATS coverage {ats_score}). "
                f"No genuine evidence found in the candidate's resume for: "
                f"{', '.join(unmet) if unmet else 'the remaining requirements'}. "
                "Not fabricating content to close this gap -- the resume below is honest about "
                "what the candidate can currently support; add real experience/projects covering "
                "these and re-run if you want to close it."
            )
            state["change_log"].append(note)
            _log(state, "Real coverage gap: no more genuine evidence to draw on. Proceeding to render anyway.")
        else:
            _log(state, "ATS coverage threshold met. Proceeding to page-fit verification.")
        state["_phase"] = "fit"
        return state

    new_k = min(k + INITIAL_STEP, total_relevant_evidence)
    added_ids = ranked[k:new_k]
    newly_covered = sorted({
        req for req, info in state["requirement_coverage"].items()
        if any(eid in info["evidence_ids"] for eid in added_ids)
    })
    state["_k"] = new_k
    note = f"v{state['draft_version']}->v{state['draft_version']+1}: ATS coverage {ats_score} below {ATS_PASS_THRESHOLD} threshold."
    if newly_covered:
        note += f" Re-mined evidence store; added {len(added_ids)} more item(s) covering: {', '.join(newly_covered)}."
    state["change_log"].append(note)
    _log(state, f"Replanning: {note}")
    return state


# ---------------------------------------------------------------------------
# PHASE 2: page-fit verification (does the rendered PDF actually fit one page)
# ---------------------------------------------------------------------------

def _run_fit_iteration(state: dict) -> dict:
    from . import pdf_render  # local import: avoids a circular import at module load time
    import pdfplumber

    state["_fit_iterations"] += 1
    if state["_fit_iterations"] > MAX_FIT_ITERATIONS:
        state["status"] = "ready"
        state["change_log"].append(
            "Reached the page-fit trim limit. Delivering the current draft rather than cutting "
            "further into core content."
        )
        _log(state, "Fit-phase iteration limit reached; delivering best-effort layout.")
        return state

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pdf_render.render_resume_pdf(state, tmp_path)
        with pdfplumber.open(tmp_path) as pdf:
            page_count = len(pdf.pages)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    _log(state, f"Rendered draft v{state['draft_version']} and measured it: {page_count} page(s).")

    if page_count <= 1:
        state["status"] = "ready"
        state["change_log"].append(
            (f"Rendered and verified: final resume fits on {page_count} page. "
             f"Trimmed {len(state['excluded_ids'])} low-relevance item(s) to make it fit.")
            if state["excluded_ids"] else
            "Rendered and verified: final resume fits on one page with no trimming needed."
        )
        _log(state, "Verification passed: resume fits one page. Marking ready for delivery.")
        return state

    candidate_id, reason = _next_trim_candidate(state)
    if candidate_id is None:
        state["status"] = "ready"
        state["change_log"].append(
            f"Resume still spans {page_count} pages after trimming every non-essential achievement, "
            "certification, and low-relevance bullet. Further reduction would mean cutting education, "
            "skills, or the summary, which this agent will not do automatically -- delivering as-is "
            "for a human to make that call."
        )
        _log(state, "No more safely-trimmable content; delivering best-effort multi-page resume.")
        return state

    state["excluded_ids"].append(candidate_id)
    state["change_log"].append(
        f"Page-fit check: draft measured at {page_count} pages. Removed {reason} to reclaim space, then re-rendering to verify."
    )
    _log(state, f"Adapting: removed {candidate_id} ({reason}); re-rendering to re-check page count.")
    return state


def _next_trim_candidate(state: dict) -> tuple:
    """
    Pick the single next item to cut when the render is over one page.
    Order of preference (least damaging first):
      1. An achievement with zero JD relevance.
      2. A certification with zero JD relevance.
      3. An achievement/certification WITH some relevance (only once the
         zero-relevance pool is exhausted).
      4. The single weakest currently-included bullet (lowest requirement
         coverage, no metric, project over experience) -- last resort, and
         only if more than one bullet remains, so the resume never ends up
         with zero experience/projects shown.
    Education, skills, and summary are never candidates.
    Returns (evidence_id_or_None, human_readable_reason).
    """
    excluded = set(state["excluded_ids"])
    evidence_by_id = {e["id"]: e for e in state["candidate"]["evidence"]}

    flat_pool = [
        e for e in state["candidate"]["evidence"]
        if e["type"] in TRIMMABLE_FLAT_TYPES and e["id"] not in excluded
    ]
    if flat_pool:
        flat_pool.sort(key=lambda e: _req_count_for_evidence(state, e["id"]))
        worst = flat_pool[0]
        req_count = _req_count_for_evidence(state, worst["id"])
        label = "achievement" if worst["type"] == "achievements" else "certification"
        if req_count == 0:
            reason = f"a {label} unrelated to any requirement in this job description (\"{worst['title']}\")"
        else:
            reason = f"the least JD-relevant remaining {label} (\"{worst['title']}\")"
        return worst["id"], reason

    latest_bullets = state["draft_history"][-1]["bullets"] if state["draft_history"] else []
    current_bullet_ids = [b["source_id"] for b in latest_bullets if b.get("source_id") and b["source_id"] not in excluded]
    if len(current_bullet_ids) > 1:
        ranked = _rank_evidence_by_relevance(state)
        # ranked is best-first; walk from the worst end, skipping ids
        # already excluded, to find the weakest bullet still showing.
        for eid in reversed(ranked):
            if eid in current_bullet_ids:
                ev = evidence_by_id[eid]
                reason = f"the lowest-relevance remaining bullet (\"{ev['title']}\")"
                return eid, reason

    return None, ""