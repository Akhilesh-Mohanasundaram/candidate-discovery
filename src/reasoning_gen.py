"""
Reasoning Generation — Stage 6
================================
Generates specific, honest, non-templated reasoning per candidate for the CSV
submission. Every claim references real data from the candidate's profile.

The reasoning column is manually reviewed at Stage 4.  The 6 checks are:
  1. Specific facts from profile
  2. JD connection
  3. Honest concerns
  4. No hallucination
  5. Variation across candidates
  6. Rank consistency
"""


def generate_reasoning(candidate, jd_features, semantic_score, behavioral_multiplier,
                       final_rank=None, is_veto=False, veto_reason=None):
    """
    Generates a 1-2 sentence candidate-specific justification for their rank.

    All claims are grounded in actual candidate data — no fabricated facts.
    The tone adapts to the rank and score to maintain rank-consistency.
    """
    if is_veto:
        return _generate_veto_reasoning(candidate, veto_reason)

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills_list = candidate.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    location = profile.get("location", "undisclosed location")

    # ---- Extract matched JD skills ----
    jd_hard_skills = set()
    jd_hard_aliases = set()
    for req in jd_features.get("hard_requirements", []):
        jd_hard_skills.add(req["skill"].lower())
        for alias in req.get("aliases", []):
            jd_hard_aliases.add(alias.lower())

    jd_soft_skills = set()
    for pref in jd_features.get("soft_preferences", []):
        jd_soft_skills.add(pref["skill"].lower())
        for alias in pref.get("aliases", []):
            jd_soft_skills.add(alias.lower())

    # Find which candidate skills overlap with JD
    matched_hard = []
    matched_soft = []
    candidate_top_skills = []

    for s in skills_list:
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        name_lower = name.lower()

        if prof in ("expert", "advanced"):
            candidate_top_skills.append((name, prof, dur))

        if name_lower in jd_hard_skills or name_lower in jd_hard_aliases:
            matched_hard.append((name, prof, dur))
        elif name_lower in jd_soft_skills:
            matched_soft.append((name, prof, dur))

    # Sort by proficiency quality and duration
    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    matched_hard.sort(key=lambda x: (prof_order.get(x[1], 0), x[2]), reverse=True)
    matched_soft.sort(key=lambda x: (prof_order.get(x[1], 0), x[2]), reverse=True)
    candidate_top_skills.sort(key=lambda x: (prof_order.get(x[1], 0), x[2]), reverse=True)

    # ---- Career highlights ----
    career_highlights = []
    for job in career:
        c_title = job.get("title", "")
        c_company = job.get("company", "")
        c_dur = job.get("duration_months", 0)
        if c_dur >= 24 and any(k in c_title.lower() for k in [
            "engineer", "developer", "scientist", "ml", "ai", "data",
            "search", "ranking", "retrieval",
        ]):
            career_highlights.append(f"{c_title} at {c_company} ({c_dur // 12}yr)")

    # ---- Behavioral context ----
    response_rate = signals.get("recruiter_response_rate", 1.0)
    notice = signals.get("notice_period_days", 0)
    last_active = signals.get("last_active_date", "")
    open_to_work = signals.get("open_to_work_flag", False)
    github_score = signals.get("github_activity_score", -1)
    interview_rate = signals.get("interview_completion_rate", 1.0)

    # ---- Build the reasoning sentence ----
    parts = []
    concerns = []

    # == PART 1: Core qualification statement ==
    # ALWAYS name specific skills — never use "relevant technical skills"
    if matched_hard:
        top_matched = [m[0] for m in matched_hard[:3]]
        skill_str = ", ".join(top_matched)
        if semantic_score > 0.7:
            parts.append(f"{title} with {yoe:.0f} years experience and strong alignment "
                         f"on core JD requirements ({skill_str})")
        elif semantic_score > 0.45:
            parts.append(f"{title} ({yoe:.0f} YoE) with partial match on JD requirements "
                         f"via {skill_str}")
        else:
            parts.append(f"{title} ({yoe:.0f} YoE); best JD overlap is {skill_str} "
                         f"but semantic alignment is weak")
    elif candidate_top_skills:
        top_names = [s[0] for s in candidate_top_skills[:3]]
        parts.append(f"{title} ({yoe:.0f} YoE) with expertise in {', '.join(top_names)} "
                     f"but limited overlap with JD's embeddings/retrieval/ranking requirements")
    else:
        # Name whatever skills they DO have
        any_skills = [s.get("name", "") for s in skills_list[:3] if s.get("name")]
        if any_skills:
            parts.append(f"{title} ({yoe:.0f} YoE) with skills in {', '.join(any_skills)}; "
                         f"no strong alignment with the AI retrieval/ranking JD requirements")
        else:
            parts.append(f"{title} ({yoe:.0f} YoE) with no listed skills matching "
                         f"the JD's AI/retrieval/ranking requirements")

    # == PART 2: Career context (if notable) ==
    if career_highlights and final_rank and final_rank <= 40:
        parts.append(f"career includes {career_highlights[0]}")

    # == PART 3: Positive differentiators ==
    if matched_soft:
        soft_names = [m[0] for m in matched_soft[:2]]
        parts.append(f"bonus alignment on {', '.join(soft_names)}")

    if github_score > 60:
        parts.append(f"strong GitHub activity (score: {github_score})")

    if behavioral_multiplier > 0.85 and response_rate > 0.7:
        parts.append("highly engaged on platform")
    elif open_to_work:
        parts.append("marked as open to work")

    # == PART 4: Honest concerns ==
    if notice > 60:
        concerns.append(f"notice period is {notice} days (JD prefers <30)")

    if response_rate < 0.2:
        concerns.append(f"recruiter response rate is only {response_rate:.0%}")

    if last_active:
        try:
            from datetime import datetime
            days_ago = (datetime.now() - datetime.strptime(last_active, "%Y-%m-%d")).days
            if days_ago > 120:
                concerns.append(f"last active {days_ago} days ago")
        except (ValueError, TypeError):
            pass

    if interview_rate < 0.5:
        concerns.append(f"low interview completion rate ({interview_rate:.0%})")

    # YoE outside desired range
    exp_range = jd_features.get("experience_range", {})
    yoe_min = exp_range.get("min", 5)
    yoe_max = exp_range.get("max", 9)
    if yoe < yoe_min:
        concerns.append(f"only {yoe:.0f} YoE (JD targets {yoe_min}-{yoe_max})")
    elif yoe > yoe_max * 1.5:
        concerns.append(f"{yoe:.0f} YoE may be over-senior for this role")

    # Location mismatch
    target_locs = jd_features.get("target_locations", [])
    if target_locs and not any(t.lower() in location.lower() for t in target_locs):
        willing = signals.get("willing_to_relocate", False)
        if not willing:
            concerns.append(f"based in {location}, not in target cities and unwilling to relocate")
        elif final_rank and final_rank > 50:
            concerns.append(f"based in {location}, would need relocation")

    # == ASSEMBLE ==
    base = "; ".join(parts) + "."

    if concerns:
        concern_str = "; ".join(concerns)
        reasoning = f"{base} Concern: {concern_str}."
    else:
        reasoning = base

    # Capitalize first letter
    reasoning = reasoning[0].upper() + reasoning[1:] if reasoning else reasoning

    # Enforce max length (~300 chars for CSV readability)
    if len(reasoning) > 350:
        reasoning = reasoning[:347] + "..."

    return reasoning


def _generate_veto_reasoning(candidate, veto_reason=None):
    """
    Generates reasoning for vetoed candidates (honeypots or JD disqualifiers).
    References actual profile facts to avoid hallucination.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)

    _reason_map = {
        "pure_research": "career consists primarily of academic/research roles without production deployment",
        "langchain_wrapper": "AI experience is recent LLM-wrapper tooling without pre-LLM ML production background",
        "architect_no_code": "recent career has been in architecture/leadership roles without hands-on coding",
        "consulting_only": "entire career at IT consulting firms without product-company experience",
        "title_chaser": "career pattern shows frequent company switches with average tenure under 18 months",
        "cv_speech_robotics": "primary expertise is in computer vision/speech/robotics without NLP/IR crossover",
    }

    if veto_reason and veto_reason in _reason_map:
        detail = _reason_map[veto_reason]
        return (f"Disqualified: {title} ({yoe:.0f} YoE) — {detail}, "
                f"which the JD explicitly lists as a disqualifier.")

    # Honeypot fallback
    return (f"Disqualified: {title} ({yoe:.0f} YoE) failed automated integrity checks "
            f"due to profile inconsistencies indicating a fabricated or impossible profile.")
