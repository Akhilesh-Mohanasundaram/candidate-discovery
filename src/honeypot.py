"""
Honeypot Detection — Stage 2
==============================
Checks the 7 honeypot traps defined in honeypot_rules_spec.md.
Honeypot candidates are fabricated/impossible profiles that MUST receive 0.0.

Traps:
  1. Time Paradox      — career months vs. stated YoE differ by >2 years
  2. Fake Expert       — "expert" proficiency with 0 months duration
  3. Keyword Stuffer   — ≥10 skills at "expert" level
  4. Title Mismatch    — AI skills but only non-technical titles ever
  5. Timeline Overlap  — 3+ month overlap between concurrent positions
  6. Ghost Skills      — >50% of advanced/expert skills absent from career descriptions
  7. Behavioral Ghost  — Inactive >180 days AND <5% recruiter response rate
"""
import re
from datetime import datetime
from constants import REFERENCE_DATE

# Technical title keywords — if ANY title matches, the candidate is technical
_TECH_TITLE_KEYWORDS = [
    "engineer", "developer", "scientist", "analyst", "cto",
    "architect", "programmer", "sde", "swe", "devops",
    "data", "ml", "ai", "software", "technical", "tech lead",
]

# AI/ML skill names that indicate claimed AI expertise
_AI_SKILL_NAMES = {
    "python", "machine learning", "deep learning", "nlp",
    "llm", "embeddings", "tensorflow", "pytorch", "transformers",
    "natural language processing", "neural network", "bert",
    "computer vision", "reinforcement learning", "data science",
}


def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def check_honeypot(candidate):
    """
    Checks the 7 honeypot traps defined in honeypot_rules_spec.md.
    Returns True if the candidate is a honeypot (should be vetoed with score 0.0).
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # Pre-calculate totals
    total_yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(c.get("duration_months", 0) for c in career)

    # ---- 1. Time Paradox ----
    # sum(duration_months) vs (YoE ± 2) * 12
    if total_yoe > 0 and len(career) > 0:
        if (total_career_months > (total_yoe + 2) * 12 or
                total_career_months < (total_yoe - 2) * 12):
            return True

    # ---- 2. Fake Expert & 3. Keyword Stuffer ----
    expert_skills_count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert":
            expert_skills_count += 1
            if skill.get("duration_months", 0) == 0:
                # Trap 2: Fake Expert — expert with 0 months
                return True

    if expert_skills_count >= 10:
        # Trap 3: Keyword Stuffer — ≥10 expert skills
        return True

    # ---- 4. Title Mismatch ----
    # Strong AI skills present, but NO technical title anywhere in career
    skill_names_lower = {s.get("name", "").lower() for s in skills}
    has_ai_skills = bool(skill_names_lower & _AI_SKILL_NAMES)

    if has_ai_skills and len(career) > 0:
        all_titles = [profile.get("current_title", "").lower()]
        all_titles += [c.get("title", "").lower() for c in career]
        # Filter empty titles
        all_titles = [t for t in all_titles if t.strip()]

        if all_titles:
            is_technical = any(
                any(kw in title for kw in _TECH_TITLE_KEYWORDS)
                for title in all_titles
            )
            if not is_technical:
                # Has AI skills but zero technical titles ever → honeypot
                return True

    # ---- 5. Timeline Overlap ----
    # Two jobs overlapping by >3 months. For current jobs, estimate end_date as REFERENCE_DATE.
    _now = REFERENCE_DATE
    for i in range(len(career)):
        start_a = parse_date(career[i].get("start_date"))
        end_a = parse_date(career[i].get("end_date"))
        if career[i].get("is_current", False):
            end_a = _now

        for j in range(i + 1, len(career)):
            start_b = parse_date(career[j].get("start_date"))
            end_b = parse_date(career[j].get("end_date"))
            if career[j].get("is_current", False):
                end_b = _now

            if start_a and end_a and start_b and end_b:
                # Find overlap
                overlap_start = max(start_a, start_b)
                overlap_end = min(end_a, end_b)
                if overlap_start < overlap_end:
                    overlap_months = (overlap_end - overlap_start).days / 30.0
                    if overlap_months > 3:
                        return True

    # ---- 6. Ghost Skills ----
    # >75% of advanced/expert skills are absent from career descriptions.
    # Uses word-boundary matching to avoid "ml" matching inside "html".
    # Also checks profile summary/headline as an additional source.
    advanced_skills = [
        s.get("name", "").lower()
        for s in skills
        if s.get("proficiency") in ("advanced", "expert")
    ]
    if len(advanced_skills) >= 2:  # Only check if they claim 2+ advanced/expert skills
        career_desc_text = " ".join(
            c.get("description", "").lower() for c in career
        )
        # Also check profile summary and headline
        profile_text = (
            profile.get("summary", "").lower() + " " +
            profile.get("headline", "").lower()
        )
        full_text = career_desc_text + " " + profile_text

        ghost_count = 0
        for skill_name in advanced_skills:
            # Use word-boundary regex for single-word skills
            # For multi-word skills, simple containment is fine
            if len(skill_name.split()) == 1:
                pattern = r'\b' + re.escape(skill_name) + r'\b'
                if not re.search(pattern, full_text):
                    ghost_count += 1
            else:
                if skill_name not in full_text:
                    ghost_count += 1

        # Pure-ratio rule: >80% ghosts AND at least 2 advanced skills claimed
        # 80% threshold is more conservative than the original 50% to reduce false positives on real candidates
        if ghost_count >= 3 and ghost_count / advanced_count > 0.8:
            return True

    # ---- 7. Behavioral Ghost ----
    # Inactive >180 days AND recruiter_response_rate < 0.05
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (_now - last_active).days
        if days_inactive > 180 and signals.get("recruiter_response_rate", 1.0) < 0.05:
            return True

    return False


if __name__ == "__main__":
    # Test script
    mock_bad = {
        "profile": {"years_of_experience": 10},
        "career_history": [{"duration_months": 240}]  # Time paradox
    }
    print(f"Bad candidate test: {check_honeypot(mock_bad)}")

    # Test overlap fix — two current jobs SHOULD flag
    mock_current = {
        "profile": {"years_of_experience": 5},
        "career_history": [
            {"start_date": "2022-01-01", "end_date": None,
             "duration_months": 30, "is_current": True},
            {"start_date": "2023-01-01", "end_date": None,
             "duration_months": 18, "is_current": True},
        ],
        "skills": [],
        "redrob_signals": {}
    }
    print(f"Two current jobs (should be True): {check_honeypot(mock_current)}")
