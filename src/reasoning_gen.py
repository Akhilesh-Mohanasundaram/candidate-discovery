def generate_reasoning(candidate, jd_features, semantic_score, behavioral_multiplier, final_rank=None, is_veto=False):
    """
    Member A's Reasoning Generation Logic.
    Generates a 1-2 sentence non-templated-looking justification for the candidate's rank.
    
    Args:
        candidate (dict): The candidate profile object.
        jd_features (dict): JD features.
        semantic_score (float): The semantic match score.
        behavioral_multiplier (float): The multiplier calculated from signals.
        final_rank (int): The final assigned rank (optional).
        is_veto (bool): Whether the candidate hit a honeypot/veto rule.
    """
    if is_veto:
        return "Disqualified due to strong negative signals or profile inconsistencies (failed automated integrity checks)."
        
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    
    # Extract top overlapping skills
    skills = candidate.get("skills", [])
    expert_skills = [s["name"] for s in skills if s.get("proficiency") in ["expert", "advanced"]]
    
    # Match against JD hard reqs
    jd_hard = [r["skill"].lower() for r in jd_features.get("hard_requirements", [])]
    matched_skills = [s for s in expert_skills if s.lower() in jd_hard]
    
    if not matched_skills and expert_skills:
        matched_skills = expert_skills[:2] # Fallback to top candidate skills
        
    skill_str = ", ".join(matched_skills[:2]) if matched_skills else "relevant technical skills"
    
    # Base sentence based on career/semantic
    if semantic_score > 0.8:
        base_sentence = f"Strong {title} with {yoe} years of experience and deep expertise in {skill_str}."
    elif semantic_score > 0.5:
        base_sentence = f"Solid {title} with {yoe} years of experience and foundational knowledge of {skill_str}."
    else:
        base_sentence = f"Experienced {title} ({yoe} YoE) but lacks deep alignment with core AI infrastructure requirements."
        
    # Behavioral modifier
    behavioral_sentence = ""
    last_active = signals.get("last_active_date", "")
    response_rate = signals.get("recruiter_response_rate", 1.0)
    notice = signals.get("notice_period_days", 0)
    
    if behavioral_multiplier > 0.9:
        behavioral_sentence = "Highly engaged candidate with strong recent platform activity."
    elif behavioral_multiplier < 0.5 or response_rate < 0.2:
        behavioral_sentence = "Note: Low recent responsiveness or activity may impact availability."
    elif notice > 60:
        behavioral_sentence = f"Note: Stated notice period is {notice} days, which exceeds ideal target."
        
    reasoning = f"{base_sentence} {behavioral_sentence}".strip()
    return reasoning

if __name__ == "__main__":
    # Test reasoning gen
    mock_candidate = {
        "profile": {"years_of_experience": 6, "current_title": "Senior AI Engineer"},
        "skills": [{"name": "Python", "proficiency": "expert"}],
        "redrob_signals": {"recruiter_response_rate": 0.1, "notice_period_days": 90}
    }
    mock_jd = {"hard_requirements": [{"skill": "Python"}]}
    print(generate_reasoning(mock_candidate, mock_jd, 0.9, 0.4))
