import re
from datetime import datetime

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
    Returns True if the candidate is a honeypot (should be vetoed).
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    # Pre-calculate totals
    total_yoe = profile.get("years_of_experience", 0)
    total_career_months = sum([c.get("duration_months", 0) for c in career])
    
    # 1. Time Paradox: sum of career months is vastly different from claimed years_of_experience
    if total_career_months > (total_yoe + 2) * 12 or total_career_months < (total_yoe - 2) * 12:
        return True
        
    # 2. Fake Expert & 3. Keyword Stuffer
    expert_skills_count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert":
            expert_skills_count += 1
            if skill.get("duration_months", 0) == 0:
                # 2. Fake Expert
                return True
                
    if expert_skills_count >= 10:
        # 3. Keyword Stuffer
        return True
        
    # 4. Title Mismatch
    # If they have strong AI skills, their titles should be technical.
    tech_keywords = ["engineer", "developer", "scientist", "analyst", "cto", "architect", "programmer"]
    has_ai_skills = any(s.get("name", "").lower() in ["python", "machine learning", "deep learning", "nlp", "llm", "embeddings"] for s in skills)
    
    if has_ai_skills:
        all_titles = [profile.get("current_title", "").lower()] + [c.get("title", "").lower() for c in career]
        is_technical = any(any(kw in title for kw in tech_keywords) for title in all_titles)
        if not is_technical and len(all_titles) > 0:
            # Strong AI skills but no technical title ever
            return True
            
    # 5. Timeline Overlap
    # Two full-time jobs overlapping by more than 3 months
    # (Since schema doesn't specify full-time, we just use raw overlap > 3 months)
    for i in range(len(career)):
        start_a = parse_date(career[i].get("start_date"))
        end_a = parse_date(career[i].get("end_date")) or datetime.now()
        for j in range(i + 1, len(career)):
            start_b = parse_date(career[j].get("start_date"))
            end_b = parse_date(career[j].get("end_date")) or datetime.now()
            
            if start_a and end_a and start_b and end_b:
                # Find overlap
                overlap_start = max(start_a, start_b)
                overlap_end = min(end_a, end_b)
                if overlap_start < overlap_end:
                    overlap_months = (overlap_end - overlap_start).days / 30.0
                    if overlap_months > 3:
                        return True
                        
    # 6. Ghost Skills
    # Skipping the strict version unless we want aggressive filtering.
    # The spec says: "(Optional/Soft trap)... For strict honeypot, we only flag if >50% of advanced/expert skills are ghosts."
    advanced_skills = [s.get("name", "").lower() for s in skills if s.get("proficiency") in ["advanced", "expert"]]
    if len(advanced_skills) > 0:
        career_desc_text = " ".join([c.get("description", "").lower() for c in career])
        ghost_count = sum(1 for s in advanced_skills if s not in career_desc_text)
        if ghost_count / len(advanced_skills) > 0.5:
            return True
            
    # 7. Behavioral Ghost
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (datetime.now() - last_active).days
        if days_inactive > 180 and signals.get("recruiter_response_rate", 1.0) < 0.05:
            return True
            
    return False

if __name__ == "__main__":
    # Test script
    mock_bad = {
        "profile": {"years_of_experience": 10},
        "career_history": [{"duration_months": 240}] # Time paradox
    }
    print(f"Bad candidate test: {check_honeypot(mock_bad)}")
