def calculate_career_quality_score(career_history, expected_yoe_min, expected_yoe_max):
    """
    Evaluates title progression, stability, and relevance.
    """
    if not career_history:
        return 0.1
        
    score = 0.0
    total_months = 0
    tier_weights = {
        "1-10": 0.5, "11-50": 0.6, "51-200": 0.7, "201-500": 0.8, 
        "501-1000": 0.9, "1001-5000": 1.0, "5001-10000": 1.0, "10001+": 1.0
    }
    
    for job in career_history:
        months = job.get("duration_months", 0)
        total_months += months
        
        # Stability bonus (longer tenure is better, up to 48 months per job)
        stability_score = min(1.0, months / 48.0)
        
        # Company size proxy for engineering maturity (simplified)
        size_str = job.get("company_size", "1-10")
        size_score = tier_weights.get(size_str, 0.5)
        
        # Relevance: is it a technical role?
        title = job.get("title", "").lower()
        relevance_score = 1.0 if any(k in title for k in ["engineer", "developer", "scientist", "ml", "ai", "data"]) else 0.4
        
        # Job score weighted by duration
        job_score = stability_score * size_score * relevance_score
        score += job_score * months
        
    if total_months == 0:
        return 0.1
        
    normalized = score / total_months
    
    # Apply YoE penalty if outside bounds (but softly)
    yoe = total_months / 12.0
    if yoe < expected_yoe_min:
        normalized *= 0.8
    elif yoe > expected_yoe_max * 1.5:
        normalized *= 0.9
        
    return min(1.0, normalized)

def calculate_logistics_score(candidate, target_locations):
    """
    Evaluates location overlap and notice period.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    score = 1.0
    
    # Location
    location = profile.get("location", "").lower()
    is_match = any(t.lower() in location for t in target_locations)
    if not is_match and not signals.get("willing_to_relocate", False):
        score *= 0.5 # Penalty for bad location and unwilling to relocate
        
    # Notice Period
    notice = signals.get("notice_period_days", 0)
    if notice <= 30:
        score *= 1.0
    elif notice <= 60:
        score *= 0.8
    elif notice <= 90:
        score *= 0.5
    else:
        score *= 0.2
        
    return score
