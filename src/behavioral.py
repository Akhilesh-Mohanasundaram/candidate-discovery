import math
from datetime import datetime

def calculate_behavioral_multiplier(signals):
    """
    Calculates the 23-signal behavioral multiplier [0.3 - 1.0]
    """
    multiplier = 1.0
    
    # 1. Recruiter Response Rate
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate < 0.2:
        multiplier *= 0.6
    elif response_rate > 0.8:
        multiplier *= 1.1
        
    # 2. Recency Decay (e^(-days/90))
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_date = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = max(0, (datetime.now() - last_date).days)
            recency_multiplier = math.exp(-days_inactive / 90.0)
            # Bound the recency penalty so it doesn't go below 0.5 on its own
            recency_multiplier = max(0.5, recency_multiplier)
            multiplier *= recency_multiplier
        except ValueError:
            pass
            
    # 3. Profile Completeness
    completeness = signals.get("profile_completeness_score", 50)
    if completeness < 50:
        multiplier *= 0.8
    elif completeness > 90:
        multiplier *= 1.05
        
    # 4. Interview Completion Rate
    interview_rate = signals.get("interview_completion_rate", 1.0)
    if interview_rate < 0.5:
        multiplier *= 0.7
        
    # Clamp final output
    return max(0.3, min(1.0, multiplier))
