"""
Behavioral Signal Multiplier — Stage 5
=======================================
Implements the full 23-signal availability formula as a multiplier [0.3 – 1.0]
on raw ranking scores. Treats availability as a gate, not a bonus.

All 23 signals from redrob_signals_doc.txt are consumed:
  Group 1 — Activity & Recency (25%): signals 2, 3, 4, 6
  Group 2 — Responsiveness (22%): signals 7, 8
  Group 3 — Profile Quality (15%): signals 1, 10, 11, 21, 22, 23
  Group 4 — Interview & Hiring Readiness (18%): signals 12, 13, 14, 15, 19, 20
  Group 5 — Market Demand & Technical Credibility (20%): signals 5, 9, 16, 17, 18
"""
import math
from datetime import datetime


# JD specifies hybrid work mode in Pune/Noida
_PREFERRED_WORK_MODES = {"hybrid", "flexible"}

# Reasonable salary band for a Senior AI Engineer (5-9 YoE) in India — INR LPA
# Source: AmbitionBox / Glassdoor (India 2026 market data)
_SALARY_FLOOR_LPA = 20
_SALARY_CEILING_LPA = 75


def calculate_behavioral_multiplier(signals):
    """
    Calculates the 23-signal behavioral multiplier [0.3 - 1.0].

    The multiplier penalizes inactive, unresponsive, or unavailable candidates
    while rewarding high-engagement profiles. It acts as a gate — even a
    technically perfect candidate who is behaviourally unavailable gets
    down-weighted significantly.

    Args:
        signals (dict): The candidate's redrob_signals object.

    Returns:
        float: A multiplier in the range [0.3, 1.0].
    """
    if not signals:
        return 0.5  # No signals = uncertain, moderate penalty

    # Accumulate a composite score from 0..1 across all signal groups,
    # then map to the [0.3, 1.0] range at the end.
    score_components = []
    _now = datetime.now()

    # =========================================================================
    # GROUP 1: Activity & Recency (Signals 2, 3, 4, 6) — Weight: 25%
    # =========================================================================

    # Signal 2: signup_date — Account maturity (older = more established)
    maturity_score = 0.5  # default: neutral
    signup_date = signals.get("signup_date")
    if signup_date:
        try:
            signup_dt = datetime.strptime(signup_date, "%Y-%m-%d")
            account_age_days = max(0, (_now - signup_dt).days)
            # 1+ year on platform → full credit; <30 days → low
            maturity_score = min(1.0, account_age_days / 365.0)
        except (ValueError, TypeError):
            pass

    # Signal 3: last_active_date — Recency decay (e^(-days/90))
    recency_score = 0.5  # default: neutral
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_date = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = max(0, (_now - last_date).days)
            recency_score = math.exp(-days_inactive / 90.0)
        except (ValueError, TypeError):
            pass

    # Signal 4: open_to_work_flag
    otw_score = 0.7 if signals.get("open_to_work_flag", False) else 0.4

    # Signal 6: applications_submitted_30d — active job seekers
    apps_30d = signals.get("applications_submitted_30d", 0)
    app_score = min(1.0, apps_30d / 10.0) if apps_30d > 0 else 0.2

    activity_score = (
        0.15 * maturity_score +
        0.40 * recency_score +
        0.25 * otw_score +
        0.20 * app_score
    )
    score_components.append(("activity", activity_score, 0.25))

    # =========================================================================
    # GROUP 2: Responsiveness (Signals 7, 8) — Weight: 22%
    # =========================================================================

    # Signal 7: recruiter_response_rate
    response_rate = signals.get("recruiter_response_rate", 0.5)
    resp_score = min(1.0, response_rate / 0.8)  # 80%+ response rate = perfect

    # Signal 8: avg_response_time_hours
    avg_resp_time = signals.get("avg_response_time_hours", 48)
    # Under 12 hours = excellent, 12-48 = okay, 48+ = poor
    if avg_resp_time <= 12:
        time_score = 1.0
    elif avg_resp_time <= 48:
        time_score = 0.7
    elif avg_resp_time <= 96:
        time_score = 0.4
    else:
        time_score = 0.2

    responsiveness_score = 0.7 * resp_score + 0.3 * time_score
    score_components.append(("responsiveness", responsiveness_score, 0.22))

    # =========================================================================
    # GROUP 3: Profile Quality (Signals 1, 10, 11, 21, 22, 23) — Weight: 13%
    # =========================================================================

    # Signal 1: profile_completeness_score
    completeness = signals.get("profile_completeness_score", 50)
    completeness_score = min(1.0, completeness / 90.0)

    # Signal 10: connection_count
    connections = signals.get("connection_count", 0)
    connection_score = min(1.0, connections / 200.0)

    # Signal 11: endorsements_received
    endorsements = signals.get("endorsements_received", 0)
    endorsement_score = min(1.0, endorsements / 50.0)

    # Signals 21, 22, 23: verification flags
    verified_count = sum([
        1 if signals.get("verified_email", False) else 0,
        1 if signals.get("verified_phone", False) else 0,
        1 if signals.get("linkedin_connected", False) else 0,
    ])
    verification_score = verified_count / 3.0

    profile_quality = (
        0.35 * completeness_score +
        0.20 * connection_score +
        0.20 * endorsement_score +
        0.25 * verification_score
    )
    score_components.append(("profile_quality", profile_quality, 0.13))

    # =========================================================================
    # GROUP 4: Interview & Hiring Readiness (Signals 12, 13, 14, 15, 19, 20)
    #           — Weight: 20%
    # =========================================================================

    # Signal 12: notice_period_days
    notice = signals.get("notice_period_days", 60)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.8
    elif notice <= 90:
        notice_score = 0.5
    else:
        notice_score = 0.2

    # Signal 13: expected_salary_range_inr_lpa — salary reasonableness
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    if salary_range and isinstance(salary_range, dict):
        sal_min = salary_range.get("min", 0)
        sal_max = salary_range.get("max", 0)
        if sal_max > 0:
            # Penalise if expectations are wildly outside the band
            mid = (sal_min + sal_max) / 2.0
            if _SALARY_FLOOR_LPA <= mid <= _SALARY_CEILING_LPA:
                salary_score = 1.0
            elif mid < _SALARY_FLOOR_LPA:
                salary_score = 0.7  # Suspiciously low, but not penalised hard
            elif mid <= _SALARY_CEILING_LPA * 1.5:
                salary_score = 0.5  # Above band but maybe negotiable
            else:
                salary_score = 0.3  # Far above band
        else:
            salary_score = 0.5  # No data, neutral
    else:
        salary_score = 0.5  # No data, neutral

    # Signal 14: preferred_work_mode — JD is hybrid/flexible
    work_mode = signals.get("preferred_work_mode", "").lower().strip()
    if work_mode in _PREFERRED_WORK_MODES:
        work_mode_score = 1.0
    elif work_mode == "onsite":
        work_mode_score = 0.8  # Onsite is compatible with hybrid
    elif work_mode == "remote":
        work_mode_score = 0.7  # Remote-only may not fit hybrid expectation (soft penalty)
    else:
        work_mode_score = 0.6  # Unknown / not specified

    # Signal 15: willing_to_relocate
    willing_to_relocate = signals.get("willing_to_relocate", False)
    relocation_score = 0.8 if willing_to_relocate else 0.4

    # Signal 19: interview_completion_rate
    interview_rate = signals.get("interview_completion_rate", 0.8)
    interview_score = min(1.0, interview_rate)

    # Signal 20: offer_acceptance_rate (-1 = no prior offers)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate < 0:
        offer_score = 0.5  # No data, neutral
    else:
        offer_score = min(1.0, offer_rate)

    hiring_readiness = (
        0.25 * notice_score +
        0.10 * salary_score +
        0.10 * work_mode_score +
        0.10 * relocation_score +
        0.25 * interview_score +
        0.20 * offer_score
    )
    score_components.append(("hiring_readiness", hiring_readiness, 0.20))

    # =========================================================================
    # GROUP 5: Market Demand & Technical Credibility (Signals 5, 9, 16, 17, 18)
    #           — Weight: 20%
    # =========================================================================

    # Signal 5: profile_views_received_30d
    views = signals.get("profile_views_received_30d", 0)
    views_score = min(1.0, views / 20.0)

    # Signal 9: skill_assessment_scores — average across assessments
    assessments = signals.get("skill_assessment_scores", {})
    if assessments and isinstance(assessments, dict):
        avg_assessment = sum(assessments.values()) / len(assessments)
        assessment_score = min(1.0, avg_assessment / 80.0)
    else:
        assessment_score = 0.5  # No assessments, neutral

    # Signal 16: github_activity_score (-1 = no GitHub)
    github = signals.get("github_activity_score", -1)
    if github < 0:
        github_score = 0.3  # No GitHub, slight penalty
    else:
        github_score = min(1.0, github / 70.0)

    # Signal 17: search_appearance_30d
    search_appearances = signals.get("search_appearance_30d", 0)
    search_score = min(1.0, search_appearances / 30.0)

    # Signal 18: saved_by_recruiters_30d
    saved = signals.get("saved_by_recruiters_30d", 0)
    saved_score = min(1.0, saved / 10.0)

    market_demand = (
        0.15 * views_score +
        0.30 * assessment_score +
        0.20 * github_score +
        0.15 * search_score +
        0.20 * saved_score
    )
    score_components.append(("market_demand", market_demand, 0.20))

    # =========================================================================
    # FINAL: Weighted combination → map to [0.3, 1.0]
    # =========================================================================

    raw_composite = sum(score * weight for _, score, weight in score_components)

    # Map [0, 1] → [0.3, 1.0]
    multiplier = 0.3 + 0.7 * raw_composite

    # Hard clamp
    return max(0.3, min(1.0, multiplier))
