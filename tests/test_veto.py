import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from veto_checker import check_veto

@pytest.fixture
def jd_features():
    return {"hard_requirements": [], "soft_preferences": [], "experience_range": {"min": 5, "max": 9}, "target_locations": []}

def test_veto_consulting(jd_features):
    cand = {
        "profile": {"years_of_experience": 8, "current_title": "Consultant"},
        "career_history": [{"title": "Consultant", "company": "TCS", "duration_months": 96, "start_date": "2018-01-01"}]
    }
    veto, reason = check_veto(cand, jd_features)
    assert veto
    assert reason == "consulting_only"

def test_veto_architect(jd_features):
    cand = {
        "profile": {"years_of_experience": 10, "current_title": "Chief Technology Officer"},
        "career_history": [{"title": "Chief Technology Officer", "company": "Startup", "duration_months": 24, "start_date": "2024-01-01"}]
    }
    veto, reason = check_veto(cand, jd_features)
    assert veto
    assert reason == "architect_no_code"

def test_veto_pure_research(jd_features):
    cand = {
        "profile": {"years_of_experience": 8, "current_title": "Postdoc Researcher"},
        "career_history": [
            {"title": "Postdoc Researcher", "company": "University", "duration_months": 48, "start_date": "2020-01-01"},
            {"title": "PhD Student", "company": "University", "duration_months": 48, "start_date": "2016-01-01"}
        ]
    }
    veto, reason = check_veto(cand, jd_features)
    assert veto
    assert reason == "pure_research"

def test_veto_title_chaser(jd_features):
    cand = {
        "profile": {"years_of_experience": 4, "current_title": "VP of AI"},
        "career_history": [
            {"title": "VP of AI", "company": "A", "duration_months": 6, "start_date": "2023-01-01"},
            {"title": "Director of AI", "company": "B", "duration_months": 6, "start_date": "2022-06-01"},
            {"title": "Lead AI", "company": "C", "duration_months": 6, "start_date": "2022-01-01"},
            {"title": "Senior AI", "company": "D", "duration_months": 6, "start_date": "2021-06-01"}
        ]
    }
    veto, reason = check_veto(cand, jd_features)
    assert veto
    assert reason == "title_chaser"

def test_veto_clean(jd_features):
    cand = {
        "profile": {"years_of_experience": 5, "current_title": "AI Engineer"},
        "career_history": [{"title": "AI Engineer", "company": "Tech Corp", "duration_months": 60, "start_date": "2020-01-01"}]
    }
    veto, reason = check_veto(cand, jd_features)
    assert not veto
    assert reason is None
