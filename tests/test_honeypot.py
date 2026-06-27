import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from honeypot import check_honeypot

def test_honeypot_time_paradox():
    cand = {"profile": {"years_of_experience": 10}, "career_history": [{"duration_months": 12, "title": "Dev"}]}
    assert check_honeypot(cand)

def test_honeypot_timeline_overlap():
    cand = {
        "profile": {"years_of_experience": 5},
        "career_history": [
            {"duration_months": 24, "title": "Role 1", "is_current": True, "start_date": "2022-01-01"},
            {"duration_months": 24, "title": "Role 2", "is_current": True, "start_date": "2022-01-01"}
        ]
    }
    assert check_honeypot(cand)

def test_honeypot_fake_expert():
    cand = {
        "profile": {"years_of_experience": 5},
        "career_history": [{"duration_months": 60, "title": "Dev"}],
        "skills": [{"name": "PyTorch", "proficiency": "expert", "duration_months": 0}]
    }
    assert check_honeypot(cand)

def test_honeypot_keyword_stuffer():
    cand = {
        "profile": {"years_of_experience": 5},
        "career_history": [{"duration_months": 60, "title": "Dev", "start_date": "2020-01-01"}],
        "skills": [{"name": f"Skill {i}", "proficiency": "expert", "duration_months": 24} for i in range(11)]
    }
    assert check_honeypot(cand)

def test_honeypot_ghost_skills():
    cand = {
        "profile": {"years_of_experience": 5},
        "career_history": [{"duration_months": 60, "title": "Dev", "description": "Did some coding.", "start_date": "2020-01-01"}],
        "skills": [
            {"name": "SkillA", "proficiency": "expert", "duration_months": 24},
            {"name": "SkillB", "proficiency": "expert", "duration_months": 24},
            {"name": "SkillC", "proficiency": "expert", "duration_months": 24},
            {"name": "SkillD", "proficiency": "expert", "duration_months": 24}
        ]
    }
    assert check_honeypot(cand)

def test_honeypot_clean():
    cand = {
        "profile": {"years_of_experience": 5, "current_title": "AI Engineer"},
        "career_history": [{"duration_months": 60, "title": "AI Engineer", "description": "Used Python and PyTorch.", "start_date": "2020-01-01"}],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 60},
            {"name": "PyTorch", "proficiency": "advanced", "duration_months": 48}
        ],
        "redrob_signals": {"last_active_date": "2026-06-01", "recruiter_response_rate": 0.8}
    }
    assert not check_honeypot(cand)
