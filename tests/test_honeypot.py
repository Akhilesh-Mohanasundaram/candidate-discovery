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
