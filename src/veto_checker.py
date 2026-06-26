"""
VETO Disqualifier Checker — JD-Specific Rejection Rules
=========================================================
Implements the 6 veto disqualifier types defined in the JD:
  1. pure_research     — Only academic/research roles, no production deployment
  2. langchain_wrapper  — LangChain-only <12mo experience without pre-LLM ML
  3. architect_no_code  — Hasn't coded in 18+ months, only architect/VP titles
  4. consulting_only    — Entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini
  5. title_chaser       — Average tenure <18 months across 3+ job hops
  6. cv_speech_robotics — Primary CV/speech/robotics expertise without NLP/IR overlap

These are distinct from honeypot traps (which catch fabricated profiles).
VETO candidates receive a hard 0.0 score.
"""
import re
from datetime import datetime
from honeypot import parse_date
from constants import VetoType

# Consulting firms explicitly called out in the JD
_CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "tech mahindra",
    "hcl", "hcl technologies", "cts",
}

# Technical title keywords that indicate hands-on engineering
_TECH_TITLE_KEYWORDS = {
    "engineer", "developer", "programmer", "scientist", "analyst",
    "cto", "sde", "swe", "ml", "ai", "data", "devops", "backend",
    "frontend", "fullstack", "full stack", "platform", "infrastructure",
}

# Architecture/leadership titles that indicate no-code roles
_ARCH_TITLE_KEYWORDS = {
    "architect", "vp of engineering", "vice president",
    "director of engineering", "director of technology",
    "chief technology", "head of engineering", "engineering manager",
}

# Research-only title keywords
_RESEARCH_TITLE_KEYWORDS = {
    "researcher", "research scientist", "research engineer",
    "research fellow", "postdoc", "postdoctoral", "phd candidate",
    "research assistant", "academic", "professor", "lecturer",
}

# NLP/IR skills that indicate relevant crossover
_NLP_IR_SKILLS = {
    "nlp", "natural language processing", "information retrieval",
    "text mining", "search", "ranking", "retrieval", "rag",
    "embeddings", "transformers", "bert", "gpt", "llm",
    "language model", "text classification", "ner",
    "named entity recognition", "sentiment analysis",
    "question answering", "semantic search", "vector database",
    "elasticsearch", "solr", "lucene",
}

# CV/Speech/Robotics primary skills — only unambiguous domain-specific ones
_CV_SPEECH_ROBOTICS_SKILLS = {
    "computer vision", "opencv", "image processing", "object detection",
    "image segmentation", "image classification",
    "robotics", "ros", "robot operating system", "slam",
    "speech recognition", "asr", "automatic speech recognition",
    "text to speech", "tts", "speech synthesis", "audio processing",
    "lidar", "point cloud", "3d vision", "autonomous driving",
}

# LangChain-era wrapper indicators
_LANGCHAIN_WRAPPER_SKILLS = {
    "langchain", "llamaindex", "llama index", "openai api",
    "gpt-4", "gpt-3", "chatgpt", "prompt engineering",
}

# Pre-LLM ML production skills
_PRE_LLM_ML_SKILLS = {
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "random forest", "svm", "feature engineering", "model deployment",
    "mlops", "ml pipeline", "recommendation system", "ranking",
    "retrieval", "embeddings", "bert", "transformers", "nlp",
    "computer vision", "time series", "forecasting",
}


def check_veto(candidate, jd_features):
    """
    Checks a candidate against the 6 JD-specific VETO disqualifiers.

    Returns:
        tuple: (is_veto: bool, veto_reason: str or None)
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    current_title = profile.get("current_title", "").lower()

    # Gather all skill names (lowercase)
    skill_names = {s.get("name", "").lower() for s in skills}

    # Gather all career titles and company names
    all_titles = [current_title] + [c.get("title", "").lower() for c in career]
    all_companies = [c.get("company", "").lower() for c in career]

    # ---- VETO 1: Pure Research ----
    if _check_pure_research(all_titles, career):
        return True, VetoType.PURE_RESEARCH.value

    # ---- VETO 2: LangChain Wrapper ----
    if _check_langchain_wrapper(skill_names, career):
        return True, VetoType.LANGCHAIN_WRAPPER.value

    # ---- VETO 3: Architect / No Code ----
    if _check_architect_no_code(current_title, career):
        return True, VetoType.ARCHITECT_NO_CODE.value

    # ---- VETO 4: Consulting Only ----
    if _check_consulting_only(all_companies):
        return True, VetoType.CONSULTING_ONLY.value

    # ---- VETO 5: Title Chaser ----
    if _check_title_chaser(career):
        return True, VetoType.TITLE_CHASER.value

    # ---- VETO 6: CV / Speech / Robotics Only ----
    if _check_cv_speech_robotics(skill_names):
        return True, VetoType.CV_SPEECH_ROBOTICS.value

    return False, None


def _check_pure_research(all_titles, career):
    """VETO 1: Spent career in pure research without production deployment."""
    if not career or len(career) < 2:
        return False

    research_months = 0
    total_months = 0
    for job in career:
        title = job.get("title", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        if any(kw in title for kw in _RESEARCH_TITLE_KEYWORDS):
            research_months += dur

    if total_months == 0:
        return False

    # If 80%+ of career time is research-only titles, VETO
    if research_months / total_months >= 0.80:
        return True

    return False


def _check_langchain_wrapper(skill_names, career):
    """VETO 2: LangChain-era skills <12mo without pre-LLM ML production experience."""
    has_langchain_skills = bool(skill_names & _LANGCHAIN_WRAPPER_SKILLS)
    if not has_langchain_skills:
        return False

    has_pre_llm_ml = bool(skill_names & _PRE_LLM_ML_SKILLS)
    if has_pre_llm_ml:
        return False  # They have real ML background, not just wrappers

    # Check if their career history shows substantive ML work
    career_text = " ".join(c.get("description", "").lower() for c in career)
    ml_in_career = any(kw in career_text for kw in [
        "machine learning", "deep learning", "model training",
        "feature engineering", "ml pipeline", "recommendation",
        "ranking", "retrieval", "embeddings",
    ])
    if ml_in_career:
        return False

    # LangChain skills present, no real ML background → VETO
    return True


def _check_architect_no_code(current_title, career):
    """VETO 3: Hasn't written production code in 18+ months — architect/VP only."""
    # Check if current title is arch/VP
    is_current_arch = any(kw in current_title for kw in _ARCH_TITLE_KEYWORDS)
    if not is_current_arch:
        return False

    # Check if last 2 roles (or last 18 months) are all arch/leadership
    if len(career) < 2:
        return False

    # Sort by most recent first (highest duration_months may not mean most recent,
    # so we use start_date if available)
    def get_sort_key(c):
        d = parse_date(c.get("start_date"))
        return d if d is not None else datetime.min

    sorted_career = sorted(career, key=get_sort_key, reverse=True)
    recent_months = 0
    all_arch = True
    for job in sorted_career:
        title = job.get("title", "").lower()
        dur = job.get("duration_months", 0)
        if not any(kw in title for kw in _ARCH_TITLE_KEYWORDS):
            all_arch = False
            break
        recent_months += dur
        if recent_months >= 18:
            break

    if all_arch and recent_months >= 18:
        return True

    return False


def _check_consulting_only(all_companies):
    """VETO 4: Entire career at consulting firms without product-company experience."""
    if not all_companies:
        return False

    # Filter out empty company names
    valid_companies = [c for c in all_companies if c.strip()]
    if not valid_companies:
        return False

    # A single consulting company is sufficient to trigger the veto if it's their entire career

    consulting_count = sum(
        1 for c in valid_companies
        if any(re.search(r'\b' + re.escape(firm) + r'\b', c) for firm in _CONSULTING_FIRMS)
    )

    # If ALL companies are consulting firms → VETO
    if consulting_count == len(valid_companies):
        return True

    return False


def _check_title_chaser(career):
    """VETO 5: Switching companies every <18 months on average across 4+ jobs."""
    if len(career) < 4:
        return False

    durations = [c.get("duration_months", 0) for c in career]
    if not durations:
        return False

    avg_tenure = sum(durations) / len(durations)

    # Average tenure <18 months across 4+ hops → VETO
    if avg_tenure < 18 and len(career) >= 4:
        return True

    return False


def _check_cv_speech_robotics(skill_names):
    """VETO 6: Primary expertise is CV/speech/robotics without NLP/IR crossover."""
    cv_sr_count = len(skill_names & _CV_SPEECH_ROBOTICS_SKILLS)
    nlp_ir_count = len(skill_names & _NLP_IR_SKILLS)

    if cv_sr_count == 0:
        return False

    # Only veto if CV/speech/robotics is truly dominant:
    # Need 4+ CV/speech/robotics skills AND zero NLP/IR skills
    if cv_sr_count >= 4 and nlp_ir_count == 0:
        return True

    # If CV/speech/robotics is >80% of domain skills and significant count
    total_domain = cv_sr_count + nlp_ir_count
    if total_domain >= 5 and cv_sr_count / total_domain >= 0.80 and nlp_ir_count == 0:
        return True

    return False
