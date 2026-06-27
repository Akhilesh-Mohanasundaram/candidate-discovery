---
title: Legend Acers Candidate Ranker
emoji: ⚡
colorFrom: purple
colorTo: teal
sdk: streamlit
sdk_version: 1.37.1
app_file: app.py
pinned: false
license: mit
short_description: AI-powered candidate ranking for India.Runs 2026 Track 01
---

# 🏆 Intelligent Candidate Discovery & Ranking System

> **Team Legend Acers** — India.Runs 2026 | Track 01: Data & AI Challenge

An end-to-end AI-powered ranking system that discovers the top 100 most qualified AI engineering candidates from a pool of 100,000+ profiles — not by matching keywords, but by actually understanding who fits the role.

---

## 🎯 Problem Statement

Traditional ATS systems rely on keyword matching, which fails to capture what actually matters: semantic skill alignment, career trajectory, behavioral availability, and profile authenticity. Our system acts as a **superhuman technical recruiter** by combining deep NLP understanding with multi-signal feature engineering and rigorous fraud detection.

## ⚡ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PHASE (precompute.py)                   │
│                                                                         │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────────┐       │
│  │ JD Parser │──▶│ Honeypot  │──▶│   VETO   │──▶│   Feature    │       │
│  │  (S1)     │   │ Detector  │   │ Checker  │   │  Engineer    │       │
│  │           │   │  (S2a)    │   │  (S2b)   │   │   (S3)       │       │
│  └──────────┘   └───────────┘   └──────────┘   └──────────────┘       │
│       │                                                │               │
│       ▼                                                ▼               │
│  jd_features.json                       ┌───────────────┐             │
│                                         │   Semantic     │             │
│                                         │ Skill Match(S4)│             │
│                                         └───────────────┘             │
│                                                │                       │
│                                                ▼                       │
│                                         ┌───────────────┐             │
│                                         │  Behavioral   │             │
│                                         │  Multiplier   │             │
│                                         │     (S5)      │             │
│                                         └───────────────┘             │
│                                                │                       │
│                                                ▼                       │
│                                         feature_matrix.npz             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ONLINE PHASE (rank.py) — ≤ 5 min CPU              │
│                                                                         │
│  ┌──────────────────┐   ┌───────────────┐   ┌────────────────────────┐ │
│  │  Vectorized       │──▶│  Apply        │──▶│  Reasoning Generation │ │
│  │  Score Fusion     │   │  Multiplier   │   │  + CSV Output         │ │
│  │  (numpy)          │   │               │   │  (S6)                 │ │
│  └──────────────────┘   └───────────────┘   └────────────────────────┘ │
│                                                        │               │
│                                                        ▼               │
│                                                 submission.csv         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🧠 Methodology

### Stage 1 — JD Understanding & Decomposition
Parses the job description into structured requirement tiers:
- **HARD requirements**: Must-have skills (embeddings, vector databases, Python, ranking evaluation)
- **SOFT preferences**: Nice-to-have skills (LLM fine-tuning, learning-to-rank, HR-tech)
- **VETO disqualifiers**: Automatic rejection criteria (pure research, consulting-only, title-chaser)

*Note (Tech Debt): Due to the fixed JD in this hackathon, these tiers are currently hardcoded in `jd_parser.py` rather than dynamically extracted via an LLM. Dynamic extraction would be required for a production system.*

### Stage 2a — Honeypot & Trap Detection
Implements **7 strict detection rules** to catch fabricated profiles before they enter the scoring pipeline:

| # | Trap Type | Detection Logic |
|---|-----------|-----------------|
| 1 | **Time Paradox** | Career history months vs. stated YoE differ by >2 years |
| 2 | **Fake Expert** | "Expert" proficiency with 0 months duration |
| 3 | **Keyword Stuffer** | ≥10 skills at "expert" level |
| 4 | **Title Mismatch** | AI skills but only non-technical titles throughout career |
| 5 | **Timeline Overlap** | 3+ month overlap between concurrent positions (handles 'current' jobs dynamically) |
| 6 | **Ghost Skills** | >80% of advanced/expert skills absent from descriptions (adjusted to catch low-volume fabrications) |
| 7 | **Behavioral Ghost** | Inactive >180 days AND <5% recruiter response rate |

All honeypot candidates receive a hard **0.0 score**.

### Stage 2b — JD VETO Disqualifier Checks
Applies the **6 disqualifier rules** explicitly stated in the JD to filter misaligned (but real) candidates:

| # | VETO Type | Detection Logic |
|---|-----------|-----------------|
| 1 | **Pure Research** | 80%+ career time in academic/research-only roles without production deployment |
| 2 | **LangChain Wrapper** | LLM-wrapper-only skills (<12mo) without pre-LLM ML production background |
| 3 | **Architect / No Code** | Current title is architect/VP + last 18+ months in non-coding roles (uses robust datetime sorting) |
| 4 | **Consulting Only** | Entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini (catches single-company careers via word-boundary matching) |
| 5 | **Title Chaser** | Average tenure <18 months across 4+ positions (3+ job hops) |
| 6 | **CV/Speech/Robotics** | Primary expertise in CV/speech/robotics without NLP/IR crossover |

All VETO candidates also receive a hard **0.0 score**.

### Stage 3 — Multi-Signal Feature Engineering
Extracts features across 3 families per candidate:
- **Career Quality**: Title progression, company size tier, tenure stability, technical role relevance
- **Logistics Score**: Location match against target cities, notice period penalty, relocation willingness
- **YoE Alignment**: Soft penalty for candidates outside the 5–9 year expected range

### Stage 4 — Hybrid Scoring Engine
Combines signals using a weighted fusion formula:

```
Raw Score = 0.45 × Career Quality + 0.40 × Semantic Skill Match + 0.15 × Logistics
Final Score = Raw Score × Behavioral Multiplier
```

**Semantic Skill Matching** uses `all-MiniLM-L6-v2` (sentence-transformers) to compute cosine similarity between candidate skills and JD requirements, weighted by:
- Proficiency level (beginner: 0.3 → expert: 1.0)
- Duration of usage (normalized to 36 months, zero duration strictly yields zero weight)
- Endorsement count (logarithmic scaling, up to 5% bonus)

Matches are strictly grouped by the underlying JD requirement to prevent artificial score saturation, and the cosine similarity threshold is aggressively tuned (0.45) via ablation testing to ensure high signal-to-noise ratio.

VETO checks are applied first — any disqualified profile returns 0.0 immediately.

### Stage 5 — Behavioral Signal Multiplier
Analyzes all **23 Redrob behavioral signals** organized into 5 weighted groups:

| Group | Weight | Signals Used |
|-------|--------|-------------|
| Activity & Recency | 25% | `signup_date`, `last_active_date`, `open_to_work_flag`, `applications_submitted_30d` |
| Responsiveness | 22% | `recruiter_response_rate`, `avg_response_time_hours` |
| Profile Quality | 13% | `profile_completeness_score`, `connection_count`, `endorsements_received`, verification flags |
| Hiring Readiness | 20% | `notice_period_days`, `expected_salary_range_inr_lpa` (realistic baseline 20–75 LPA), `preferred_work_mode` (remote-preference receives soft 0.7x penalty), `willing_to_relocate`, `interview_completion_rate`, `offer_acceptance_rate` |
| Market Demand | 20% | `profile_views_received_30d`, `skill_assessment_scores` (passive candidates receive neutral 0.5 floor), `github_activity_score`, `search_appearance_30d`, `saved_by_recruiters_30d` |

The multiplier is clamped to **[0.3 – 1.0]** using exponential recency decay (`e^(-days/90)`).

### Stage 6 — Reasoning Generation
Generates candidate-specific, non-templated reasoning that:
- References actual skills, titles, and YoE from the profile (including impactful startup stints down to 6 months)
- Connects to specific JD requirements
- Honestly surfaces concerns (long notice period, low response rate, inactivity)
- Adapts tone to match the candidate's rank position (reasons are mapped strictly via a central `VetoType` enum)

## 📂 Repository Structure

```
candidate-discovery/
├── precompute.py                      # Offline: scans 100K → feature_matrix.npz
├── rank.py                            # Online: reads matrix → submission.csv
├── validate_submission.py             # CSV format validator
├── submission_metadata.yaml           # Portal metadata
├── requirements.txt
├── README.md                          # This file
│
├── dataset/                           # Hackathon provided data
│   ├── candidates.jsonl               # 100K profiles
│   └── sample_candidates.json         # Sample profiles
│
├── docs/                              # Hackathon guidelines & docs
│   ├── job_description.txt
│   └── submission_spec.txt
│
├── src/                               # Core pipeline logic
│   ├── constants.py                   # NEW: shared enums/VETO types
│   ├── jd_parser.py
│   ├── honeypot.py
│   ├── veto_checker.py
│   ├── feature_engineer.py
│   ├── scorer.py
│   ├── behavioral.py
│   └── reasoning_gen.py
│
├── artifacts/                         # Pre-computed state
│   ├── jd_features.json
│   ├── feature_matrix.npz
│   └── candidate_index.json           # NEW: fast lookup
│
├── tests/                             # NEW: Fully automated pytest suite
│   ├── test_honeypot.py
│   ├── test_veto.py
│   └── test_scorer.py
│
├── eval/
│   └── evaluate.py                    # Submission distribution analyzer
│
└── sandbox/
    ├── app.py                         # Streamlit demo application
    └── requirements.txt               # Sandbox-specific dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Pre-compute Feature Matrix (offline, can exceed 5 min)

```bash
python precompute.py --candidates dataset/candidates.jsonl
```

This generates `artifacts/feature_matrix.npz` containing pre-scored features for all candidates.

### 3. Run the Ranker (must complete within 5 minutes)

```bash
python rank.py --candidates dataset/candidates.jsonl --out submission.csv
```

*Note: `rank.py` relies entirely on a lightweight candidate index generated during precompute, avoiding massive redundant sequential file scans during the online phase.*

### 4. Validate the Submission

```bash
python validate_submission.py --csv submission.csv --candidates dataset/candidates.jsonl
```

### 5. Launch the Demo Sandbox (locally)

```bash
cd sandbox && streamlit run app.py
```

Upload a JSON file with ≤100 candidates to see the ranker in action with a bias comparison panel.

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/<USERNAME>/legend-acers-ranker)

### 6. Run the Test Suite

```bash
pytest tests/
```

## 📊 Evaluation & Metrics

As per the hackathon specification, **local calculation of true metrics (NDCG, MAP) is impossible** because the actual ground truth is hidden by the organizers. 

We maintain a local submission distribution analyzer to ensure our ranking scores are monotonic, varied, and properly reasoned before submission:

```bash
python eval/evaluate.py
```

This script validates:
- Score distribution variance
- Monotonic decreasing order
- Inclusion of nuanced reasoning and concerns

Your final score (Composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10) will be computed automatically by the hackathon portal when you upload your `submission.csv`.

## ⚙️ Compute Budget

| Constraint | Our System |
|------------|-----------|
| Runtime | **< 7 seconds** (ranking phase) |
| Memory | ~2 GB peak |
| Compute | CPU only (numpy vectorization) |
| Network | None (fully offline) |
| Disk | < 500 MB intermediate state |

The ranking step uses vectorized numpy operations on the pre-computed feature matrix — no per-candidate model inference during the online phase.

## 🛡️ AI Tools Declaration

We used **Claude** (Anthropic) and **Gemini** (Google) as a development assistant for code iteration, architecture discussions, and documentation. All engineering decisions, weight tuning, ground truth labeling, and system design were performed by team members with full understanding of the codebase.

## 👥 Team

**Legend Acers** — India.Runs 2026  

- [Akhilesh Mohanasundaram](https://github.com/Akhikesh-Mohanasundaram).  
- [Kishore B](https://github.com/Kishore-1803)  
- [Sanggit Saaran K C S](https://github.com/sanggitsaaran)  
- [Venkatram K S](https://github.com/venkatramks)  
---

<p align="center">
  <em>Built for the India.Runs 2026 Hackathon — Track 01: Intelligent Candidate Discovery</em><br/>
  <a href="https://hack2skill.com/event/india_runs/">hack2skill.com/event/india_runs</a>
</p>
