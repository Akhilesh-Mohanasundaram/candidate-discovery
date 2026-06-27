<div align="center">

# ⚡ Intelligent Candidate Discovery & Ranking System

### Team Legend Acers — India.Runs 2026 | Track 01: Data & AI Challenge

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-purple?style=for-the-badge)](https://huggingface.co/spaces/venkat-1212/legend-acers-ranker)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Akhilesh-Mohanasundaram/candidate-discovery)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Rank candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role.*

</div>

---

## 🎯 Problem Statement

Traditional ATS systems fail because they rely on keyword matching — they can't distinguish a genuine senior AI engineer from someone who stuffed the right words into their profile. Our system acts as a **superhuman technical recruiter** by combining:

- **Semantic NLP** to understand skill equivalences beyond exact keywords
- **Multi-signal behavioral scoring** to assess genuine hiring readiness
- **Rigorous fraud detection** to eliminate fabricated and misaligned profiles
- **Deterministic reasoning** to justify every ranking decision with real profile data

> Given 100,000+ candidate profiles, we surface the **top 100 most qualified Senior AI Engineers** — fraud-free, semantically ranked, with per-candidate justification.

---

## 🏗️ Architecture

The system runs in two phases to meet the 5-minute online constraint:

```
╔══════════════════════════════════════════════════════════════════╗
║           OFFLINE PHASE  —  precompute.py  (no time limit)      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  [Stage 1]        [Stage 2a]       [Stage 2b]      [Stage 3]    ║
║  JD Parser   →  Honeypot Det.  →  VETO Checker  →  Feature Eng. ║
║  HARD/SOFT/      7 Trap Rules      6 JD Rules      Career +      ║
║  VETO Tiers      Fraud Filter      Disqualifier    Logistics     ║
║                                                                  ║
║  [Stage 4]                         [Stage 5]                    ║
║  Semantic Skill Match          →   Behavioral Multiplier        ║
║  MiniLM-L6-v2 Embeddings           23 Redrob Signals           ║
║                                                                  ║
║                    ↓  artifacts/feature_matrix.npz  ↓           ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║        ONLINE PHASE  —  rank.py  (< 5 minutes, CPU-only)        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Load Matrix  →  Score Fusion  →  Apply Multiplier  →  Top 100  ║
║  (numpy)         Vectorized        Hard VETO Gates    lexsort   ║
║                                                                  ║
║  [Stage 6]  Reasoning Generation  →  submission.csv             ║
║             Per-candidate, data-grounded, non-templated         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

**Actual runtime: < 7 seconds** (40× under the 5-minute constraint)

---

## 🧠 Methodology

### Stage 1 — JD Understanding & Decomposition

The job description is parsed into structured requirement tiers that drive every downstream stage:

| Tier | Skills | Purpose |
|------|--------|---------|
| **HARD** | Embeddings, Vector DB, Python, Ranking Evaluation | Must-have — weighted 40% of score |
| **SOFT** | LLM Fine-tuning, Learning-to-Rank, HR-tech, Distributed Systems | Nice-to-have — boost score |
| **VETO** | 6 disqualifier types (see Stage 2b) | Hard gate — score forced to 0.0 |

---

### Stage 2a — Honeypot & Fraud Detection

Seven distinct trap rules catch fabricated profiles **before** they enter the scoring pipeline. All honeypot candidates receive a hard **0.0 score**.

| # | Trap | Detection Logic |
|---|------|----------------|
| 1 | **Time Paradox** | Career duration vs. stated YoE differ by > 2 years |
| 2 | **Fake Expert** | `expert` proficiency declared with 0 months duration |
| 3 | **Keyword Stuffer** | ≥ 10 skills at `expert` level simultaneously |
| 4 | **Title Mismatch** | AI skills listed but only non-technical titles throughout career |
| 5 | **Timeline Overlap** | > 3 month overlap between concurrent positions (current jobs handled via `datetime.now()`) |
| 6 | **Ghost Skills** | > 80% of advanced/expert skills absent from all career descriptions (word-boundary regex) |
| 7 | **Behavioral Ghost** | Inactive > 180 days **AND** recruiter response rate < 5% |

---

### Stage 2b — JD VETO Disqualifier Checks

Six role-specific disqualifiers filter misaligned (but real) candidates. All VETO candidates receive a hard **0.0 score**.

| # | VETO | Detection Logic |
|---|------|----------------|
| 1 | **Pure Research** | 80%+ career time in academic/research roles without production deployment |
| 2 | **LangChain Wrapper** | LLM-wrapper-only skills (< 12 months) without pre-LLM ML production background |
| 3 | **Architect / No Code** | Current architect/VP title + 18+ months without hands-on coding (datetime-sorted) |
| 4 | **Consulting Only** | Entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini (word-boundary matched, single-company careers included) |
| 5 | **Title Chaser** | Average tenure < 18 months across 4+ job hops |
| 6 | **CV / Speech / Robotics** | Primary domain is CV/speech/robotics with zero NLP/IR overlap |

---

### Stage 3 — Multi-Signal Feature Engineering

Two normalized scores [0.0–1.0] per candidate:

**Career Quality Score (45% of final)**
- Title progression (AI/ML engineer vs. generic)
- Company size tier (startup → FAANG, 6 tiers)
- Tenure stability (capped at 48-month maximum per role)
- YoE soft penalty for candidates outside the 5–9 year target range

**Logistics Score (15% of final)**
- Location match against 6 target cities (Pune, Noida, Mumbai, Delhi NCR, Delhi, Hyderabad)
- Notice period penalty (≤ 30 days = 1.0, > 90 days = 0.2)
- Relocation willingness

---

### Stage 4 — Semantic Skill Matching (40% of final)

```
skill_score = cosine_sim(skill_embed, jd_embed) × proficiency_weight × duration_weight × endorsement_bonus

proficiency_weight : beginner=0.3 | intermediate=0.6 | advanced=0.8 | expert=1.0
duration_weight    : 0.0 if months==0, else min(months/36, 1.0)   ← guards against fabricated claims
endorsement_bonus  : 1.0 + min(0.05, log1p(endorsements)/100)     ← capped at +5%
similarity_threshold: 0.45 (ablation-tuned)
```

- Model: `all-MiniLM-L6-v2` (sentence-transformers, 384-dim)
- Matches are **grouped by underlying JD requirement** — prevents score saturation from synonym flooding
- JD embeddings are **cached globally** — computed once per run, not per candidate

---

### Stage 5 — Behavioral Signal Multiplier

All **23 Redrob behavioral signals** are processed as a multiplier applied to the raw score:

```
multiplier = clamp(0.3 + 0.7 × raw_composite, 0.3, 1.0)
```

| Group | Weight | Signals |
|-------|--------|---------|
| Activity & Recency | 25% | `signup_date`, `last_active_date` (exp decay `e^(-days/90)`), `open_to_work_flag`, `applications_submitted_30d` |
| Responsiveness | 22% | `recruiter_response_rate`, `avg_response_time_hours` |
| Profile Quality | 13% | `profile_completeness_score`, `connection_count`, `endorsements_received`, verification flags |
| Hiring Readiness | 20% | `notice_period_days`, `expected_salary_range_inr_lpa` (20–75 LPA band), `preferred_work_mode`, `willing_to_relocate`, `interview_completion_rate`, `offer_acceptance_rate` |
| Market Demand | 20% | `profile_views_received_30d`, `skill_assessment_scores`, `github_activity_score`, `search_appearance_30d`, `saved_by_recruiters_30d` |

The multiplier floor of **0.3** ensures no candidate is completely suppressed by behavioral signals alone — only hard VETO and honeypot gates force a score to 0.0.

---

### Stage 6 — Reasoning Generation

Every ranked candidate receives a unique, non-templated justification that:

- **Names actual skills, companies, and durations** from the profile — never generic phrases
- **Connects matched skills to specific JD requirements** (HARD and SOFT)
- **Honestly surfaces concerns** — long notice periods, low response rates, YoE boundary, location mismatch
- **Adapts tone to rank position** — top 40 candidates receive career highlight sentences
- **Maps VETO/honeypot reasons** via a shared `VetoType` enum (no silent drift between modules)
- Capped at **350 characters** for CSV readability

---

## 📊 Scoring Formula

```
Raw   = 0.45 × Career_Quality + 0.40 × Semantic_Match + 0.15 × Logistics
Final = Raw × Behavioral_Multiplier          # multiplier ∈ [0.3, 1.0]
Final = 0.0  if honeypot OR veto             # hard gate — no exceptions
```

Tiebreaking: `numpy.lexsort` on `(-score, candidate_id)` — fully deterministic across runs.

---

## ⚙️ Compute Budget

| Constraint | Requirement | Our System |
|------------|-------------|-----------|
| Online phase runtime | ≤ 5 minutes | **< 7 seconds** |
| Hardware | CPU-only | CPU-only (numpy vectorization) |
| Memory | — | ~2 GB peak |
| Network | None | Fully offline after model download |

The online phase loads a precomputed `feature_matrix.npz` and `candidate_index.json`, runs vectorized numpy scoring, and writes `submission.csv` — **no model inference, no file scanning** during ranking.

---

## 📂 Repository Structure

```
candidate-discovery/
│
├── app.py                         # HuggingFace Spaces entry point (Streamlit)
├── precompute.py                  # Offline: 100K candidates → feature_matrix.npz
├── rank.py                        # Online: matrix → submission.csv  (< 7 sec)
├── validate_submission.py         # CSV format & integrity validator
├── submission_metadata.yaml       # Hackathon portal metadata
├── requirements.txt               # Pinned dependencies
│
├── src/                           # Core pipeline modules
│   ├── constants.py               # Shared VetoType enum (prevents reasoning drift)
│   ├── jd_parser.py               # JD → HARD/SOFT/VETO structured tiers
│   ├── honeypot.py                # 7-trap fraud detection
│   ├── veto_checker.py            # 6-rule JD disqualifier checks
│   ├── feature_engineer.py        # Career quality + logistics scoring
│   ├── scorer.py                  # Semantic skill matching (MiniLM embeddings)
│   ├── behavioral.py              # 23-signal behavioral multiplier
│   └── reasoning_gen.py           # Per-candidate non-templated reasoning
│
├── artifacts/
│   ├── jd_features.json           # Structured JD specification
│   ├── feature_matrix.npz         # Precomputed scores (gitignored — generated)
│   └── candidate_index.json       # Fast O(1) candidate lookup (gitignored — generated)
│
├── tests/
│   ├── test_honeypot.py           # Honeypot trap unit tests
│   ├── test_veto.py               # VETO rule unit tests
│   └── test_scorer.py             # Semantic scorer unit tests
│
├── eval/
│   └── evaluate.py                # Score distribution & reasoning quality analyzer
│
├── sandbox/
│   └── app.py                     # Local dev version of the Streamlit demo
│
└── docs/                          # Hackathon guidelines & spec documents
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pre-compute feature matrix (offline — no time limit)

```bash
python precompute.py --candidates dataset/candidates.jsonl
```

Generates `artifacts/feature_matrix.npz` and `artifacts/candidate_index.json`.

### 3. Run the ranker (online — must complete within 5 minutes)

```bash
python rank.py --candidates dataset/candidates.jsonl --out submission.csv
```

### 4. Validate the submission

```bash
python validate_submission.py --csv submission.csv
```

### 5. Run the test suite

```bash
pytest tests/ -v
```

### 6. Launch the demo locally

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

**Try it now:** [huggingface.co/spaces/venkat-1212/legend-acers-ranker](https://huggingface.co/spaces/venkat-1212/legend-acers-ranker)

The demo accepts a JSON file of up to 100 candidates (or click **"Load 6 Sample Candidates"** to try immediately). It runs the full 6-stage pipeline and shows:
- Ranked results table with per-candidate reasoning
- Score breakdown (Semantic · Career · Logistics · Behavioral)
- Honeypot and VETO detection counts
- Side-by-side bias comparison vs. traditional ATS
- Downloadable `submission.csv`

---

## 📈 Evaluation

As per the hackathon specification, local computation of NDCG/MAP is not possible — ground truth is hidden by the organizers. The final score is computed by the portal:

```
Composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

We maintain a local distribution analyzer to ensure our ranking is monotonic, varied, and properly reasoned:

```bash
python eval/evaluate.py
```

---

## 🛡️ AI Tools Declaration

Development used **Claude (Anthropic)** and **Gemini (Google)** as coding and architecture assistants. All engineering decisions, weight calibration, threshold tuning, and system design were performed and validated by team members with full understanding of the codebase.

---

## 👥 Team Legend Acers

| Name | Role | GitHub |
|------|------|--------|
| **Kishore B** | Systems Lead / Demo Lead | [@Kishore-1803](https://github.com/Kishore-1803) |
| **Sanggit Saaran K C S** | Feature Engineering Lead | [@sanggitsaaran](https://github.com/sanggitsaaran) |
| **Akhilesh Mohanasundaram** | AI Lead / Pipeline Architect | [@Akhilesh-Mohanasundaram](https://github.com/Akhilesh-Mohanasundaram) |
| **Venkatram K S** | Evaluation Lead / Scoring | [@venkatramks](https://github.com/venkatramks) |

---

<div align="center">

*Built for the India.Runs 2026 Hackathon — Track 01: Intelligent Candidate Discovery*

[hack2skill.com/event/india_runs](https://hack2skill.com/event/india_runs)

</div>
