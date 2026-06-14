# Intelligent Candidate Discovery Ranker ⚡
**Team:** Legend Acers  
**Hackathon:** India.Runs Data & AI Challenge (Track 01)

This repository contains an end-to-end AI-powered ranking system designed to discover the top 100 most qualified AI engineering candidates from a pool of 100,000 resumes. 

Instead of relying on flawed keyword-matching ATS, this system acts as a superhuman technical recruiter by mathematically measuring semantic skill alignment, career stability, logistics, and crucial platform behavioral availability.

## 🌟 Key Features

- **Hybrid Semantic Scoring (35%):** Utilizes `SentenceTransformers` (`all-MiniLM-L6-v2`) to map candidate skills against Job Description requirements, weighing proficiency and duration of usage.
- **Career & Logistics (50%):** Standardizes title progression, company tier, and tenure stability. Penalizes candidates with poor location overlap or 90+ day notice periods.
- **Behavioral Multiplier (15%):** Analyzes `redrob_signals`. Applies an exponential recency decay (`e^(-days/90)`) to penalize candidates who have abandoned the platform, alongside evaluating recruiter response rates.
- **Honeypot Enforcement (Veto):** Implements 7 strict logical traps (e.g., Timeline Overlap, Fake Experts, Keyword Stuffers) that instantly hard-reject fabricated or hallucinated candidate profiles.
- **Extreme Speed (Offline/Online Split):** Precomputes the 487MB `candidates.jsonl` into a highly compressed Numpy array matrix, allowing the actual ranking script to execute in **under 7 seconds**.
- **Automated Reasoning:** Deterministically generates a 1-2 sentence human-readable justification for the exact placement of each Top-100 candidate.

## 📂 Repository Structure

- `src/`
  - `jd_parser.py`: Parses unstructured JDs to extract hard/soft requirements.
  - `scorer.py`: Core logic for semantic skill cosine similarity matching.
  - `reasoning_gen.py`: Dynamic templating engine for recruiter justifications.
  - `honeypot.py`: Implementation of the 7-trap fabrication detection logic.
  - `feature_engineer.py`: Extracts career quality and logistical numerical signals.
  - `behavioral.py`: Calculates the behavioral availability multiplier.
- `eval/`
  - `ground_truth.json`: 20 manually graded profiles for system calibration.
  - `evaluate.py`: Calculates `NDCG@10` against the ground truth to tune weights.
- `sandbox/`
  - `app.py`: An interactive Streamlit demo application.
- `precompute.py`: Scans 100K candidates offline to build `artifacts/feature_matrix.npz`.
- `rank.py`: The lightning-fast 5-minute ranking execution script. Produces `submission.csv`.
- `honeypot_rules_spec.md`: The documented logical specifications for the trap filters.
- `portal_methodology_summary.txt`: The system architecture summary for the hackathon portal.

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install sentence-transformers scikit-learn numpy pandas streamlit
   ```

2. **Generate Offline Matrix:**
   *Note: Ensure `candidates.jsonl` is placed in the root or updated in the script path.*
   ```bash
   python precompute.py
   ```

3. **Execute Ranker:**
   Produces `submission.csv` containing the Top 100 ranking.
   ```bash
   python rank.py
   ```

4. **Launch the Demo Sandbox UI:**
   ```bash
   cd sandbox && streamlit run app.py
   ```

## 🧠 Methodology Check

Our ranking formula executes at $O(1)$ time per candidate during the live phase due to precomputation.
The final score combines the vectors as:
```math
Final = (0.45 * Career + 0.40 * Semantic + 0.15 * Logistics) * Behavioral
```
Any candidate triggering a `check_honeypot()` failure has their score zeroed.
