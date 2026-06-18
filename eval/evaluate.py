"""
Evaluation Script
=================
This script analyzes your submission.csv to ensure the distribution
of scores and logic seems correct.

Note on Metrics (NDCG@10, NDCG@50, MAP, P@10):
As per the India.Runs hackathon specification:
"Scoring happens once, after submissions close. There is no public partition,
no live leaderboard, and no per-submission feedback during the competition.
Your score is computed against the full hidden ground truth."

You CANNOT check your true NDCG or MAP metrics locally because the ground truth
(which candidates are actually the best) is hidden by the organizers.

The previous `ground_truth.json` file was a synthetic file and has been removed
to avoid confusion, as it produced inaccurate 0/20 scores.

This script instead validates the distribution of your submission.
"""
import csv
import numpy as np

def evaluate_distribution():
    try:
        with open('submission.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            submission = list(reader)
    except FileNotFoundError:
        print("Error: submission.csv not found. Run rank.py first.")
        return

    if not submission:
        print("Error: submission.csv is empty.")
        return

    print("=" * 50)
    print("SUBMISSION DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # 1. Basic Stats
    ranks = [int(row['rank']) for row in submission]
    scores = [float(row['score']) for row in submission]
    
    print(f"Total Candidates Ranked: {len(submission)}")
    print(f"Score Range: {min(scores):.4f} to {max(scores):.4f}")
    
    # 2. Score Monotonicity Check
    is_monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    print(f"Scores are Monotonically Decreasing: {'YES' if is_monotonic else 'NO'}")
    
    # 3. Check for Score Variance (Are they all the same?)
    unique_scores = len(set(scores))
    print(f"Unique Score Values: {unique_scores}/{len(submission)}")
    if unique_scores < 10:
        print("  Warning: Very low score variance. Your model might not be differentiating candidates well.")
        
    # 4. Check for 'Concern' or 'Veto' in reasoning
    concerns = sum(1 for row in submission if 'Concern:' in row.get('reasoning', '') or 'VETO' in row.get('reasoning', ''))
    print(f"Reasoning includes specific concerns/nuance: {concerns} candidates")
    if concerns == 0:
        print("  Warning: No concerns listed in reasoning. You might lose points in Stage 4 Manual Review if reasoning is too generic.")

    print("\n" + "=" * 50)
    print("HOW TO CHECK TRUE METRICS?")
    print("=" * 50)
    print("You cannot check NDCG@10 or MAP locally.")
    print("Your final score will be calculated by the organizers when you upload")
    print("this submission.csv to the portal.")
    print("\nNext Steps:")
    print("1. Validate format using `python validate_submission.py`")
    print("2. Submit the CSV and your repository to the hackathon portal.")

if __name__ == "__main__":
    evaluate_distribution()
