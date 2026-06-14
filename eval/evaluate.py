import json
import csv
import numpy as np

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def evaluate():
    try:
        with open('eval/ground_truth.json', 'r') as f:
            gt_data = json.load(f).get("ground_truth", [])
    except FileNotFoundError:
        print("Error: Ground truth file not found.")
        return
        
    gt_map = {item["candidate_id"]: item["tier"] for item in gt_data}
    
    try:
        with open('submission.csv', 'r') as f:
            reader = csv.DictReader(f)
            submission = list(reader)
    except FileNotFoundError:
        print("Error: submission.csv not found. Run rank.py first.")
        return
        
    # Build relevance array based on submission order
    relevance_array = []
    found_gt_count = 0
    
    for row in submission:
        cid = row["candidate_id"]
        if cid in gt_map:
            relevance_array.append(gt_map[cid])
            found_gt_count += 1
        else:
            # If not in ground truth, assume tier 0 or neutral (we'll use 0 for strict eval)
            relevance_array.append(0)
            
    print(f"Evaluation Results:")
    print(f"Found {found_gt_count} ground truth candidates in the top 100.")
    
    if found_gt_count == 0:
        print("Warning: None of the ground truth candidates appeared in the submission.")
        print("Recommendation: Check if honeypot logic is overly aggressive or semantic weights are too low.")
        return
        
    ndcg_10 = ndcg_at_k(relevance_array, 10)
    ndcg_50 = ndcg_at_k(relevance_array, 50)
    
    # Calculate P@10 (Precision at 10 for tiers >= 3)
    p_10 = sum(1 for r in relevance_array[:10] if r >= 3) / 10.0
    
    print(f"NDCG@10: {ndcg_10:.4f}")
    print(f"NDCG@50: {ndcg_50:.4f}")
    print(f"P@10 (Tier >= 3): {p_10:.4f}")
    
    print("\n--- Tuning Recommendations ---")
    if ndcg_10 < 0.6:
        print("Recommendation: NDCG@10 is low. Consider increasing the Semantic Skill Match weight (currently 35%).")
    else:
        print("Recommendation: NDCG@10 is strong. The current 40/35/15/10 weight distribution is effective.")

if __name__ == "__main__":
    evaluate()
