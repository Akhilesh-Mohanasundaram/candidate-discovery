"""
Submission Validator for India.Runs Hackathon
Validates the CSV submission against the spec in submission_spec.txt.

Usage:
    python validate_submission.py --csv submission.csv [--candidates candidates.jsonl]
"""
import csv
import sys
import argparse
import json
import gzip
import os


def validate(csv_path, candidates_path=None):
    errors = []
    warnings = []

    # --- 1. Load CSV ---
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"FAIL: File '{csv_path}' not found.")
        return False
    except Exception as e:
        print(f"FAIL: Could not parse CSV -- {e}")
        return False

    # --- 2. Check row count ---
    if len(rows) != 100:
        errors.append(f"Expected exactly 100 data rows, got {len(rows)}.")

    # --- 3. Check required columns ---
    required_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    if rows:
        header = list(rows[0].keys())
        for col in required_cols[:3]:  # reasoning is optional per spec
            if col not in header:
                errors.append(f"Missing required column: '{col}'.")
        if 'reasoning' not in header:
            warnings.append("Column 'reasoning' is missing. Strongly recommended for Stage 4.")

    # --- 4. Validate ranks ---
    ranks = []
    for i, row in enumerate(rows):
        try:
            r = int(row.get('rank', ''))
            ranks.append(r)
        except (ValueError, TypeError):
            errors.append(f"Row {i+1}: rank is not a valid integer ('{row.get('rank', '')}')")

    if ranks:
        expected_ranks = set(range(1, 101))
        actual_ranks = set(ranks)
        if actual_ranks != expected_ranks:
            missing = expected_ranks - actual_ranks
            extra = actual_ranks - expected_ranks
            if missing:
                errors.append(f"Missing ranks: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            if extra:
                errors.append(f"Unexpected ranks: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")
        if len(ranks) != len(set(ranks)):
            errors.append("Duplicate ranks detected.")

    # --- 5. Validate candidate IDs ---
    cids = [row.get('candidate_id', '') for row in rows]
    if len(cids) != len(set(cids)):
        errors.append("Duplicate candidate_ids detected.")

    for cid in cids:
        if not cid or not cid.startswith('CAND_'):
            errors.append(f"Invalid candidate_id format: '{cid}'")

    # --- 6. Validate scores ---
    scores = []
    for i, row in enumerate(rows):
        try:
            s = float(row.get('score', ''))
            scores.append(s)
        except (ValueError, TypeError):
            errors.append(f"Row {i+1}: score is not a valid float ('{row.get('score', '')}')")

    if scores and len(scores) == len(rows):
        # Check monotonically non-increasing
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1] + 1e-9:  # small epsilon for float precision
                errors.append(f"Score at rank {i+1} ({scores[i]}) > score at rank {i} ({scores[i-1]}). Scores must be non-increasing.")
                break

    # --- 7. Validate against candidates file (optional) ---
    if candidates_path and os.path.exists(candidates_path):
        valid_ids = set()
        try:
            if candidates_path.endswith('.gz'):
                f = gzip.open(candidates_path, 'rt', encoding='utf-8')
            elif candidates_path.endswith('.json'):
                with open(candidates_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    valid_ids = {c.get('candidate_id') for c in data}
                    f = None
            else:
                f = open(candidates_path, 'r', encoding='utf-8')

            if f:
                count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            valid_ids.add(obj.get('candidate_id'))
                        except Exception:
                            pass
                        count += 1
                        if count % 1000 == 0:
                            sys.stdout.write(f"\r  Scanned {count} candidates for validation...")
                            sys.stdout.flush()
                f.close()
                if count >= 1000:
                    print()  # newline after loop

            invalid = [cid for cid in cids if cid not in valid_ids]
            if invalid:
                errors.append(f"candidate_ids not found in dataset: {invalid[:5]}{'...' if len(invalid) > 5 else ''}")
            else:
                print(f"  [OK] All candidate_ids verified against {os.path.basename(candidates_path)}")
        except Exception as e:
            warnings.append(f"Could not validate against candidates file: {e}")

    # --- 8. Check reasoning quality ---
    reasoning_entries = [row.get('reasoning', '').strip() for row in rows]
    empty_count = sum(1 for r in reasoning_entries if not r)
    if empty_count > 0:
        warnings.append(f"{empty_count} rows have empty reasoning (strongly recommended for Stage 4).")

    unique_reasonings = set(reasoning_entries)
    if len(unique_reasonings) < len(rows) * 0.5:
        warnings.append("Low reasoning variation detected -- many duplicate strings. Stage 4 penalizes templated reasoning.")

    # --- Output ---
    print(f"\n{'='*60}")
    print(f"  Submission Validation Report")
    print(f"  File: {csv_path}")
    print(f"{'='*60}")
    print(f"  Total rows: {len(rows)}")
    print(f"  Unique candidate_ids: {len(set(cids))}")
    if scores:
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")

    if errors:
        print(f"\n  [FAIL] ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  [OK] All format checks passed.")

    if warnings:
        print(f"\n  [WARN] WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    print(f"{'='*60}")

    if errors:
        print("  RESULT: FAIL -- fix errors before submitting.\n")
        return False
    else:
        print("  RESULT: PASS -- submission is format-valid.\n")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate hackathon submission CSV.')
    parser.add_argument('--csv', default='submission.csv', help='Path to submission CSV')
    parser.add_argument('--candidates', default=None, help='Path to candidates.jsonl or .json for ID verification')
    args = parser.parse_args()
    success = validate(args.csv, args.candidates)
    sys.exit(0 if success else 1)
