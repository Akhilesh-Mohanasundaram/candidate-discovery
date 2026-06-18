# Honeypot Detection Rules Spec (Member B Implementation Guide)

This document provides the exact detection logic for the 7 honeypot trap types in the candidate dataset. These candidates MUST be assigned a final score of `0.0`.

Member B: Implement these exact conditions in `src/honeypot.py`.

## Trap 1: Time Paradox
**Concept:** Candidate has more years of experience than their age allows, or claims to have worked at a company before it was founded. Since we don't have company founding dates in the schema, we rely on total `duration_months` in `career_history` vs `years_of_experience` in `profile`.
**Condition:** 
- If `profile.years_of_experience * 12` is vastly different from sum of `duration_months` across all `career_history` entries (e.g., stated 15 years experience but career history sum is only 24 months, or vice versa by a large margin).
- *Strict Rule:* `sum(duration_months)` > `(years_of_experience + 2) * 12` OR `sum(duration_months) < (years_of_experience - 2) * 12` (assuming they filled out their whole history).

## Trap 2: Fake Expert
**Concept:** Claiming "expert" proficiency in a skill but having 0 months of use.
**Condition:** 
- In `skills` array, ANY skill has `proficiency == "expert"` AND `duration_months == 0`.

## Trap 3: Keyword Stuffer
**Concept:** An impossible number of "expert" level skills.
**Condition:** 
- In `skills` array, the count of skills where `proficiency == "expert"` is `>= 10`.

## Trap 4: Title Mismatch
**Concept:** Perfect AI skills but entirely unrelated career trajectory.
**Condition:** 
- `skills` array has strong AI skills (e.g., embeddings, LLM), BUT `profile.current_title` and ALL titles in `career_history` are strictly non-technical (e.g., "Marketing Manager", "Sales Exec"). Check against a whitelist of technical title keywords ("Engineer", "Developer", "Scientist", "Analyst", "CTO").

## Trap 5: Timeline Overlap
**Concept:** Multiple full-time overlapping jobs.
**Condition:** 
- In `career_history`, two or more entries where `end_date` of Job A > `start_date` of Job B by more than 3 months (allowing for transition overlap), AND both are marked as full-time (or just based on date overlap since schema doesn't specify employment type).

## Trap 6: Ghost Skills
**Concept:** Listed skills are completely absent from career descriptions.
**Condition:** 
- (Optional/Soft trap): The candidate lists a highly specific skill (e.g., "Pinecone"), but it does not appear in any `career_history[].description`. *For strict honeypot, we only flag if >50% of advanced/expert skills are ghosts.*

## Trap 7: Behavioral Ghost
**Concept:** Perfect on paper but behaviorally unavailable.
**Condition:** 
- `redrob_signals.last_active_date` is > 180 days ago AND `redrob_signals.recruiter_response_rate` < 0.05.
