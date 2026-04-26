"""Multi-dimensional grading for OSM Map Quality Environment.

Scoring dimensions:
  1. Tag completeness  — are required tags present with correct values?
  2. Data consistency   — do tags agree with coordinates/context?
  3. Action efficiency  — did the agent use minimal steps?
  4. Coordinate accuracy — are coordinates valid and in the right region?
  5. Merge completion   — was the duplicate properly merged?
"""
from typing import Dict, Any

SCORE_MIN = 0.05
SCORE_MAX = 0.95


def clamp(score: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, round(float(score), 4)))


def grade_easy(episode: Dict[str, Any]) -> float:
    """Score based on whether the name tag is set on a POI feature.
    
    Dimensions:
      - Name presence (0.70 weight)
      - Name quality: len >= 3 (0.20 weight)
      - Efficiency bonus (0.05 weight)
    """
    tags = episode.get("current_tags", {})
    name = tags.get("name", "").strip()
    
    score = 0.05
    
    # Dim 1: Name exists
    if not name:
        return clamp(0.05)
    score += 0.50
    
    # Dim 2: Name quality (not just "x" or "ab")
    if len(name) >= 3:
        score += 0.30
    elif len(name) >= 1:
        score += 0.10
    
    # Dim 3: Efficiency — fewer steps = small bonus
    steps = episode.get("steps_taken", 10)
    if steps <= 2:
        score += 0.10
    elif steps <= 5:
        score += 0.05
    
    return clamp(score)


def grade_medium(episode: Dict[str, Any]) -> float:
    """Score based on address field completeness and data consistency.
    
    Dimensions:
      - Field completeness (0.70 weight)
      - Consistency check (0.15 weight)
      - Efficiency bonus (0.10 weight)
    """
    tags = episode.get("current_tags", {})
    
    score = 0.05
    
    # Dim 1: Required address fields
    fields = {
        "addr:street":   0.20,
        "addr:city":     0.20,
        "addr:postcode": 0.17,
        "addr:country":  0.13,
    }
    for key, weight in fields.items():
        val = tags.get(key, "").strip()
        if val and val not in ("WRONG_DATA", ""):
            score += weight
    
    # Dim 2: Data consistency — city + postcode should match known pairs
    city = tags.get("addr:city", "").strip()
    postcode = tags.get("addr:postcode", "").strip()
    known_pairs = {
        "Hyderabad": ["500033", "500034", "500082"],
        "Secunderabad": ["500003", "500025"],
    }
    if city in known_pairs and postcode in known_pairs[city]:
        score += 0.10
    elif city and postcode:
        score += 0.05  # at least they tried
    
    # Dim 3: Efficiency
    steps = episode.get("steps_taken", 20)
    if steps <= 5:
        score += 0.10
    elif steps <= 10:
        score += 0.05
    
    return clamp(score)


def grade_hard(episode: Dict[str, Any]) -> float:
    """Multi-dimensional scoring for the hardest task tier.
    
    Dimensions:
      - Tag completeness (0.45 weight)
      - Coordinate validity (0.15 weight)
      - Duplicate merge (0.10 weight)
      - Data consistency (0.10 weight)
      - Action efficiency (0.10 weight)
      - Sequence quality (0.05 weight)
    """
    tags = episode.get("current_tags", {})
    coords = episode.get("coordinates", {})
    merged = episode.get("duplicate_merged", False)
    steps = episode.get("steps_taken", 30)
    actions = episode.get("actions_history", [])
    
    score = 0.05
    
    # Dim 1: Tag completeness (0.45)
    tag_weights = {
        "name":        0.12,
        "amenity":     0.05,
        "addr:city":   0.08,
        "addr:street": 0.08,
        "website":     0.07,
        "phone":       0.05,
    }
    for field, weight in tag_weights.items():
        val = tags.get(field, "").strip()
        if val:
            score += weight
    
    # Dim 2: Coordinate validity (0.15)
    lat = coords.get("lat", 0.0)
    lon = coords.get("lon", 0.0)
    if 17.0 <= lat <= 18.0 and 78.0 <= lon <= 79.0:
        score += 0.15
    elif -90 <= lat <= 90 and -180 <= lon <= 180:
        score += 0.05  # at least valid GPS
    
    # Dim 3: Duplicate merged (0.10)
    if merged:
        score += 0.10
    
    # Dim 4: Data consistency (0.10)
    # Check if tags are internally consistent
    city = tags.get("addr:city", "")
    if 17.43 <= lat <= 17.50 and "Secunderabad" in city:
        score += 0.10  # coords + city agree
    elif 17.35 <= lat <= 17.43 and "Hyderabad" in city:
        score += 0.10
    elif city:
        score += 0.03  # city present but may not match coords
    
    # Dim 5: Action efficiency (0.10)
    if steps <= 7:  # optimal is 6-7 steps
        score += 0.10
    elif steps <= 12:
        score += 0.06
    elif steps <= 20:
        score += 0.03
    
    # Dim 6: Sequence quality (0.05)
    # Did the agent fix coordinates BEFORE setting address?
    if actions:
        try:
            coord_idx = actions.index("fix_coordinates") if "fix_coordinates" in actions else 999
            set_idx = next((i for i, a in enumerate(actions)
                          if a == "set_tag"), 999)
            if coord_idx < set_idx:
                score += 0.05  # correct order: fix coords first
            elif coord_idx < 999:
                score += 0.02  # at least they fixed coords
        except (ValueError, StopIteration):
            pass
    
    return clamp(score)


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

GRADERS: Dict[str, Any] = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}


def grade(task_id: str, episode: Dict[str, Any]) -> float:
    """Dispatch grading to the correct task grader."""
    grader_fn = GRADERS.get(task_id)
    if grader_fn is None:
        raise ValueError(f"No grader registered for task_id: {task_id!r}")
    return grader_fn(episode)