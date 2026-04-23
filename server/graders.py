from typing import Dict, Any

SCORE_MIN = 0.0
SCORE_MAX = 1.0


def clamp(score: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, round(float(score), 4)))


def grade_easy(episode: Dict[str, Any]) -> float:
    """Score based on whether the name tag is set on a cafe feature."""
    tags = episode.get("current_tags", {})
    name = tags.get("name", "").strip()
    if not name:
        return clamp(0.0)
    if len(name) < 3:
        return clamp(0.4)
    return clamp(1.0)


def grade_medium(episode: Dict[str, Any]) -> float:
    """Score based on how many address fields were correctly filled."""
    tags = episode.get("current_tags", {})
    fields = {
        "addr:street": 0.25,
        "addr:city": 0.25,
        "addr:postcode": 0.25,
        "addr:country": 0.25,
    }
    score = 0.0
    for field, weight in fields.items():
        if tags.get(field, "").strip():
            score += weight
    return clamp(score)


def grade_hard(episode: Dict[str, Any]) -> float:
    """Score based on multiple criteria for a hospital feature."""
    tags = episode.get("current_tags", {})
    coordinates = episode.get("coordinates", {})
    duplicate_merged = episode.get("duplicate_merged", False)

    score = 0.0

    # Name check (0.2)
    name = tags.get("name", "").strip()
    if name and len(name) >= 3:
        score += 0.2

    # Coordinates check (0.2)
    lat = coordinates.get("lat", 0.0)
    lon = coordinates.get("lon", 0.0)
    if 17.0 <= lat <= 18.0 and 78.0 <= lon <= 79.0:
        score += 0.2

    # Address fields (0.15 each = 0.3 total)
    if tags.get("addr:city", "").strip():
        score += 0.15
    if tags.get("addr:street", "").strip():
        score += 0.15

    # Duplicate merged (0.15)
    if duplicate_merged:
        score += 0.15

    # Website (0.15)
    website = tags.get("website", "").strip()
    if website and website.startswith("http"):
        score += 0.15

    return clamp(score)


def grade(task_id: str, episode: Dict[str, Any]) -> float:
    """Dispatch to the appropriate grader function."""
    graders = {
        "task_easy": grade_easy,
        "task_medium": grade_medium,
        "task_hard": grade_hard,
    }
    grader_fn = graders.get(task_id)
    if grader_fn is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return grader_fn(episode)
