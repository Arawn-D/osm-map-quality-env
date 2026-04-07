from typing import Dict, Any

SCORE_MIN = 0.05
SCORE_MAX = 0.95


def clamp(score: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, round(float(score), 4)))


def grade_easy(episode: Dict[str, Any]) -> float:
    """Score based on whether the name tag is set on a cafe feature."""
    tags = episode.get("current_tags", {})
    name = tags.get("name", "").strip()
    if not name:
        return clamp(0.05)
    if len(name) < 3:
        return clamp(0.4)
    return clamp(0.95)


def grade_medium(episode: Dict[str, Any]) -> float:
    """Score based on how many address fields were correctly filled."""
    tags = episode.get("current_tags", {})
    score = 0.05
    fields = {
        "addr:street": 0.25,
        "addr:city": 0.25,
        "addr:postcode": 0.25,
        "addr:country": 0.15,
    }
    for key, weight in fields.items():
        if tags.get(key, "").strip():
            score += weight
    return clamp(score)


def grade_hard(episode: Dict[str, Any]) -> float:
    """Score based on tag completeness and coordinate fix for a hospital feature."""
    tags = episode.get("current_tags", {})
    coords = episode.get("coordinates", {})
    merged = episode.get("duplicate_merged", False)

    score = 0.05

    tag_weights = {
        "name": 0.15,
        "amenity": 0.10,
        "addr:city": 0.10,
        "addr:street": 0.10,
        "website": 0.10,
        "phone": 0.05,
    }
    for field, weight in tag_weights.items():
        if tags.get(field, "").strip():
            score += weight

    lat = coords.get("lat", 0.0)
    lon = coords.get("lon", 0.0)
    if 17.0 <= lat <= 18.0 and 78.0 <= lon <= 79.0:
        score += 0.15

    if merged:
        score += 0.15

    return clamp(score)


GRADERS: Dict[str, Any] = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}


def grade(task_id: str, episode: Dict[str, Any]) -> float:
    """Dispatch grading to the correct task grader."""
    grader_fn = GRADERS.get(task_id)
    if grader_fn is None:
        raise ValueError(f"No grader registered for task_id: {task_id!r}")
    return grader_fn(episode)
