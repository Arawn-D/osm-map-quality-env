from typing import Dict, Any


def grade_easy(episode: Dict[str, Any]) -> float:
    """Task Easy: Set the name tag on a cafe. Score 0.0-1.0."""
    tags = episode.get("current_tags", {})
    name = tags.get("name", "").strip()
    if not name:
        return 0.0
    if len(name) < 3:
        return 0.5
    return 1.0


def grade_medium(episode: Dict[str, Any]) -> float:
    """Task Medium: Complete 4 address tags. 0.25 per correct tag."""
    tags = episode.get("current_tags", {})
    expected = {
        "addr:street":   "jubilee hills road",
        "addr:city":     "hyderabad",
        "addr:postcode": "500033",
        "addr:country":  "in",
    }
    score = 0.0
    for key, exp_val in expected.items():
        actual = tags.get(key, "").strip().lower()
        if actual == exp_val:
            score += 0.25
        elif actual and exp_val in actual:
            score += 0.15
    return round(min(score, 1.0), 4)


def grade_hard(episode: Dict[str, Any]) -> float:
    """Task Hard: 6 issues each worth 1/6. Partial credit included."""
    tags = episode.get("current_tags", {})
    coords = episode.get("coordinates", {"lat": 99.9, "lon": 78.5})
    weight = 1.0 / 6
    score = 0.0

    name = tags.get("name", "").strip()
    if name.lower() == "yashoda hospital":
        score += weight
    elif "yashoda" in name.lower():
        score += weight * 0.5

    lat = coords.get("lat", 99.9)
    lon = coords.get("lon", 78.5)
    if 17.0 <= lat <= 18.0 and 78.0 <= lon <= 79.0:
        score += weight

    if tags.get("addr:city", "").lower() == "secunderabad":
        score += weight

    if "alexander road" in tags.get("addr:street", "").lower():
        score += weight

    if "yashodahospitals.com" in tags.get("website", ""):
        score += weight

    if episode.get("duplicate_merged", False):
        score += weight

    return round(min(score, 1.0), 4)


GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}


def grade(task_id: str, episode: Dict[str, Any]) -> float:
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task: {task_id}")
    return GRADERS[task_id](episode)
