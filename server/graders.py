from typing import Dict, Any

# Score bounds: STRICTLY between 0.0 and 1.0
# Min = 0.05 (attempted but nothing correct)
# Max = 0.95 (excellent but always room for improvement)
SCORE_MIN = 0.05
SCORE_MAX = 0.95


def clamp(score: float) -> float:
    """Clamp score to strictly (0.0, 1.0) - never exactly 0.0 or 1.0."""
    return max(SCORE_MIN, min(SCORE_MAX, round(float(score), 4)))


def grade_easy(episode: Dict[str, Any]) -> float:
    """Task Easy: Set the name tag on a cafe. Score 0.05-0.95."""
    tags = episode.get("current_tags", {})
    name = tags.get("name", "").strip()
    if not name:
        return clamp(0.05)
    if len(name) < 3:
        return clamp(0.4)
    return clamp(0.95)


def grade_medium(episode: Dict[str, Any]) -> float:
    """Task Medium: Add opening hours + website to a shop. Score 0.05-0.95."""
    tags = episode.get("current_tags", {})
    score = 0.05
    has_hours = bool(tags.get("opening_hours", "").strip())
    has_website = bool(tags.get("website", "").strip())
    has_phone = bool(tags.get("phone", "").strip())
    has_name = bool(tags.get("name", "").strip())

    if has_name:
        score += 0.15
    if has_hours:
        score += 0.35
    if has_website:
        score += 0.25
    if has_phone:
        score += 0.10

    return clamp(score)


def grade_hard(episode: Dict[str, Any]) -> float:
    """Task Hard: Fully tag a medical facility. Score 0.05-0.95."""
    tags = episode.get("current_tags", {})
    score = 0.05

    required = {
        "name": 0.15,
        "amenity": 0.10,
        "healthcare": 0.15,
        "opening_hours": 0.10,
        "phone": 0.10,
        "website": 0.10,
        "addr:street": 0.10,
        "addr:city": 0.05,
        "emergency": 0.05,
    }

    for field, weight in required.items():
        val = tags.get(field, "").strip()
        if val:
            score += weight

    # Bonus: check amenity value is medical-related
    amenity = tags.get("amenity", "")
    if amenity in ("hospital", "clinic", "doctors", "pharmacy", "dentist"):
        score += 0.05

    return clamp(score)


GRADERS: Dict[str, Any] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
