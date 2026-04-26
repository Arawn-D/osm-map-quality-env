"""Dynamic task generation with noise injection for OSM Map Quality Environment.

Each /reset call generates a fresh task variation with:
  - Randomized POI types, street names, and missing fields
  - Noise injection: typos, conflicting values, stale data
  - Consistent required_fixes contract for grading

Original 3 task IDs (task_easy, task_medium, task_hard) are preserved.
Backward-compatible: same API contract, same grader expectations.
"""
import copy
import random
import time
from typing import Dict, Any, List

# ═══════════════════════════════════════════════════════════════════
# Noise injection utilities — make data messy (like real OSM)
# ═══════════════════════════════════════════════════════════════════

def _inject_typo(value: str) -> str:
    """Introduce a realistic typo into a string."""
    if len(value) < 3:
        return value
    pos = random.randint(1, len(value) - 2)
    mutations = [
        lambda v, p: v[:p] + v[p+1] + v[p] + v[p+2:],  # swap adjacent
        lambda v, p: v[:p] + v[p+2:],                     # delete char
        lambda v, p: v[:p] + v[p] + v[p] + v[p+1:],      # duplicate char
    ]
    try:
        return random.choice(mutations)(value, pos)
    except (IndexError, TypeError):
        return value


def _inject_conflict(value: str, alternatives: list) -> str:
    """Replace value with a plausible but wrong alternative."""
    alts = [a for a in alternatives if a != value]
    return random.choice(alts) if alts else value


def _inject_stale(value: str) -> str:
    """Mark data as outdated."""
    prefixes = ["UNVERIFIED: ", "OLD: ", "~", ""]
    suffix = random.choice(["", " (2019)", " [unconfirmed]", ""])
    return random.choice(prefixes) + value + suffix


NOISE_INJECTORS = {
    "typo":     _inject_typo,
    "stale":    _inject_stale,
    "missing":  lambda v: "",
}


# ═══════════════════════════════════════════════════════════════════
# Base task data pools
# ═══════════════════════════════════════════════════════════════════

EASY_POIS = [
    {"amenity": "cafe",       "name_hint": "Hyderabad Chai Point"},
    {"amenity": "pharmacy",   "name_hint": "MedPlus Pharmacy"},
    {"amenity": "school",     "name_hint": "Delhi Public School"},
    {"amenity": "restaurant", "name_hint": "Biryani House"},
    {"amenity": "library",    "name_hint": "Central Library"},
    {"amenity": "bank",       "name_hint": "State Bank Branch"},
    {"amenity": "hospital",   "name_hint": "Apollo Clinic"},
    {"amenity": "clinic",     "name_hint": "City Health Center"},
]

HYDERABAD_STREETS = [
    "Banjara Hills Road No. 12",
    "Jubilee Hills Road",
    "Madhapur Main Road",
    "Kukatpally Housing Board Road",
    "Gachibowli Flyover Road",
    "Ameerpet Circle Road",
    "Begumpet Station Road",
    "Secunderabad Clock Tower Road",
]

INDIA_CITIES = [
    {"city": "Hyderabad",    "postcode": "500034", "state": "Telangana"},
    {"city": "Secunderabad", "postcode": "500003", "state": "Telangana"},
    {"city": "Hyderabad",    "postcode": "500033", "state": "Telangana"},
    {"city": "Hyderabad",    "postcode": "500082", "state": "Telangana"},
]

HOSPITAL_NAMES = [
    {"correct": "Yashoda Hospital",    "typo": "Yashoda Hospitl"},
    {"correct": "Apollo Hospital",     "typo": "Appollo Hospital"},
    {"correct": "KIMS Hospital",       "typo": "KIMS Hosptial"},
    {"correct": "Care Hospital",       "typo": "Caer Hospital"},
]

HOSPITAL_WEBSITES = [
    "https://yashodahospitals.com",
    "https://apollohospitals.com",
    "https://kimshospitals.com",
    "https://carehospitals.com",
]


# ═══════════════════════════════════════════════════════════════════
# Dynamic task generators
# ═══════════════════════════════════════════════════════════════════

def _generate_easy(seed: int) -> Dict[str, Any]:
    """Generate an easy task: missing name tag on a POI."""
    rng = random.Random(seed)
    poi = rng.choice(EASY_POIS)
    street = rng.choice(HYDERABAD_STREETS)
    node_id = f"node/{7000000 + seed}"

    # Base tags — always missing "name"
    tags = {
        "amenity": poi["amenity"],
        "addr:city": "Hyderabad",
        "addr:street": street,
    }

    # Random extra tags for realism
    if rng.random() > 0.5:
        tags["opening_hours"] = rng.choice([
            "Mo-Su 08:00-22:00", "Mo-Fr 09:00-18:00",
            "Mo-Sa 10:00-21:00", "24/7",
        ])
    if rng.random() > 0.6:
        tags["cuisine"] = rng.choice(["indian", "chinese", "south_indian", "multi"])
    if rng.random() > 0.7:
        tags["phone"] = f"+91-40-{rng.randint(2000000, 9999999)}"

    # Noise: occasionally inject a typo in addr:city
    if rng.random() > 0.7:
        tags["addr:city"] = _inject_typo("Hyderabad")

    return {
        "id": "task_easy",
        "name": "Missing Name Tag Fix",
        "difficulty": "easy",
        "description": f"A {poi['amenity']} in Hyderabad is missing its name tag. "
                       f"Inspect the feature and add an appropriate name.",
        "max_steps": 10,
        "initial_feature": {
            "id": node_id,
            "type": "node",
            "tags": tags,
            "lat": round(17.38 + rng.uniform(0, 0.08), 4),
            "lon": round(78.40 + rng.uniform(0, 0.10), 4),
        },
        "required_fixes": [
            {"type": "set_tag", "key": "name", "any_non_empty_value": True},
        ],
        "total_issues": 1,
        "initially_visible_tags": ["amenity", "addr:city"],
        "_seed": seed,
    }


def _generate_medium(seed: int) -> Dict[str, Any]:
    """Generate a medium task: multiple missing address fields."""
    rng = random.Random(seed)
    geo = rng.choice(INDIA_CITIES)
    node_id = f"way/{8000000 + seed}"

    # Required fields — always need these 4
    expected = {
        "addr:street": rng.choice(HYDERABAD_STREETS),
        "addr:city": geo["city"],
        "addr:postcode": geo["postcode"],
        "addr:country": "IN",
    }

    # Base tags — deliberately incomplete
    tags = {
        "building": rng.choice(["residential", "commercial", "yes", "apartments"]),
        "levels": str(rng.randint(1, 12)),
        "addr:housenumber": str(rng.randint(1, 200)),
    }

    # Noise: sometimes include a WRONG value that needs overwriting
    if rng.random() > 0.6:
        wrong_field = rng.choice(list(expected.keys()))
        cities = ["Mumbai", "Chennai", "Bangalore", "Delhi"]
        tags[wrong_field] = rng.choice(cities) if wrong_field == "addr:city" else "WRONG_DATA"

    required_fixes = [
        {"type": "set_tag", "key": k, "expected": v}
        for k, v in expected.items()
    ]

    return {
        "id": "task_medium",
        "name": "Multi-Field Address Completion",
        "difficulty": "medium",
        "description": f"A {tags['building']} building has {len(required_fixes)} missing or incorrect "
                       f"address fields. Complete the address data.",
        "max_steps": 20,
        "initial_feature": {
            "id": node_id,
            "type": "way",
            "tags": tags,
            "lat": round(17.35 + rng.uniform(0, 0.10), 4),
            "lon": round(78.40 + rng.uniform(0, 0.12), 4),
        },
        "required_fixes": required_fixes,
        "total_issues": len(required_fixes),
        "initially_visible_tags": ["building", "levels", "addr:housenumber"],
        "_seed": seed,
    }


def _generate_hard(seed: int) -> Dict[str, Any]:
    """Generate a hard task: conflicting duplicates + bad coordinates."""
    rng = random.Random(seed)
    hospital = rng.choice(HOSPITAL_NAMES)
    website = rng.choice(HOSPITAL_WEBSITES)
    street = rng.choice(HYDERABAD_STREETS)
    node_id = f"node/{9000000 + seed}"

    # Invalid coordinates — randomized bad values
    bad_lat = rng.choice([99.9999, -99.0, 0.0, 200.0, rng.uniform(50, 89)])
    good_lat = round(17.40 + rng.uniform(0, 0.10), 4)
    good_lon = round(78.45 + rng.uniform(0, 0.10), 4)

    # Primary feature — has errors
    primary_tags = {
        "amenity": "hospital",
        "name": hospital["typo"],  # misspelled
        "phone": f"040-{rng.randint(2000000, 9999999)}",
        "addr:city": "Hyderabad",  # wrong — should match secondary
    }

    # Noise: randomly add extra misleading tags
    if rng.random() > 0.5:
        primary_tags["old_name"] = rng.choice(["City Hospital", "General Hospital", ""])
    if rng.random() > 0.6:
        primary_tags["source"] = rng.choice(["survey", "bing", "local_knowledge"])

    # Secondary (duplicate) feature — has correct data
    secondary_city = rng.choice(["Secunderabad", "Hyderabad"])
    secondary_tags = {
        "amenity": "hospital",
        "name": hospital["correct"],
        "addr:street": street,
        "addr:city": secondary_city,
        "website": website,
    }

    required_fixes = [
        {"type": "set_tag", "key": "name",        "expected": hospital["correct"]},
        {"type": "fix_coordinates", "lat_range": [17.0, 18.0], "lon_range": [78.0, 79.0]},
        {"type": "set_tag", "key": "addr:city",    "expected": secondary_city},
        {"type": "set_tag", "key": "addr:street",  "expected": street},
        {"type": "set_tag", "key": "website",      "expected": website},
        {"type": "merge_duplicate"},
    ]

    return {
        "id": "task_hard",
        "name": "Duplicate & Conflicting Feature Resolution",
        "difficulty": "hard",
        "description": "Two near-duplicate hospital features with conflicting tags, "
                       "invalid coordinates, and data quality issues. Resolve all conflicts.",
        "max_steps": 30,
        "initial_feature": {
            "id": node_id,
            "type": "node",
            "tags": primary_tags,
            "lat": bad_lat,
            "lon": good_lon,
        },
        "secondary_feature": {
            "id": f"node/{9500000 + seed}",
            "type": "node",
            "tags": secondary_tags,
            "lat": good_lat,
            "lon": good_lon,
        },
        "required_fixes": required_fixes,
        "total_issues": 6,
        "initially_visible_tags": ["amenity", "name", "addr:city"],
        "_seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════
# Public API — backward-compatible
# ═══════════════════════════════════════════════════════════════════

GENERATORS = {
    "task_easy":   _generate_easy,
    "task_medium": _generate_medium,
    "task_hard":   _generate_hard,
}


def get_task(task_id: str, seed: int = None) -> Dict[str, Any]:
    """Get a task by ID with optional seed for reproducibility.
    
    If seed is None, generates a fresh random variation each call.
    """
    if task_id not in GENERATORS:
        raise ValueError(f"Unknown task: {task_id}. Available: {sorted(GENERATORS.keys())}")
    if seed is None:
        seed = int(time.time() * 1000) % 1_000_000
    return GENERATORS[task_id](seed)


def list_tasks() -> List[Dict[str, Any]]:
    """List available tasks with action schema (for /tasks endpoint)."""
    action_schema = {
        "action_type": {
            "type": "string",
            "enum": ["set_tag", "remove_tag", "fix_coordinates",
                     "merge_duplicate", "flag_invalid", "mark_complete"],
        },
        "tag_key":     {"type": "string", "required": False},
        "tag_value":   {"type": "string", "required": False},
        "coordinates": {"type": "object", "keys": ["lat", "lon"], "required": False},
        "confidence":  {"type": "number", "min": 0.0, "max": 1.0},
    }

    task_info = [
        {
            "id": "task_easy",
            "name": "Missing Name Tag Fix",
            "difficulty": "easy",
            "description": "A POI is missing its name tag. Inspect and fix it.",
            "max_steps": 10,
            "features": ["noise_injection", "dynamic_generation"],
        },
        {
            "id": "task_medium",
            "name": "Multi-Field Address Completion",
            "difficulty": "medium",
            "description": "A building has multiple missing or incorrect address fields.",
            "max_steps": 20,
            "features": ["conflicting_data", "noise_injection", "dynamic_generation"],
        },
        {
            "id": "task_hard",
            "name": "Duplicate & Conflicting Feature Resolution",
            "difficulty": "hard",
            "description": "Near-duplicate features with conflicting tags, invalid coordinates, "
                           "and data quality issues. Requires multi-step reasoning.",
            "max_steps": 30,
            "features": ["partial_observability", "cascading_errors", "conflicting_data",
                         "noise_injection", "dynamic_generation"],
        },
    ]

    return [
        {**info, "action_schema": action_schema}
        for info in task_info
    ]