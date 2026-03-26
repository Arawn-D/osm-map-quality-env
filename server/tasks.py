import copy
from typing import Dict, Any, List

TASKS: Dict[str, Dict[str, Any]] = {

    "task_easy": {
        "id": "task_easy",
        "name": "Missing Name Tag Fix",
        "difficulty": "easy",
        "description": "A cafe in Hyderabad is missing its name tag. Fix it based on context.",
        "max_steps": 10,
        "initial_feature": {
            "id": "node/1234567",
            "type": "node",
            "tags": {
                "amenity": "cafe",
                "cuisine": "indian",
                "addr:city": "Hyderabad",
                "addr:street": "Banjara Hills Road No. 12",
                "opening_hours": "Mo-Su 08:00-22:00",
            },
            "lat": 17.4126,
            "lon": 78.4482,
        },
        "required_fixes": [
            {"type": "set_tag", "key": "name", "any_non_empty_value": True},
        ],
        "total_issues": 1,
    },

    "task_medium": {
        "id": "task_medium",
        "name": "Multi-Field Address Completion",
        "difficulty": "medium",
        "description": "A building has 4 missing address tags. Fix addr:street, addr:city, addr:postcode, addr:country.",
        "max_steps": 20,
        "initial_feature": {
            "id": "way/9876543",
            "type": "way",
            "tags": {
                "building": "residential",
                "levels": "4",
                "addr:housenumber": "42",
            },
            "lat": 17.3850,
            "lon": 78.4867,
        },
        "required_fixes": [
            {"type": "set_tag", "key": "addr:street",   "expected": "Jubilee Hills Road"},
            {"type": "set_tag", "key": "addr:city",     "expected": "Hyderabad"},
            {"type": "set_tag", "key": "addr:postcode", "expected": "500033"},
            {"type": "set_tag", "key": "addr:country",  "expected": "IN"},
        ],
        "total_issues": 4,
    },

    "task_hard": {
        "id": "task_hard",
        "name": "Duplicate and Conflicting Feature Resolution",
        "difficulty": "hard",
        "description": "Two near-duplicate hospital features with conflicting tags and invalid coordinates. Merge and fix all issues.",
        "max_steps": 30,
        "initial_feature": {
            "id": "node/1111111",
            "type": "node",
            "tags": {
                "amenity": "hospital",
                "name": "Yashoda Hospitl",
                "phone": "040-2345678",
                "addr:city": "Hyderabad",
            },
            "lat": 99.9999,
            "lon": 78.5011,
        },
        "secondary_feature": {
            "id": "node/2222222",
            "type": "node",
            "tags": {
                "amenity": "hospital",
                "name": "Yashoda Hospital",
                "addr:street": "Alexander Road",
                "addr:city": "Secunderabad",
                "website": "https://yashodahospitals.com",
            },
            "lat": 17.4449,
            "lon": 78.5011,
        },
        "required_fixes": [
            {"type": "set_tag",         "key": "name",        "expected": "Yashoda Hospital"},
            {"type": "fix_coordinates", "lat_range": [17.0, 18.0], "lon_range": [78.0, 79.0]},
            {"type": "set_tag",         "key": "addr:city",   "expected": "Secunderabad"},
            {"type": "set_tag",         "key": "addr:street",  "expected": "Alexander Road"},
            {"type": "set_tag",         "key": "website",      "expected": "https://yashodahospitals.com"},
            {"type": "merge_duplicate"},
        ],
        "total_issues": 6,
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}")
    return copy.deepcopy(TASKS[task_id])


def list_tasks() -> List[Dict[str, Any]]:
    action_schema = {
        "action_type": {"type": "string", "enum": ["set_tag", "remove_tag", "fix_coordinates", "merge_duplicate", "flag_invalid", "mark_complete"]},
        "tag_key":     {"type": "string", "required": False},
        "tag_value":   {"type": "string", "required": False},
        "coordinates": {"type": "object", "keys": ["lat", "lon"], "required": False},
        "confidence":  {"type": "number", "min": 0.0, "max": 1.0},
    }
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
            "action_schema": action_schema,
        }
        for t in TASKS.values()
    ]
