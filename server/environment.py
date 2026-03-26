import uuid
from typing import Optional, Dict, Any

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass

from .tasks import get_task
from .graders import grade

# Import models - handle both package and standalone usage
try:
    from ..models import MapAction, MapObservation, MapState
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import MapAction, MapObservation, MapState


class OSMMapQualityEnvironment(Environment):
    """Real-world OSM map quality checking environment."""

    def __init__(self):
        super().__init__()
        self._state = MapState()
        self._current_task: Optional[Dict[str, Any]] = None
        self._current_feature: Optional[Dict[str, Any]] = None
        self._secondary_feature: Optional[Dict[str, Any]] = None
        self._remaining_fixes: list = []
        self._duplicate_merged: bool = False
        self._task_id: str = "task_easy"

    def reset(self, task_id: str = None) -> MapObservation:
        if task_id is None:
            task_id = self._task_id
        task = get_task(task_id)
        self._current_task = task
        self._task_id = task_id
        self._current_feature = dict(task["initial_feature"])
        self._current_feature["tags"] = dict(task["initial_feature"]["tags"])
        self._secondary_feature = task.get("secondary_feature")
        self._remaining_fixes = list(task["required_fixes"])
        self._duplicate_merged = False
        self._state = MapState(
            task_id=task_id,
            step_count=0,
            episode_id=str(uuid.uuid4()),
            accumulated_reward=0.0,
            issues_fixed=0,
            issues_total=task["total_issues"],
            last_action_type="",
            is_done=False,
        )
        return MapObservation(
            feature_id=self._current_feature["id"],
            feature_type=self._current_feature["type"],
            current_tags=dict(self._current_feature["tags"]),
            issues_remaining=len(self._remaining_fixes),
            feedback=f"Episode started. Task: {task['name']}. Fix {task['total_issues']} issue(s).",
            reward=0.0,
            done=False,
            task_id=task_id,
            step_count=0,
            secondary_feature=self._secondary_feature,
        )

    def step(self, action: MapAction) -> MapObservation:
        if self._state.is_done:
            return self._terminal_obs("Episode done. Call reset().")
        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        reward, feedback = self._apply_action(action)
        self._state.accumulated_reward += reward
        max_steps = self._current_task["max_steps"]
        episode_complete = (len(self._remaining_fixes) == 0 or self._state.step_count >= max_steps)
        if episode_complete:
            self._state.is_done = True
            if len(self._remaining_fixes) == 0:
                reward += 0.2
                self._state.accumulated_reward += 0.2
                feedback += " All issues resolved! Bonus +0.20."
        return MapObservation(
            feature_id=self._current_feature["id"],
            feature_type=self._current_feature["type"],
            current_tags=dict(self._current_feature["tags"]),
            issues_remaining=len(self._remaining_fixes),
            feedback=feedback,
            reward=reward,
            done=self._state.is_done,
            task_id=self._task_id,
            step_count=self._state.step_count,
            secondary_feature=self._secondary_feature,
        )

    @property
    def state(self) -> MapState:
        return self._state

    def get_episode_snapshot(self) -> Dict[str, Any]:
        return {
            "current_tags": dict(self._current_feature.get("tags", {})),
            "coordinates": {"lat": self._current_feature.get("lat", 0.0), "lon": self._current_feature.get("lon", 0.0)},
            "duplicate_merged": self._duplicate_merged,
        }

    def _apply_action(self, action: MapAction):
        tags = self._current_feature["tags"]
        if action.action_type == "set_tag":
            if not action.tag_key:
                return -0.05, "set_tag requires tag_key."
            tags[action.tag_key] = action.tag_value or ""
            return self._check_fix_set_tag(action.tag_key, action.tag_value or "")
        elif action.action_type == "remove_tag":
            if action.tag_key and action.tag_key in tags:
                del tags[action.tag_key]
                return 0.05, f"Removed tag '{action.tag_key}'."
            return -0.05, f"Tag '{action.tag_key}' not found."
        elif action.action_type == "fix_coordinates":
            if not action.coordinates:
                return -0.05, "fix_coordinates requires coordinates with lat/lon."
            lat = action.coordinates.get("lat", 0)
            lon = action.coordinates.get("lon", 0)
            self._current_feature["lat"] = lat
            self._current_feature["lon"] = lon
            return self._check_fix_coordinates(lat, lon)
        elif action.action_type == "merge_duplicate":
            if self._secondary_feature:
                for k, v in self._secondary_feature["tags"].items():
                    if k not in tags:
                        tags[k] = v
                self._duplicate_merged = True
                self._remaining_fixes = [f for f in self._remaining_fixes if f.get("type") != "merge_duplicate"]
                self._state.issues_fixed += 1
                return 0.2, "Merged duplicate feature. Tags inherited."
            return -0.05, "No duplicate feature to merge."
        elif action.action_type == "flag_invalid":
            return 0.02, "Feature flagged for review."
        elif action.action_type == "mark_complete":
            if len(self._remaining_fixes) == 0:
                self._state.is_done = True
                return 0.1, "Correct! Marking complete."
            return -0.1, f"Premature. {len(self._remaining_fixes)} issue(s) remain."
        return -0.1, f"Unknown action_type: '{action.action_type}'."

    def _check_fix_set_tag(self, key, value):
        fixed = False
        new_remaining = []
        for fix in self._remaining_fixes:
            if fix["type"] != "set_tag" or fix["key"] != key:
                new_remaining.append(fix)
                continue
            if fix.get("any_non_empty_value") and value.strip():
                fixed = True
            elif fix.get("expected") and value.strip().lower() == fix["expected"].lower():
                fixed = True
            else:
                new_remaining.append(fix)
        if fixed:
            self._remaining_fixes = new_remaining
            self._state.issues_fixed += 1
            return 0.3, f"Correct! Tag '{key}'='{value}' resolves an issue. +0.30"
        return 0.05, f"Tag '{key}'='{value}' set (no matching fix)."

    def _check_fix_coordinates(self, lat, lon):
        fixed = False
        new_remaining = []
        for fix in self._remaining_fixes:
            if fix["type"] != "fix_coordinates":
                new_remaining.append(fix)
                continue
            if fix["lat_range"][0] <= lat <= fix["lat_range"][1] and fix["lon_range"][0] <= lon <= fix["lon_range"][1]:
                fixed = True
            else:
                new_remaining.append(fix)
        if fixed:
            self._remaining_fixes = new_remaining
            self._state.issues_fixed += 1
            return 0.3, f"Valid coordinates ({lat},{lon}) accepted. +0.30"
        return -0.1, f"Coordinates ({lat},{lon}) out of expected range."

    def _terminal_obs(self, feedback):
        return MapObservation(
            feature_id=self._current_feature.get("id", ""),
            feature_type=self._current_feature.get("type", ""),
            current_tags=dict(self._current_feature.get("tags", {})),
            issues_remaining=len(self._remaining_fixes),
            feedback=feedback,
            reward=0.0,
            done=True,
            task_id=self._task_id,
            step_count=self._state.step_count,
        )
