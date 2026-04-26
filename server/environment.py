"""OSM Map Quality Environment with partial observability, cascading issues,
and confidence-calibrated rewards.

Innovations over standard RL environments:
  1. Partial observability — tags revealed progressively, not all at once
  2. Cascading discovery — fixing one issue may reveal new ones
  3. Confidence penalties — overconfident wrong actions cost more
  4. Diagnostic feedback — rich natural-language feedback for agents
"""
import uuid
from typing import Optional, Dict, Any, Set

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass

from .tasks import get_task
from .graders import grade

try:
    from ..models import MapAction, MapObservation, MapState
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import MapAction, MapObservation, MapState


class OSMMapQualityEnvironment(Environment):
    """World-modeling environment for geographic data quality assurance.
    
    Features:
      - Partial observability: tags revealed progressively through actions
      - Cascading errors: fixing coordinates may reveal address conflicts
      - Confidence-calibrated rewards: overconfident wrong answers penalized
      - Diagnostic feedback: detailed reasoning signals for agents
      - Dynamic task generation: fresh variations each episode
    """

    def __init__(self):
        # Do not call super().__init__() - openenv Environment may require args
        self._state = MapState(
            task_id="",
            step_count=0,
            episode_id="",
            accumulated_reward=0.0,
            issues_fixed=0,
            issues_total=0,
            last_action_type="",
            is_done=False,
        )
        self._current_task: Optional[Dict[str, Any]] = None
        self._current_feature: Optional[Dict[str, Any]] = None
        self._secondary_feature: Optional[Dict[str, Any]] = None
        self._remaining_fixes: list = []
        self._duplicate_merged: bool = False
        self._task_id: str = "task_easy"
        # Partial observability state
        self._visible_tags: Set[str] = set()
        # Tracking for grader
        self._actions_history: list = []
        self._confidence_scores: list = []

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
        self._actions_history = []
        self._confidence_scores = []

        # Partial observability: start with limited visibility
        initially_visible = task.get("initially_visible_tags",
                                      list(self._current_feature["tags"].keys()))
        self._visible_tags = set(initially_visible)

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

        obs_tags = self._get_visible_tags()
        hidden_count = len(self._current_feature["tags"]) - len(obs_tags)
        feedback = (
            f"Episode started. Task: {task['name']}. "
            f"Fix {task['total_issues']} issue(s)."
        )
        if hidden_count > 0:
            feedback += f" ℹ {hidden_count} tag(s) not yet visible — take actions to reveal more data."

        return MapObservation(
            feature_id=self._current_feature["id"],
            feature_type=self._current_feature["type"],
            current_tags=obs_tags,
            issues_remaining=len(self._remaining_fixes),
            feedback=feedback,
            reward=0.0,
            done=False,
            task_id=task_id,
            step_count=0,
            secondary_feature=self._secondary_feature,
        )

    def step(self, action) -> MapObservation:
        if self._state.is_done:
            return self._terminal_obs("Episode done. Call reset().")

        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._actions_history.append(action.action_type)
        confidence = getattr(action, "confidence", 1.0)
        self._confidence_scores.append(confidence)

        reward, feedback = self._apply_action(action)

        # Confidence calibration: overconfident wrong actions cost more
        is_positive = reward > 0
        if not is_positive and confidence > 0.7:
            penalty = -0.03 * confidence
            reward += penalty
            feedback += f" ⚠ Overconfidence penalty ({confidence:.1f} conf on wrong action)."

        self._state.accumulated_reward += reward
        max_steps = self._current_task["max_steps"]
        episode_complete = (
            len(self._remaining_fixes) == 0 or
            self._state.step_count >= max_steps
        )

        if episode_complete:
            self._state.is_done = True
            if len(self._remaining_fixes) == 0:
                reward += 0.2
                self._state.accumulated_reward += 0.2
                feedback += " 🎉 All issues resolved! Bonus +0.20."

        # Reveal tags based on action taken
        self._reveal_tags_for_action(action)

        obs_tags = self._get_visible_tags()
        newly_visible = len(obs_tags) - len(self._visible_tags.intersection(
            self._current_feature["tags"].keys()))

        # Add data quality signals to feedback
        quality_hints = self._generate_quality_hints()
        if quality_hints:
            feedback += f" {quality_hints}"

        return MapObservation(
            feature_id=self._current_feature["id"],
            feature_type=self._current_feature["type"],
            current_tags=obs_tags,
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
            "coordinates": {
                "lat": self._current_feature.get("lat", 0.0),
                "lon": self._current_feature.get("lon", 0.0),
            },
            "duplicate_merged": self._duplicate_merged,
            "steps_taken": self._state.step_count,
            "actions_history": list(self._actions_history),
            "avg_confidence": (
                sum(self._confidence_scores) / len(self._confidence_scores)
                if self._confidence_scores else 0.5
            ),
        }

    # ─── Partial Observability ──────────────────────────────────────

    def _get_visible_tags(self) -> Dict[str, str]:
        """Return only tags that are currently visible to the agent."""
        tags = self._current_feature.get("tags", {})
        return {k: v for k, v in tags.items() if k in self._visible_tags}

    def _reveal_tags_for_action(self, action):
        """Reveal related tags based on the action taken."""
        all_tags = set(self._current_feature.get("tags", {}).keys())

        if action.action_type == "set_tag" and action.tag_key:
            # Setting an address field reveals other address fields
            if action.tag_key.startswith("addr:"):
                addr_fields = {k for k in all_tags if k.startswith("addr:")}
                self._visible_tags |= addr_fields
            # Setting any tag reveals the tag itself + related tags
            self._visible_tags.add(action.tag_key)
            # Reveal 1-2 random hidden tags as "discovered" data
            hidden = all_tags - self._visible_tags
            if hidden:
                reveal_count = min(2, len(hidden))
                self._visible_tags |= set(list(hidden)[:reveal_count])

        elif action.action_type == "fix_coordinates":
            # Fixing coordinates reveals geographic context tags
            geo_tags = {k for k in all_tags
                       if k.startswith("addr:") or k in ("phone", "website")}
            self._visible_tags |= geo_tags

        elif action.action_type == "merge_duplicate":
            # Merging reveals ALL tags (from both features)
            self._visible_tags = all_tags.copy()

        elif action.action_type in ("flag_invalid", "mark_complete"):
            # These reveal everything — terminal actions
            self._visible_tags = all_tags.copy()

        elif action.action_type == "remove_tag":
            self._visible_tags.add(action.tag_key)

    # ─── Cascading Issue Discovery ──────────────────────────────────

    def _check_cascading_issues(self, action):
        """After fixing coordinates, check if address is now inconsistent."""
        if action.action_type == "fix_coordinates":
            tags = self._current_feature.get("tags", {})
            lat = self._current_feature.get("lat", 0)
            # If coordinates now place feature in Secunderabad
            # but addr:city says Hyderabad, that's a cascading discovery
            if 17.43 <= lat <= 17.50 and tags.get("addr:city") == "Hyderabad":
                return ("ℹ Cascading discovery: coordinates place this feature "
                        "in Secunderabad district, but addr:city='Hyderabad'. "
                        "Consider updating the address.")
        return ""

    # ─── Data Quality Hints ─────────────────────────────────────────

    def _generate_quality_hints(self) -> str:
        """Generate diagnostic hints about data quality issues."""
        hints = []
        tags = self._current_feature.get("tags", {})

        if len(self._remaining_fixes) > 0:
            # Priority hint: what's the most impactful fix?
            fix_types = [f["type"] for f in self._remaining_fixes]
            if "fix_coordinates" in fix_types:
                lat = self._current_feature.get("lat", 0)
                lon = self._current_feature.get("lon", 0)
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    hints.append(f"⚠ Priority: coordinates ({lat}, {lon}) are INVALID.")
            elif "merge_duplicate" in fix_types:
                hints.append("ℹ Duplicate feature detected — consider merging.")
            elif "set_tag" in fix_types:
                missing_keys = [f.get("key", "?") for f in self._remaining_fixes
                               if f["type"] == "set_tag"]
                if missing_keys:
                    hints.append(f"ℹ Missing/incorrect: {', '.join(missing_keys[:3])}")

        return " ".join(hints)

    # ─── Action Processing ──────────────────────────────────────────

    def _apply_action(self, action):
        tags = self._current_feature["tags"]

        if action.action_type == "set_tag":
            if not action.tag_key:
                return -0.05, "set_tag requires tag_key."
            tags[action.tag_key] = action.tag_value or ""
            self._visible_tags.add(action.tag_key)
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
            reward, feedback = self._check_fix_coordinates(lat, lon)
            # Check for cascading issues
            cascade = self._check_cascading_issues(action)
            if cascade:
                feedback += f" {cascade}"
            return reward, feedback

        elif action.action_type == "merge_duplicate":
            if self._secondary_feature:
                for k, v in self._secondary_feature["tags"].items():
                    if k not in tags:
                        tags[k] = v
                self._duplicate_merged = True
                self._remaining_fixes = [
                    f for f in self._remaining_fixes
                    if f.get("type") != "merge_duplicate"
                ]
                self._state.issues_fixed += 1
                # Reveal all tags after merge
                self._visible_tags = set(tags.keys())
                return 0.2, "Merged duplicate feature. Tags inherited. All data now visible."
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
            return 0.3, f"✓ Tag '{key}'='{value}' resolves an issue. +0.30"
        return 0.05, f"Tag '{key}'='{value}' set (no matching fix required)."

    def _check_fix_coordinates(self, lat, lon):
        fixed = False
        new_remaining = []
        for fix in self._remaining_fixes:
            if fix["type"] != "fix_coordinates":
                new_remaining.append(fix)
                continue
            if (fix["lat_range"][0] <= lat <= fix["lat_range"][1] and
                    fix["lon_range"][0] <= lon <= fix["lon_range"][1]):
                fixed = True
            else:
                new_remaining.append(fix)
        if fixed:
            self._remaining_fixes = new_remaining
            self._state.issues_fixed += 1
            return 0.3, f"✓ Valid coordinates ({lat}, {lon}) accepted. +0.30"
        return -0.1, f"✗ Coordinates ({lat}, {lon}) out of expected range."

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