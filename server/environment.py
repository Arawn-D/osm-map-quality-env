import uuid
from typing import Optional, Dict, Any
from .tasks import get_task
from .graders import grade
from models import MapAction, MapObservation, MapState


class OSMMapQualityEnvironment:
    def __init__(self):
        self._state = MapState()
        self._current_task = None
        self._current_feature = None
        self._remaining_fixes = []
        self._duplicate_merged = False
        self._task_id = "task_easy"

    def reset(self, task_id: str = None) -> MapObservation:
        if task_id is None:
            task_id = self._task_id
        task = get_task(task_id)
        self._current_task = task
        self._current_feature = dict(task["initial_feature"])
        self._current_feature["tags"] = dict(task["initial_feature"]["tags"])
        self._remaining_fixes = list(task["required_fixes"])
        self._duplicate_merged = False
        self._state = MapState(task_id=task_id, episode_id=str(uuid.uuid4()))
        return MapObservation(
            feature_id=self._current_feature["id"],
            current_tags=self._current_feature["tags"],
        )

    def step(self, action: MapAction) -> MapObservation:
        self._state.step_count += 1
        reward, feedback = self._apply_action(action)
        if len(self._remaining_fixes) == 0:
            self._state.is_done = True
        return MapObservation(
            feature_id=self._current_feature["id"],
            current_tags=self._current_feature["tags"],
            reward=reward,
            done=self._state.is_done,
        )

    @property
    def state(self):
        return self._state

    def get_episode_snapshot(self):
        return {
            "current_tags": self._current_feature["tags"],
            "duplicate_merged": self._duplicate_merged,
        }

    def _apply_action(self, action):
        tags = self._current_feature["tags"]
        if action.action_type == "set_tag":
            tags[action.tag_key] = action.tag_value
            return 0.3, "Tag set"
        elif action.action_type == "remove_tag":
            if action.tag_key in tags:
                del tags[action.tag_key]
            return 0.1, "Tag removed"
        return 0.0, "Unknown action"
