from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Use plain base classes to avoid pydantic/dataclass conflicts.
# The openenv framework injects its own base classes at runtime in production.
class Action:
    pass

class Observation:
    pass

class State:
    pass


@dataclass
class MapAction(Action):
    """Agent action on an OSM map feature."""
    action_type: str = ""
    tag_key: Optional[str] = None
    tag_value: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    confidence: float = 1.0

@dataclass
class MapObservation(Observation):
    """What the agent observes after each step."""
    feature_id: str = ""
    feature_type: str = ""
    current_tags: Dict[str, str] = field(default_factory=dict)
    issues_remaining: int = 0
    feedback: str = ""
    reward: float = 0.0
    done: bool = False
    task_id: str = ""
    step_count: int = 0
    secondary_feature: Optional[Dict[str, Any]] = None

@dataclass
class MapState(State):
    """Full episode state tracking."""
    task_id: str = ""
    step_count: int = 0
    episode_id: str = ""
    accumulated_reward: float = 0.0
    issues_fixed: int = 0
    issues_total: int = 0
    last_action_type: str = ""
    is_done: bool = False