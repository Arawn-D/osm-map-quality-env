"""Pydantic models for the OSM Map Quality Environment.
Spec requirement: typed Observation, Action, and Reward Pydantic models.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class MapAction(BaseModel):
    """Agent action on an OSM map feature."""
    action_type: str = Field(default="mark_complete", description="Type of action")
    tag_key: Optional[str] = Field(default=None, description="Tag key (for set_tag / remove_tag)")
    tag_value: Optional[str] = Field(default=None, description="Tag value (for set_tag)")
    coordinates: Optional[Dict[str, float]] = Field(default=None, description="lat/lon for fix_coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Agent confidence 0-1")

    class Config:
        extra = "allow"


class MapObservation(BaseModel):
    """Observation returned from the environment after each step."""
    feature_id: str = Field(default="", description="Unique feature identifier")
    feature_type: str = Field(default="", description="OSM feature type (e.g. amenity)")
    current_tags: Dict[str, Any] = Field(default_factory=dict, description="Current OSM tags")
    secondary_feature: Optional[Dict[str, Any]] = Field(default=None, description="Related feature (e.g. duplicate)")
    issues_remaining: int = Field(default=0, description="Number of unresolved issues")
    feedback: str = Field(default="", description="Human-readable feedback on last action")
    reward: float = Field(default=0.0, description="Reward for last action")
    done: bool = Field(default=False, description="Whether the episode is complete")
    task_id: str = Field(default="", description="Current task identifier")
    step_count: int = Field(default=0, description="Steps taken so far")


class MapReward(BaseModel):
    """Reward signal returned after grading."""
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Final grade score 0.0-1.0")
    task_id: str = Field(default="", description="Task that was graded")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Per-criterion score breakdown")
    success: bool = Field(default=False, description="Whether the agent passed the success threshold")


class MapState(BaseModel):
    """Full environment state (internal)."""
    task_id: str = Field(default="")
    feature_id: str = Field(default="")
    feature_type: str = Field(default="")
    current_tags: Dict[str, Any] = Field(default_factory=dict)
    secondary_feature: Optional[Dict[str, Any]] = Field(default=None)
    issues: List[str] = Field(default_factory=list)
    step_count: int = Field(default=0)
    max_steps: int = Field(default=30)
    done: bool = Field(default=False)
    coordinates: Dict[str, float] = Field(default_factory=lambda: {"lat": 0.0, "lon": 0.0})
    duplicate_merged: bool = Field(default=False)
    episode_actions: List[Dict[str, Any]] = Field(default_factory=list)
