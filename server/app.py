"""FastAPI application for the OSM Map Quality Environment."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, Field
from typing import Optional, Dict, Any
from dataclasses import asdict
import time
import re
from collections import defaultdict
from .environment import OSMMapQualityEnvironment
from .tasks import list_tasks
from .graders import grade

VALID_TASK_IDS = {"task_easy", "task_medium", "task_hard"}
VALID_ACTION_TYPES = {
    "set_tag", "remove_tag", "fix_coordinates",
    "merge_duplicate", "flag_invalid", "mark_complete",
}


class RateLimiter:
    def __init__(self, requests_per_minute: int = 100):
        self.limit = requests_per_minute
        self.log: dict = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self.log[client_id] = [t for t in self.log[client_id] if now - t < 60]
        if len(self.log[client_id]) >= self.limit:
            return False
        self.log[client_id].append(now)
        return True


rate_limiter = RateLimiter()


def sanitize_string(value: str, max_length: int = 500) -> str:
    if not value:
        return ""
    cleaned = re.sub(r'[<>"\\;{}]', "", str(value))
    return cleaned[:max_length]


env = OSMMapQualityEnvironment()
env.reset("task_easy")

app = FastAPI(
    title="OSM Map Quality Environment",
    description=(
        "OpenStreetMap data quality environment for AI agents. "
        "Agents inspect map features and fix issues such as missing tags, "
        "invalid coordinates, and conflicting attributes."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Max 100 requests per minute."},
            headers={"Retry-After": "60"},
        )
    return await call_next(request)


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_easy", pattern=r"^task_(easy|medium|hard)$")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v):
        if v not in VALID_TASK_IDS:
            raise ValueError(f"Invalid task_id. Allowed: {sorted(VALID_TASK_IDS)}")
        return v


class StepRequest(BaseModel):
    action_type: str = Field(..., pattern=r"^(set_tag|remove_tag|fix_coordinates|merge_duplicate|flag_invalid|mark_complete)$")
    tag_key: Optional[str] = Field(None, max_length=100)
    tag_value: Optional[str] = Field(None, max_length=500)
    coordinates: Optional[Dict[str, float]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v):
        if v not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action_type. Allowed: {sorted(VALID_ACTION_TYPES)}")
        return v

    @field_validator("tag_key", "tag_value")
    @classmethod
    def sanitize_tags(cls, v):
        return sanitize_string(v) if v else None

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v):
        if v is None:
            return v
        if "lat" not in v or "lon" not in v:
            raise ValueError("coordinates must contain 'lat' and 'lon'")
        if not (-90 <= v["lat"] <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= v["lon"] <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        return v


class GraderRequest(BaseModel):
    task_id: str = Field(..., pattern=r"^task_(easy|medium|hard)$")


def obs_to_dict(obs):
    try:
        return asdict(obs)
    except Exception:
        return obs.__dict__ if hasattr(obs, "__dict__") else str(obs)


def state_to_dict(s):
    try:
        return asdict(s)
    except Exception:
        return s.__dict__ if hasattr(s, "__dict__") else str(s)


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "OSM Map Quality Environment API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "grader": "POST /grader",
            "baseline": "POST /baseline",
        },
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "env": "osm-map-quality-env",
        "version": "2.0.0",
    }


@app.post("/reset", tags=["Environment"])
def reset(req: ResetRequest = None):
    """Reset the environment to the start of a new episode."""
    task_id = (req.task_id if req else None) or "task_easy"
    try:
        obs = env.reset(task_id=task_id)
        return {"observation": obs_to_dict(obs)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reset failed: {e}")


@app.post("/step", tags=["Environment"])
def step(req: StepRequest):
    """Submit one action and advance the episode by one step."""
    try:
        class Action:
            pass

        action = Action()
        action.action_type = req.action_type
        action.tag_key = req.tag_key
        action.tag_value = req.tag_value
        action.coordinates = req.coordinates
        action.confidence = req.confidence

        obs = env.step(action)
        return {
            "observation": obs_to_dict(obs),
            "reward": obs.reward,
            "done": obs.done,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Step failed: {e}")


@app.get("/state", tags=["Environment"])
def state():
    """Return the current episode state."""
    return {"state": state_to_dict(env.state)}


@app.get("/tasks", tags=["Environment"])
def tasks():
    """List all available tasks and the action schema."""
    return {"tasks": list_tasks()}


@app.post("/grader", tags=["Grading"])
def grader(req: GraderRequest):
    """Grade the current episode state for a given task."""
    try:
        snapshot = env.get_episode_snapshot()
        score = grade(req.task_id, snapshot)
        return {
            "task_id": req.task_id,
            "score": score,
            "snapshot": snapshot,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Grading failed: {e}")


@app.post("/baseline", tags=["Grading"])
def baseline():
    """Run a deterministic baseline agent across all tasks and return scores."""
    baseline_plans = {
        "task_easy": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Hyderabad Chai Cafe"},
            {"action_type": "mark_complete"},
        ],
        "task_medium": [
            {"action_type": "set_tag", "tag_key": "addr:street", "tag_value": "Jubilee Hills Road"},
            {"action_type": "set_tag", "tag_key": "addr:city", "tag_value": "Hyderabad"},
            {"action_type": "set_tag", "tag_key": "addr:postcode", "tag_value": "500033"},
            {"action_type": "set_tag", "tag_key": "addr:country", "tag_value": "IN"},
            {"action_type": "mark_complete"},
        ],
        "task_hard": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Yashoda Hospital"},
            {"action_type": "fix_coordinates", "coordinates": {"lat": 17.4449, "lon": 78.5011}},
            {"action_type": "set_tag", "tag_key": "addr:city", "tag_value": "Secunderabad"},
            {"action_type": "merge_duplicate"},
            {"action_type": "set_tag", "tag_key": "addr:street", "tag_value": "Alexander Road"},
            {"action_type": "set_tag", "tag_key": "website", "tag_value": "https://yashodahospitals.com"},
            {"action_type": "mark_complete"},
        ],
    }

    results = {}
    for task_id, plan in baseline_plans.items():
        try:
            env.reset(task_id=task_id)
            for entry in plan:
                class Action:
                    pass
                act = Action()
                act.action_type = entry["action_type"]
                act.tag_key = entry.get("tag_key")
                act.tag_value = entry.get("tag_value")
                act.coordinates = entry.get("coordinates")
                act.confidence = 1.0
                env.step(act)
            snapshot = env.get_episode_snapshot()
            score = grade(task_id, snapshot)
            results[task_id] = {"score": score, "status": "success"}
        except Exception as e:
            results[task_id] = {"score": 0.0, "status": "error", "error": str(e)}

    env.reset("task_easy")
    return {"baseline_scores": results}


def main():
    """Entry point for running the server directly."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
