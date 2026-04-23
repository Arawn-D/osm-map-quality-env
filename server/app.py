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
    description="OpenStreetMap data quality environment for AI agents.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
    return await call_next(request)


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_easy")


class StepRequest(BaseModel):
    action_type: str
    tag_key: Optional[str] = None
    tag_value: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    confidence: float = 1.0


def obs_to_dict(obs):
    try:
        return asdict(obs)
    except Exception:
        return obs.__dict__


@app.get("/")
def root():
    return {"service": "OSM Map Quality Environment API", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "env": "osm-map-quality-env"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else None) or "task_easy"
    obs = env.reset(task_id=task_id)
    return {"observation": obs_to_dict(obs)}


@app.post("/step")
def step(req: StepRequest):
    class Action:
        pass

    action = Action()
    action.action_type = req.action_type
    action.tag_key = req.tag_key
    action.tag_value = req.tag_value
    action.coordinates = req.coordinates
    action.confidence = req.confidence
    obs = env.step(action)
    return {"observation": obs_to_dict(obs), "reward": obs.reward, "done": obs.done}


@app.get("/state")
def state():
    return asdict(env.state)


@app.get("/tasks")
def tasks():
    return {"tasks": list_tasks()}


@app.post("/grader")
def grader(req: Dict[str, str]):
    task_id = req.get("task_id", "task_easy")
    snapshot = env.get_episode_snapshot()
    score = grade(task_id, snapshot)
    return {"task_id": task_id, "score": score}


@app.post("/baseline")
def baseline():
    baseline_plans = {
        "task_easy": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Hyderabad Chai Cafe"},
            {"action_type": "mark_complete"},
        ],
        "task_medium": [
            {"action_type": "set_tag", "tag_key": "addr:street", "tag_value": "Jubilee Hills Road"},
            {"action_type": "mark_complete"},
        ],
        "task_hard": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Yashoda Hospital"},
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
                env.step(act)
            snapshot = env.get_episode_snapshot()
            score = grade(task_id, snapshot)
            results[task_id] = {"score": score, "status": "success"}
        except Exception as e:
            results[task_id] = {"score": 0.0, "status": "error", "error": str(e)}
    return {"baseline_scores": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
