from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dataclasses import asdict
import traceback

from .environment import OSMMapQualityEnvironment
from .tasks import list_tasks
from .graders import grade

# ── Shared environment instance ──────────────────────
env = OSMMapQualityEnvironment()
env.reset("task_easy")  # initialise default task

# ── FastAPI app ───────────────────────────────────────
app = FastAPI(
    title="OSM Map Quality Environment",
    description="Real-world OpenStreetMap data quality environment for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request schemas ─────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    action_type: str
    tag_key: Optional[str] = None
    tag_value: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    confidence: float = 1.0


class GraderRequest(BaseModel):
    task_id: str


# ── Helper ────────────────────────────────────────────
def obs_to_dict(obs):
    try:
        return asdict(obs)
    except Exception:
        return obs.__dict__ if hasattr(obs, '__dict__') else str(obs)


def state_to_dict(s):
    try:
        return asdict(s)
    except Exception:
        return s.__dict__ if hasattr(s, '__dict__') else str(s)


# ── /health ───────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "env": "osm-map-quality-env", "version": "1.0.0"}


# ── /reset ────────────────────────────────────────────
@app.post("/reset")
def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else None) or "task_easy"
    try:
        obs = env.reset(task_id=task_id)
        return {"observation": obs_to_dict(obs)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── /step ─────────────────────────────────────────────
@app.post("/step")
def step(req: StepRequest):
    try:
        # Build action using a simple object
        class _Action:
            pass
        action = _Action()
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
        raise HTTPException(status_code=400, detail=str(e))


# ── /state ────────────────────────────────────────────
@app.get("/state")
def state():
    return {"state": state_to_dict(env.state)}


# ── /tasks ────────────────────────────────────────────
@app.get("/tasks")
def tasks():
    return {"tasks": list_tasks()}


# ── /grader ───────────────────────────────────────────
@app.post("/grader")
def grader(req: GraderRequest):
    try:
        snapshot = env.get_episode_snapshot()
        score = grade(req.task_id, snapshot)
        return {"task_id": req.task_id, "score": score, "snapshot": snapshot}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── /baseline ─────────────────────────────────────────
@app.post("/baseline")
def baseline():
    results = {}
    tasks_to_run = ["task_easy", "task_medium", "task_hard"]
    baseline_actions = {
        "task_easy": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Hyderabad Chai Cafe"},
            {"action_type": "mark_complete"},
        ],
        "task_medium": [
            {"action_type": "set_tag", "tag_key": "addr:street",   "tag_value": "Jubilee Hills Road"},
            {"action_type": "set_tag", "tag_key": "addr:city",     "tag_value": "Hyderabad"},
            {"action_type": "set_tag", "tag_key": "addr:postcode", "tag_value": "500033"},
            {"action_type": "set_tag", "tag_key": "addr:country",  "tag_value": "IN"},
            {"action_type": "mark_complete"},
        ],
        "task_hard": [
            {"action_type": "set_tag",         "tag_key": "name",      "tag_value": "Yashoda Hospital"},
            {"action_type": "fix_coordinates", "coordinates": {"lat": 17.4449, "lon": 78.5011}},
            {"action_type": "set_tag",         "tag_key": "addr:city", "tag_value": "Secunderabad"},
            {"action_type": "merge_duplicate"},
            {"action_type": "set_tag",         "tag_key": "addr:street", "tag_value": "Alexander Road"},
            {"action_type": "set_tag",         "tag_key": "website",     "tag_value": "https://yashodahospitals.com"},
            {"action_type": "mark_complete"},
        ],
    }

    for task_id in tasks_to_run:
        try:
            env.reset(task_id=task_id)
            actions = baseline_actions[task_id]
            for a in actions:
                class _Act:
                    pass
                act = _Act()
                act.action_type = a["action_type"]
                act.tag_key = a.get("tag_key")
                act.tag_value = a.get("tag_value")
                act.coordinates = a.get("coordinates")
                act.confidence = 1.0
                env.step(act)
            snapshot = env.get_episode_snapshot()
            score = grade(task_id, snapshot)
            results[task_id] = {"score": score, "status": "success"}
        except Exception as e:
            results[task_id] = {"score": 0.0, "status": "error", "error": str(e)}

    # Reset to default after baseline run
    env.reset("task_easy")
    return {"baseline_scores": results}


# ── Run locally ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
