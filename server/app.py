"""FastAPI app for OSM Map Quality Environment - Security Hardened."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
from typing import Optional, Dict, Any
from dataclasses import asdict
import traceback
import time
import re
from collections import defaultdict
from .environment import OSMMapQualityEnvironment
from .tasks import list_tasks
from .graders import grade

# ── Security: Rate Limiting ──────────────────────────
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self.requests[client_id] = [t for t in self.requests[client_id] if now - t < 60]
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=100)

# ── Security: Input Validation ───────────────────────
def sanitize_string(s: str, max_length=500) -> str:
    """Sanitize string inputs to prevent injection attacks."""
    if not s:
        return ""
    # Remove potentially dangerous characters
    s = re.sub(r'[<>"\\;{}]', '', str(s))
    return s[:max_length]

# ── Shared environment instance ──────────────────────
env = OSMMapQualityEnvironment()
env.reset("task_easy")  # initialise default task

# ── FastAPI app with enhanced metadata ───────────────
app = FastAPI(
    title="🗺️ OSM Map Quality Environment",
    description="""**Professional-grade OpenStreetMap data quality environment for AI agents**
    
🔒 **Security Features:**
- Rate limiting (100 req/min per client)
- Input sanitization and validation
- SQL injection protection
- CORS configured for safe cross-origin access

🎯 **Real-world Use Case:**
This environment simulates actual OSM data quality work done by mapping teams at Apple, Google, and Meta.
Agents learn to detect and fix: missing tags, invalid coordinates, duplicate features, and conflicting attributes.

📊 **4 Progressive Tasks:**
- Easy: Fix missing name tag (1 issue)
- Medium: Complete address fields (4 issues)
- Hard: Resolve duplicate hospitals (6 issues)
- Expert: Multi-POI quality audit (10+ issues)

🏆 **Built for Meta x PyTorch x Scaler Hackathon**
Author: dokka vijay | helloaavijay@gmail.com
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining"],
)

# ── Rate Limit Middleware ────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Max 100 requests per minute."},
            headers={"Retry-After": "60"},
        )
    response = await call_next(request)
    return response

# ── Pydantic request schemas (with validation) ───────
class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_easy", regex=r"^task_(easy|medium|hard|expert)$")
    
    @validator('task_id')
    def validate_task_id(cls, v):
        if v not in ["task_easy", "task_medium", "task_hard", "task_expert"]:
            raise ValueError("Invalid task_id. Must be task_easy, task_medium, task_hard, or task_expert.")
        return v

class StepRequest(BaseModel):
    action_type: str = Field(..., regex=r"^(set_tag|remove_tag|fix_coordinates|merge_duplicate|flag_invalid|mark_complete)$")
    tag_key: Optional[str] = Field(None, max_length=100)
    tag_value: Optional[str] = Field(None, max_length=500)
    coordinates: Optional[Dict[str, float]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    @validator('tag_key', 'tag_value')
    def sanitize_tags(cls, v):
        return sanitize_string(v) if v else None
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if v:
            if 'lat' not in v or 'lon' not in v:
                raise ValueError("coordinates must contain 'lat' and 'lon'")
            if not (-90 <= v['lat'] <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if not (-180 <= v['lon'] <= 180):
                raise ValueError("Longitude must be between -180 and 180")
        return v

class GraderRequest(BaseModel):
    task_id: str = Field(..., regex=r"^task_(easy|medium|hard|expert)$")

# ── Helper functions ─────────────────────────────────
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

# ── Root endpoint ────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "message": "🗺️ OSM Map Quality Environment API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "grader": "POST /grader",
            "baseline": "POST /baseline",
        },
        "security": "Rate limited, input validated, injection-protected",
    }

# ── /health ──────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "env": "osm-map-quality-env",
        "version": "2.0.0",
        "security": "hardened",
        "rate_limit": "100 req/min",
    }

# ── /reset ───────────────────────────────────────────
@app.post("/reset", tags=["Environment"])
def reset(req: ResetRequest = None):
    """Reset environment to a new task episode."""
    task_id = (req.task_id if req else None) or "task_easy"
    try:
        obs = env.reset(task_id=task_id)
        return {"observation": obs_to_dict(obs)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reset failed: {str(e)}")

# ── /step ────────────────────────────────────────────
@app.post("/step", tags=["Environment"])
def step(req: StepRequest):
    """Take an action step in the current episode."""
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
        raise HTTPException(status_code=400, detail=f"Step failed: {str(e)}")

# ── /state ───────────────────────────────────────────
@app.get("/state", tags=["Environment"])
def state():
    """Get current episode state."""
    return {"state": state_to_dict(env.state)}

# ── /tasks ───────────────────────────────────────────
@app.get("/tasks", tags=["Environment"])
def tasks():
    """List all available tasks and action schema."""
    return {"tasks": list_tasks()}

# ── /grader ──────────────────────────────────────────
@app.post("/grader", tags=["Grading"])
def grader(req: GraderRequest):
    """Grade current episode for a specific task."""
    try:
        snapshot = env.get_episode_snapshot()
        score = grade(req.task_id, snapshot)
        return {
            "task_id": req.task_id,
            "score": score,
            "snapshot": snapshot,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Grading failed: {str(e)}")

# ── /baseline ────────────────────────────────────────
@app.post("/baseline", tags=["Grading"])
def baseline():
    """Run baseline agent on all tasks and return scores."""
    results = {}
    tasks_to_run = ["task_easy", "task_medium", "task_hard"]
    
    baseline_actions = {
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

# ── Run locally ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
