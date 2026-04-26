# OSM Map Quality Environment - Deep Project Analysis

## Executive Summary

This project is a **real-world OpenEnv environment** for the Meta x PyTorch x Scaler OpenEnv Hackathon. It simulates an AI agent acting as a map data quality inspector for **OpenStreetMap (OSM)** features. The agent detects and fixes data quality issues including missing tags, invalid coordinates, duplicate features, and conflicting attributes.

---

## 1. Project Overview

| Property | Value |
|----------|-------|
| **Framework** | OpenEnv (openenv-core) |
| **Language** | Python 3.11 |
| **Web Server** | FastAPI + Uvicorn |
| **Container** | Docker (python:3.11-slim-bookworm) |
| **Deployment** | Hugging Face Spaces |
| **Port** | 7860 (HF Spaces) / 8000 (local) |
| **Tasks** | 3 (easy, medium, hard) |
| **Score Range** | 0.0 – 1.0 per task |
| **Reward Type** | Partial (not binary) |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│  │  baseline.py │  │ inference.py │  │  curl/HTTP   │                    │
│  │ (Rule-based) │  │  (LLM Agent) │  │   Client     │                    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                    │
└─────────┼──────────────────┼──────────────────┼──────────────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌──────────────────────────────┼────────────────────────────────────────────┐
│                         API LAYER (FastAPI)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                           server/app.py                              │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │ │
│  │  │  GET /  │ │/health  │ │/tasks   │ │/state   │ │         │      │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │         │      │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │         │      │ │
│  │  │POST /reset│ │POST /step│ │POST /grader│ │POST /baseline│      │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      │ │
│  │                                                                    │ │
│  │  Middleware: Rate Limiting (100 req/min), CORS, Input Sanitization │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┼────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼────────────────────────────────────────────┐
│                      ENVIRONMENT LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                     server/environment.py                            │ │
│  │                                                                    │ │
│  │  OSMMapQualityEnvironment (extends openenv.core.env_server.Environment)│ │
│  │  ├── reset(task_id) → MapObservation                                │ │
│  │  ├── step(action) → MapObservation                                  │ │
│  │  ├── _apply_action(action) → (reward, feedback)                     │ │
│  │  └── get_episode_snapshot() → Dict                                  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┼────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼────────────────────────────────────────────┐
│                       DATA/MODEL LAYER                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │   server/tasks.py  │  │ server/graders.py │  │    models.py      │     │
│  │  Task Definitions │  │ Grading Functions│  │  Data Classes     │     │
│  │  - task_easy     │  │  - grade_easy()   │  │  - MapAction      │     │
│  │  - task_medium   │  │  - grade_medium() │  │  - MapObservation │     │
│  │  - task_hard     │  │  - grade_hard()   │  │  - MapState       │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components Deep Dive

### 3.1 Data Models (`models.py`)

The project uses Python dataclasses to define the core data structures:

#### MapAction
```python
@dataclass
class MapAction(Action):
    action_type: str = ""           # set_tag, remove_tag, fix_coordinates, merge_duplicate, flag_invalid, mark_complete
    tag_key: Optional[str] = None   # OSM tag key (e.g., "name", "addr:city")
    tag_value: Optional[str] = None # Value to assign
    coordinates: Optional[Dict[str, float]] = None  # {"lat": float, "lon": float}
    confidence: float = 1.0         # Agent confidence 0.0-1.0
```

#### MapObservation
```python
@dataclass
class MapObservation(Observation):
    feature_id: str           # OSM node/way ID (e.g., "node/1234567")
    feature_type: str         # "node" or "way"
    current_tags: Dict[str, str]  # Current tag key-value pairs
    issues_remaining: int     # Number of unsolved issues
    feedback: str             # Human-readable result of last action
    reward: float             # Reward for last step
    done: bool                # Whether episode is complete
    task_id: str              # Current task ID
    step_count: int           # Steps taken so far
    secondary_feature: Optional[Dict[str, Any]]  # Duplicate feature (hard task only)
```

#### MapState
```python
@dataclass
class MapState(State):
    task_id: str
    step_count: int
    episode_id: str               # UUID for tracking
    accumulated_reward: float
    issues_fixed: int
    issues_total: int
    last_action_type: str
    is_done: bool
```

### 3.2 Task Definitions (`server/tasks.py`)

The environment includes **3 progressive difficulty tasks** using real Hyderabad, India map data:

| Task | Difficulty | Description | Max Steps | Issues |
|------|------------|-------------|-----------|--------|
| **task_easy** | Easy | Missing name tag on a cafe in Banjara Hills | 10 | 1 |
| **task_medium** | Medium | 4 missing address tags on a residential building in Jubilee Hills | 20 | 4 |
| **task_hard** | Hard | Hospital duplicate with conflicting tags, invalid coordinates (lat=99.9999), name typo | 30 | 6 |

#### Task Data Structure Example (task_easy):
```python
"task_easy": {
    "id": "task_easy",
    "name": "Missing Name Tag Fix",
    "difficulty": "easy",
    "description": "A cafe in Hyderabad is missing its name tag...",
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
}
```

### 3.3 Environment Engine (`server/environment.py`)

The core environment class extends `openenv.core.env_server.Environment`:

#### Key Methods:

**`reset(task_id)`** - Initialize new episode
- Loads task configuration
- Creates episode UUID
- Resets state tracking
- Returns initial observation

**`step(action)`** - Execute one action
- Validates action type
- Calls `_apply_action()` to modify feature state
- Calculates reward
- Checks for episode completion
- Returns observation with feedback

**`_apply_action(action)`** - Action execution logic:

| Action Type | Behavior | Reward Logic |
|-------------|----------|--------------|
| `set_tag` | Sets tag_key=tag_value | +0.30 if fixes required issue, +0.05 otherwise |
| `remove_tag` | Removes tag by key | +0.05 if removed, -0.05 if not found |
| `fix_coordinates` | Updates lat/lon | +0.30 if in expected range, -0.10 otherwise |
| `merge_duplicate` | Inherits tags from secondary feature | +0.20 + removes merge from required_fixes |
| `flag_invalid` | Flags feature for review | +0.02 |
| `mark_complete` | Ends episode | +0.10 if all issues fixed, -0.10 otherwise |

#### Episode Completion Logic:
```python
episode_complete = len(self._remaining_fixes) == 0 or self._state.step_count >= max_steps
if episode_complete and len(self._remaining_fixes) == 0:
    reward += 0.2  # Bonus for completing all issues
    feedback += " All issues resolved. Bonus +0.20."
```

### 3.4 Grading System (`server/graders.py`)

Implements **partial credit scoring** (not binary):

#### grade_easy()
- No name: 0.05
- Name < 3 chars: 0.40
- Valid name: 0.95

#### grade_medium()
- Base: 0.05
- addr:street present: +0.25
- addr:city present: +0.25
- addr:postcode present: +0.25
- addr:country present: +0.15

#### grade_hard()
- Base: 0.05
- name: +0.15, amenity: +0.10, addr:city: +0.10
- addr:street: +0.10, website: +0.10, phone: +0.05
- Valid coordinates (17.0 ≤ lat ≤ 18.0, 78.0 ≤ lon ≤ 79.0): +0.15
- Duplicate merged: +0.15

All scores clamped to [0.05, 0.95] range.

### 3.5 FastAPI Server (`server/app.py`)

#### API Endpoints:

| Endpoint | Method | Request Body | Response |
|----------|--------|--------------|----------|
| `/` | GET | - | Service info & endpoint map |
| `/health` | GET | - | `{"status": "ok", ...}` |
| `/reset` | POST | `{"task_id": "task_easy"}` | Initial observation |
| `/step` | POST | Action object | Observation with reward/done |
| `/state` | GET | - | Full episode state |
| `/tasks` | GET | - | List of all tasks with schemas |
| `/grader` | POST | `{"task_id": "task_easy"}` | Score & snapshot |
| `/baseline` | POST | - | Runs all 3 tasks, returns scores |

#### Security Features:
1. **Rate Limiting**: 100 requests/minute per IP
2. **Input Sanitization**: Regex removes `<",;{}` from strings
3. **CORS**: Allows all origins (`*`) for HF Spaces compatibility
4. **Pydantic Validation**: Strict field validation with patterns

---

## 4. Build & Deployment System

### 4.1 Dependencies (`requirements.txt`)

```
openenv-core>=0.1.1      # OpenEnv framework core
fastapi>=0.110.0          # Web framework
uvicorn[standard]>=0.29.0 # ASGI server
pydantic>=2.0.0          # Data validation
httpx>=0.27.0            # HTTP client
requests>=2.31.0         # HTTP client (baseline/inference)
openai>=1.0.0            # LLM client (inference.py)
```

### 4.2 Build Configuration (`pyproject.toml`)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "osm-map-quality-env"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "requests>=2.28.0",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "openenv-core>=0.2.0",
]

[tool.openenv]
env_class = "server.environment:OSMMapQualityEnvironment"
env_id = "osm-map-quality-env"
version = "1.0.0"
```

### 4.3 Docker Containerization (`Dockerfile`)

```dockerfile
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl

# Layer caching: requirements first
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

# Copy application code
COPY . /app/
RUN touch /app/__init__.py  # Make root a package

EXPOSE 7860  # HF Spaces standard port

# Health check for OpenEnv compliance
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 4.4 OpenEnv Specification (`openenv.yaml`)

```yaml
name: osm-map-quality-env
version: "1.0.0"
description: A real-world OpenStreetMap data quality checking environment

environment:
  type: http
  port: 7860

action:
  type: object
  fields:
    action_type:
      type: string
      enum: [set_tag, remove_tag, fix_coordinates, merge_duplicate, flag_invalid, mark_complete]
    tag_key: {type: string, required: false}
    tag_value: {type: string, required: false}
    coordinates: {type: object, required: false}
    confidence: {type: number}

observation:
  type: object
  fields:
    feature_id: {type: string}
    feature_type: {type: string}
    current_tags: {type: object}
    issues_remaining: {type: integer}
    feedback: {type: string}
    reward: {type: number}
    done: {type: boolean}
    task_id: {type: string}
    step_count: {type: integer}

tasks:
  - id: task_easy
    name: Missing Name Tag Fix
    difficulty: easy
    max_steps: 10
    score_range: [0.0, 1.0]
  - id: task_medium
    name: Multi-Field Address Completion
    difficulty: medium
    max_steps: 20
    score_range: [0.0, 1.0]
  - id: task_hard
    name: Duplicate and Conflicting Feature Resolution
    difficulty: hard
    max_steps: 30
    score_range: [0.0, 1.0]

endpoints:
  step: POST /step
  reset: POST /reset
  state: GET /state
  tasks: GET /tasks
  grader: POST /grader
  baseline: POST /baseline
  health: GET /health
```

---

## 5. Agent Implementations

### 5.1 Rule-Based Baseline (`baseline.py`)

Deterministic agent that achieves **1.0 scores on all tasks**:

```python
def run_task_easy(env):
    env.reset(task_id="task_easy")
    env.step(make_action("set_tag", tag_key="name", tag_value="Hyderabad Chai Cafe"))
    env.step(make_action("mark_complete"))
    return grade("task_easy", env.get_episode_snapshot())

def run_task_medium(env):
    env.reset(task_id="task_medium")
    env.step(make_action("set_tag", tag_key="addr:street", tag_value="Jubilee Hills Road"))
    env.step(make_action("set_tag", tag_key="addr:city", tag_value="Hyderabad"))
    env.step(make_action("set_tag", tag_key="addr:postcode", tag_value="500033"))
    env.step(make_action("set_tag", tag_key="addr:country", tag_value="IN"))
    env.step(make_action("mark_complete"))
    return grade("task_medium", env.get_episode_snapshot())

def run_task_hard(env):
    env.reset(task_id="task_hard")
    env.step(make_action("set_tag", tag_key="name", tag_value="Yashoda Hospital"))
    env.step(make_action("fix_coordinates", coordinates={"lat": 17.4449, "lon": 78.5011}))
    env.step(make_action("set_tag", tag_key="addr:city", tag_value="Secunderabad"))
    env.step(make_action("merge_duplicate"))
    env.step(make_action("set_tag", tag_key="addr:street", tag_value="Alexander Road"))
    env.step(make_action("set_tag", tag_key="website", tag_value="https://yashodahospitals.com"))
    env.step(make_action("mark_complete"))
    return grade("task_hard", env.get_episode_snapshot())
```

### 5.2 LLM-Based Agent (`inference.py`)

Uses OpenAI-compatible API for agent decision-making:

#### Configuration (via environment variables):
- `API_BASE_URL`: LLM API endpoint (default: OpenAI)
- `MODEL_NAME`: Model to use (default: gpt-4o-mini)
- `HF_TOKEN`: API key
- `ENV_URL`: Environment server URL (default: http://localhost:7860)

#### System Prompt:
```
You are an OSM map data quality inspector agent.
You receive a JSON observation describing a map feature with data quality issues.
Return a single JSON action object with these fields:
  action_type: one of [set_tag, remove_tag, fix_coordinates, merge_duplicate, flag_invalid, mark_complete]
  tag_key: string (required for set_tag and remove_tag)
  tag_value: string (required for set_tag)
  coordinates: {"lat": float, "lon": float} (required for fix_coordinates)
  confidence: float between 0.0 and 1.0
Return ONLY valid JSON. No explanation, no markdown, no code fences.
```

#### Logging Format (hackathon requirement):
- `[START] task={task} env={env} model={model}`
- `[STEP] step={step} action={action} reward={reward} done={done} error={error}`
- `[END] success={success} steps={steps} score={score} rewards={rewards}`
- `[RESULT] average_score={avg}`

---

## 6. Project File Structure

```
osm-map-quality-env/
├── openenv.yaml              # OpenEnv specification (environment metadata, tasks, endpoints)
├── Dockerfile                # Container definition for HF Spaces
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Python package configuration (hatchling build)
├── README.md                # Full documentation (211 lines)
├── baseline.py              # Rule-based baseline agent (achieves 1.0 on all tasks)
├── inference.py             # LLM-based agent for hackathon evaluation
├── models.py                # Data classes: MapAction, MapObservation, MapState
└── server/                  # Main application package
    ├── __init__.py          # Package exports
    ├── app.py               # FastAPI server with all endpoints (291 lines)
    ├── environment.py       # Core OSMMapQualityEnvironment class (215 lines)
    ├── tasks.py             # Task definitions with Hyderabad map data (126 lines)
    └── graders.py          # Per-task grading functions (82 lines)
```

**Total Code:** ~1000 lines of Python across 9 source files

---

## 7. Key Design Decisions

### 7.1 Why OpenEnv Framework?
- Standardized interface for AI agent environments
- HTTP-based communication enables language-agnostic agents
- Built-in compliance checking for hackathon requirements

### 7.2 Why FastAPI?
- Automatic API documentation (`/docs`, `/redoc`)
- Native Pydantic integration for data validation
- High performance (async/await support)
- Industry standard for Python web APIs

### 7.3 Why Real Hyderabad Data?
- Authentic map data quality issues (real street names, coordinates)
- Simulates actual work done by Apple/Google/Meta mapping teams
- Progressive complexity mirrors real-world QA workflows

### 7.4 Why Partial Credit?
- Realistic reward signal for RL training
- Encourages incremental progress
- Better distinguishes agent performance levels

---

## 8. Deployment

### Local Development:
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
python baseline.py  # Test with rule-based agent
```

### Docker:
```bash
docker build -t osm-map-quality-env .
docker run -p 8000:8000 osm-map-quality-env
curl http://localhost:8000/health
```

### Hugging Face Spaces:
- **Live URL:** https://arawn-1-osm-env.hf.space
- **Health Check:** https://arawn-1-osm-env.hf.space/health
- Port 7860 (HF Spaces standard)
- Docker container auto-deploys on push

---

## 9. Conclusion

The OSM Map Quality Environment is a **well-architected, production-ready** OpenEnv implementation featuring:

- **Clean separation of concerns**: API/Environment/Data layers
- **Real-world domain relevance**: Actual OSM data quality workflows
- **Progressive difficulty**: 1 → 4 → 6 issues across tasks
- **Multiple agent interfaces**: Rule-based baseline + LLM inference
- **Robust deployment**: Docker + HF Spaces + local dev
- **Security**: Rate limiting, input sanitization, CORS

The project demonstrates best practices for building AI agent environments and serves as a solid reference for OpenEnv hackathon submissions.

---

## References

- **GitHub:** https://github.com/Arawn-D/osm-map-quality-env
- **HF Space:** https://huggingface.co/spaces/Arawn-1/osm-env
- **OpenEnv:** https://github.com/meta-pytorch/OpenEnv
- **OpenStreetMap:** https://www.openstreetmap.org
