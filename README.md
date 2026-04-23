# OSM Map Quality Environment

> **Meta x PyTorch x Scaler OpenEnv Hackathon — Round 1 Submission**
> Author: dokka vijay | helloaavijay@gmail.com

## Overview

A real-world [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an AI agent acts as a map data quality inspector for **OpenStreetMap (OSM)** features. The agent detects and fixes data quality issues — missing tags, invalid coordinates, duplicate features, and conflicting attributes — simulating the actual work done by mapping teams at Apple, Google, and Meta.

This environment is grounded in real Hyderabad map data and is directly relevant to the OSM/mapping domain.

---

## Environment Description

| Property | Value |
|---|---|
| Framework | OpenEnv (openenv-core) |
| Language | Python 3.11 |
| Server | FastAPI + Uvicorn |
| Port | 7860 |
| Tasks | 3 (easy, medium, hard) |
| Score Range | 0.0 – 1.0 per task |
| Reward Type | Partial (not binary) |

---

## Tasks

### Task 1 — Easy: Missing Name Tag Fix
- **Feature:** Cafe POI in Banjara Hills, Hyderabad
- **Issue:** `name` tag is missing
- **Agent must:** Identify the missing tag and set a meaningful name
- **Max steps:** 10
- **Grader:** Returns 1.0 if name set, 0.5 if too short, 0.0 if missing

### Task 2 — Medium: Multi-Field Address Completion
- **Feature:** Residential building in Jubilee Hills, Hyderabad
- **Issues:** 4 missing address tags (`addr:street`, `addr:city`, `addr:postcode`, `addr:country`)
- **Agent must:** Set all 4 tags with correct values
- **Max steps:** 20
- **Grader:** 0.25 per correct tag (partial credit)

### Task 3 — Hard: Duplicate & Conflicting Feature Resolution
- **Features:** Two near-duplicate hospital nodes with conflicts
- **Issues:** Name typo, invalid latitude (99.9999), wrong city, missing street/website, duplicate unresolved
- **Agent must:** Fix name, coordinates, city, merge duplicate, set street and website
- **Max steps:** 30
- **Grader:** 1/6 per resolved issue, partial credit for approximate values

---

## Action Space

| Field | Type | Values |
|---|---|---|
| `action_type` | string | `set_tag`, `remove_tag`, `fix_coordinates`, `merge_duplicate`, `flag_invalid`, `mark_complete` |
| `tag_key` | string (optional) | Any OSM tag key, e.g. `name`, `addr:city` |
| `tag_value` | string (optional) | Value to assign |
| `coordinates` | object (optional) | `{"lat": float, "lon": float}` |
| `confidence` | float | 0.0 – 1.0 |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `feature_id` | string | OSM node/way ID |
| `feature_type` | string | `node` or `way` |
| `current_tags` | object | Current tag key-value pairs |
| `issues_remaining` | int | Number of unsolved issues |
| `feedback` | string | Human-readable result of last action |
| `reward` | float | Reward for last step |
| `done` | bool | Whether episode is complete |
| `task_id` | string | Current task ID |
| `step_count` | int | Steps taken so far |
| `secondary_feature` | object | Duplicate feature (hard task only) |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns 200 if service is up |
| `/reset` | POST | Start a new episode. Body: `{"task_id": "task_easy"}` |
| `/step` | POST | Take an action. Body: action object |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Grade current episode. Body: `{"task_id": "..."}` |
| `/baseline` | POST | Run full baseline agent on all 3 tasks |

---

## Quick Start

### Option 1: Run locally with Python

```bash
# 1. Clone the repo
git clone https://github.com/Arawn-D/osm-map-quality-env.git
cd osm-map-quality-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 4. Run the baseline script
python baseline.py
```

### Option 2: Run with Docker

```bash
# Build
docker build -t osm-map-quality-env .

# Run
docker run -p 7860:7860 osm-map-quality-env

# Test health
curl http://localhost:7860/health
```

### Test the environment manually

```bash
# Reset with easy task
curl -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "set_tag", "tag_key": "name", "tag_value": "Chai Point"}'

# Grade the episode
curl -X POST http://localhost:7860/grader \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_easy"}'

# Run baseline on all 3 tasks
curl -X POST http://localhost:7860/baseline
```

---

## Baseline Scores

The rule-based baseline agent achieves:

| Task | Score |
|---|---|
| task_easy | 1.0 |
| task_medium | 1.0 |
| task_hard | 1.0 |
| **Average** | **1.0** |

Run `python baseline.py` to reproduce.

---

## Project Structure

```
osm-map-quality-env/
├── openenv.yaml          # OpenEnv spec (environment metadata, tasks, endpoints)
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── baseline.py           # Reproducible baseline inference script
├── models.py             # MapAction, MapObservation, MapState dataclasses
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI server (all endpoints)
    ├── environment.py    # Core OSMMapQualityEnvironment class
    ├── tasks.py          # Task definitions with real Hyderabad data
    └── graders.py        # Per-task graders with partial credit
```

---

## Why This Environment Stands Out

1. **Real-world domain** — Based on actual OSM mapping work done by professional teams
2. **Authentic data** — Hyderabad locations, real street names, valid coordinates
3. **Meaningful reward signal** — Partial credit at every step, not binary
4. **Progressive difficulty** — 1 issue → 4 issues → 6 issues across tasks
5. **Domain expertise** — Built by someone who works on OSM/Apple mapping projects

---

## License

MIT — open for evaluation by the hackathon judges.


## Live Demo

Deployed on Hugging Face Spaces:
[https://arawn-1-osm-env.hf.space](https://arawn-1-osm-env.hf.space)

Health check:
[https://arawn-1-osm-env.hf.space/health](https://arawn-1-osm-env.hf.space/health)

## Submission

- **GitHub**: https://github.com/Arawn-D/osm-map-quality-env
- **HF Space**: https://huggingface.co/spaces/Arawn-1/osm-env
