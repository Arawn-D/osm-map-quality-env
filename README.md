# OSM Map Quality Environment

<div align="center">

![OpenStreetMap](https://img.shields.io/badge/OpenStreetMap-7EBC6F?style=for-the-badge&logo=openstreetmap&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-orange?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Meta%20x%20PyTorch%20x%20Scaler-Round%201-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Arawn-1/osm-env)

**A production-grade reinforcement learning environment for AI-driven OpenStreetMap data quality inspection.**

[Live Demo](https://huggingface.co/spaces/Arawn-1/osm-env) | [Documentation](#documentation) | [Quick Start](#quick-start) | [API Reference](#api-reference)

</div>

---

## Overview

The **OSM Map Quality Environment** is a real-world [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment where an AI agent acts as a map data quality inspector for **OpenStreetMap (OSM)** features.

The agent autonomously detects and fixes data quality issues:
- Missing or incomplete tags
- Invalid geometries and coordinates
- Duplicate features and conflicting attributes
- Address quality problems

This environment is grounded in **real Hyderabad map data** and simulates the actual quality-control work performed by mapping teams at Apple, Google, and Meta.

---

## Key Features

| Feature | Description |
|---|---|
| Real OSM Data | Grounded in live Hyderabad, India map data |
| Multi-Task | 3 quality-inspection tasks with independent scoring |
| LLM Agent | GPT-4o-mini powered decision-making via OpenAI API |
| OpenEnv Compatible | Follows `openenv-core` protocol for hackathon evaluation |
| Pydantic Models | Type-safe request/response validation |
| Structured Logging | `[START]`, `[STEP]`, `[END]` log lines for evaluation |
| REST API | FastAPI-based environment server on port 7860 |

---

## Architecture

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns 200 if service is up |
| `/reset` | POST | Start a new episode. Body: `{"task_id": "task_easy"}` |
| `/step` | POST | Take an action. Body: action object |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Grade current episode. Body: `{"task_id": "..."}` |
| `/baseline` | POST | Run full baseline agent on all 3 tasks |
| `/predict` | POST | Return one valid action for an arbitrary observation (`auto` local-model + rule fallback) |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)
- OpenAI API key or compatible LLM endpoint

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Arawn-D/osm-map-quality-env.git
cd osm-map-quality-env

# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn app:app --host 0.0.0.0 --port 7860

# In a separate terminal, run the agent
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key-here"
export ENV_URL="http://localhost:7860"
python inference.py
```

### Docker Setup

```bash
# Build the image
docker build -t osm-map-quality-env .

# Run the container
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your-api-key-here" \
  osm-map-quality-env
```

---

```bash
# Reset with easy task
curl -X POST http://localhost:8000/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_easy"}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "set_tag", "tag_key": "name", "tag_value": "Chai Point"}'

# Grade the episode
curl -X POST http://localhost:8000/grader \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_easy"}'

# Run baseline on all 3 tasks
curl -X POST http://localhost:8000/baseline

# Predict one action for any observation
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"task_easy","observation":{"current_tags":{"amenity":"cafe"},"issues_remaining":1,"feedback":"Missing name","step_count":0}}'
```

### `/predict` modes (no-login friendly)

- `strategy=auto` (default): tries local model first (if configured), then falls back to deterministic rule policy
- `strategy=local`: requires local model, returns `503` if unavailable
- `strategy=rule`: deterministic rule policy only

Environment variables:

- `OSM_PREDICT_MODE` = `auto|local|rule` (default `auto`)
- `OSM_LOCAL_MODEL` = local/public HF model id or path (optional)
- `OSM_LOCAL_DEVICE` = `cuda|cpu` (default `cuda`)

### Error response envelope

All API errors use a structured JSON envelope:

```json
{
  "ok": false,
  "error": {
    "code": "not_found",
    "message": "Not Found",
    "request_id": "..."
  }
}
```

---

## Tasks

### 1. Tag Completeness (`tag_completeness`)
Detects OSM features missing required tags (name, amenity type, opening hours) and adds appropriate metadata.

### 2. Geometry Validity (`geometry_validity`)
Identifies and repairs invalid geometries — self-intersecting polygons, duplicate nodes, and out-of-bounds coordinates.

### 3. Address Quality (`address_quality`)
Validates and corrects address fields including street names, postal codes, and administrative boundaries.

---

## Scoring

| Metric | Weight | Description |
|---|---|---|
| Tag Completeness | 40% | Ratio of correctly tagged features |
| Geometry Validity | 35% | Ratio of geometrically valid features |
| Address Quality | 25% | Address field correctness score |

**Success Threshold:** `0.5` (50% weighted average)

Scores are clamped to `[0.0, 1.0]` and reported as `[RESULT] average_score=X.XXXX`.

---

## API Reference

### Environment Server (port 7860)

#### `GET /observation`
Returns the current environment observation for a task.

```json
{
  "task_id": "tag_completeness",
  "features": [...],
  "step": 1
}
```

#### `POST /step`
Submits an action and receives a reward.

```json
{
  "task_id": "tag_completeness",
  "action": {"action": "add_tag", "params": {"key": "name", "value": "Cafe Bistro"}}
}
```

#### `POST /grader`
Returns the final score for a completed task.

```json
{"task_id": "tag_completeness"}
```

---

## Log Format

The agent emits structured log lines compliant with the OpenEnv evaluation protocol:

```
[START] task_id=tag_completeness max_steps=10
[STEP] step=1 action=add_tag reward=0.7500 done=False
[STEP] step=2 action=complete_task reward=1.0000 done=True
[END] success=True steps=2 score=0.8750 avg_reward=0.8750
[RESULT] average_score=0.8750
```

---

## Hackathon Compliance

This submission meets all **Meta x PyTorch x Scaler OpenEnv Hackathon — Round 1** requirements:

- [x] Uses the `openai` Python client
- [x] Reads all config from environment variables
- [x] `inference.py` placed at repo root
- [x] Emits `[START]`, `[STEP]`, `[END]` structured log lines
- [x] Hosted on HuggingFace Spaces
- [x] `openenv.yaml` with required metadata and `[openenv]` tag
- [x] Score clamped to `[0.0, 1.0]`
- [x] `sys.exit(0)` on completion

---

## Author

**dokka vijay**
Email: helloaavijay@gmail.com
GitHub: [Arawn-D](https://github.com/Arawn-D)
HuggingFace: [Arawn-1](https://huggingface.co/Arawn-1)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with OpenStreetMap data | Powered by OpenEnv | Meta x PyTorch x Scaler Hackathon 2025
</div>
