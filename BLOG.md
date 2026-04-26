# OSM Map Quality Agent — Project Blog

**Author:** Dokka Vijay  
**Project:** OpenEnv Hackathon — Meta x PyTorch x Scaler  
**Date:** April 2025  
**Links:** [Live Space](https://huggingface.co/spaces/Arawn-1/osm-env) · [Trained Model](https://huggingface.co/Arawn-1/osm-map-quality-agent) · [Environment Repo](https://github.com/Arawn-D/osm-map-quality-env)

---

## Overview

This project trains an RL agent to fix real-world OpenStreetMap (OSM) data errors — missing tags, invalid coordinates, duplicate nodes, and conflicting fields — using Group Relative Policy Optimization (GRPO) over a custom world-model environment.

The environment is a FastAPI server (this repo) that exposes a stateful OSM simulation with partial observability, noise injection, and a 6-axis grader. An LLM agent calls the API, receives partial observations, and learns to take corrective actions through reward signals.

---

## The Problem

OpenStreetMap has hundreds of millions of nodes, ways, and relations. A significant fraction contain errors:

- Missing or misspelled name tags (e.g. `"Hospitl"` instead of `"Hospital"`)
- Invalid coordinate values (e.g. `lat: 99.9999`, which is outside `[-90, 90]`)
- Incomplete address fields (`addr:street`, `addr:city`, `addr:postcode`, `addr:country`)
- Duplicate nodes representing the same physical feature
- Internally inconsistent data (city name disagrees with coordinates)

Human mappers fix these one by one. This project asks: can an RL-trained LLM learn to do the same, reasoning across multiple steps under partial information?

---

## Environment Design

The environment (`server/`) is a world model, not a static dataset. Each `/reset` call generates a fresh task variation using randomized noise injection.

### Task Tiers

| Task | Difficulty | Issues | Max Steps | Key Challenge |
|---|---|---|---|---|
| `task_easy` | Easy | 1 | 10 | Identify POI type, set a valid name tag |
| `task_medium` | Medium | 4 | 20 | Complete all address fields; handle conflicting data |
| `task_hard` | Hard | 6 | 30 | Fix coordinates, resolve duplicate, sequence planning |

### Noise Injection (tasks.py)

Each episode introduces realistic data corruption:

- **Typos** — adjacent character swaps, deletions, or duplications in string values
- **Conflicts** — field replaced with a plausible-but-wrong alternative from a curated list
- **Stale data** — values prefixed with `UNVERIFIED:`, `OLD:`, or `~`

This means no two episodes are identical and the agent cannot memorize a fixed answer.

### Partial Observability

Agents start with limited tag visibility. Subsequent actions progressively reveal additional fields. Fixing coordinates, for example, may expose an address inconsistency that was not visible at the start of the episode.

### 6-Axis Grader (graders.py)

The grader scores each episode across multiple dimensions:

**task_easy grader:**
- Name presence: `+0.50`
- Name quality (length >= 3): `+0.30`
- Efficiency bonus (steps <= 2): `+0.10`
- Score range: `0.05 – 0.95`

**task_medium grader:**
- `addr:street` presence: `+0.20`
- `addr:city` presence: `+0.20`
- `addr:postcode` presence: `+0.17`
- `addr:country` presence: `+0.13`
- City/postcode consistency (known pairs): `+0.10`
- Efficiency bonus: `+0.10`

**task_hard grader:**
- Tag completeness across 6 fields (name, amenity, addr:city, addr:street, website, phone): up to `+0.45`
- Coordinate validity in Hyderabad/Secunderabad region: `+0.15`
- Duplicate merged: `+0.10`
- Data consistency (city matches coordinates): `+0.10`
- Action efficiency: `+0.10`
- Sequence quality (fix_coordinates before set_tag): `+0.05`

---

## Training Setup

### Model

- **Base:** `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit`
- **Adapter:** LoRA, rank 32, alpha 32, targeting attention and MLP projection layers
- **Framework:** Unsloth + TRL GRPO
- **Max sequence length:** 768 tokens
- **Max completion length:** 72 tokens

### GRPO Reward Design

The GRPO reward is not just the raw grader score. It stacks multiple signals:

1. **Grader score** — 6-axis episode score from `graders.py` (range `0.05–0.95`)
2. **Positional bonus** — Earlier correct actions in a rollout receive higher multipliers
3. **EOS bonus** — Reward bonus for clean episode termination (`mark_complete`)
4. **Per-task multipliers** — `task_hard` rollouts are weighted more heavily
5. **Cap** — Combined reward can exceed `1.0` when sequence bonuses stack correctly

### Training Data

- 28 hand-crafted dataset examples across all three task tiers
- Each example consists of a system prompt, a partial OSM observation, and a reference action sequence
- Dataset split: all examples used for training (no held-out validation set in this run)

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | GRPO |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Max sequence length | 768 |
| Max completion length | 72 |
| Rollouts per prompt | 5 |
| Training steps | 50 (checkpoint-50 selected) |
| Optimizer | AdamW (via Unsloth) |

---

## Results

### Reward Curve

Training ran for 50 steps with GRPO rollouts calling the live HF Space at `ENV_URL = https://arawn-1-osm-env.hf.space`.

| Metric | Value |
|---|---|
| Early mean reward (steps 1-10) | ~0.97 |
| Final mean reward (step 50) | ~1.04 |
| Reward improvement | +0.07 (~7.2%) |
| Best checkpoint | checkpoint-50 |

The reward exceeding `1.0` is expected and correct — it reflects positional and EOS bonuses stacking on top of a high grader score, not an error in the reward function.

### Grader Scores (Illustrative, post-training baseline run)

| Task | Score | Notes |
|---|---|---|
| `task_easy` | 0.95 | Name set correctly in 2 steps (max efficiency) |
| `task_medium` | 0.88 | All 4 address fields set; postcode/city pair validated |
| `task_hard` | 0.82 | Coordinates fixed, duplicate merged, correct action sequence |

---

## API Reference

The environment is a live FastAPI server with the following endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode. Body: `{"task_id": "task_easy"}` |
| `/step` | POST | Submit one action. Body: `{"action_type": "set_tag", "tag_key": "name", "tag_value": "...", "confidence": 0.9}` |
| `/state` | GET | Return current episode state |
| `/grader` | POST | Score the current episode. Body: `{"task_id": "task_hard"}` |
| `/baseline` | POST | Run deterministic baseline agent across all tasks |
| `/health` | GET | Returns version and status (`version: 2.2.0`) |
| `/docs` | GET | Swagger UI |

### Valid Action Types

```
set_tag          fix_coordinates     remove_tag
merge_duplicate  flag_invalid        mark_complete
```

### Example Interaction

```bash
# Start a hard episode
curl -X POST https://arawn-1-osm-env.hf.space/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_hard"}'

# Fix invalid coordinates
curl -X POST https://arawn-1-osm-env.hf.space/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "fix_coordinates", "coordinates": {"lat": 17.4449, "lon": 78.5011}, "confidence": 0.9}'

# Set hospital name
curl -X POST https://arawn-1-osm-env.hf.space/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "set_tag", "tag_key": "name", "tag_value": "Yashoda Hospital", "confidence": 0.9}'

# Merge duplicate node
curl -X POST https://arawn-1-osm-env.hf.space/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "merge_duplicate", "confidence": 0.85}'

# Get score
curl -X POST https://arawn-1-osm-env.hf.space/grader \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_hard"}'
```

---

## Key Design Decisions

**Why GRPO over PPO?**  
GRPO does not require a separate value/critic network. For a small model (1.5B parameters) with a sparse, multi-step reward, this reduces training instability and memory overhead significantly.

**Why a live API environment?**  
The agent calls a real HTTP server during rollouts. This forces the agent to produce structurally valid JSON and handle realistic latency, making the learned behavior closer to actual tool-use.

**Why partial observability?**  
Real OSM editing is not done with full information. Mappers inspect nodes, notice errors, fix them, and sometimes discover further issues as a result. The cascading error design replicates this.

**Why noise injection instead of a static dataset?**  
Static datasets allow memorization. Noise injection (typos, conflicts, stale data) forces genuine generalization — the agent must understand the structure of OSM data, not recall specific answers.

---

## Deployment

The environment runs as a Docker container on Hugging Face Spaces.

- **Space:** [Arawn-1/osm-env](https://huggingface.co/spaces/Arawn-1/osm-env)
- **Dockerfile:** `gunicorn` with `uvicorn` workers, port 7860
- **Version:** 2.2.0
- **Health check:** `GET /health` returns `{"status": "ok", "version": "2.2.0"}`

To rebuild the Space after a GitHub push:
1. Open Space Settings on Hugging Face
2. Confirm the source points to `Arawn-D/osm-map-quality-env` on branch `main`
3. Use Factory Reboot if an automatic rebuild does not trigger

---

## Repository Structure

```
osm-map-quality-env/
├── server/
│   ├── __init__.py
│   ├── app.py          # FastAPI app, UI v2.2.0, all endpoints
│   ├── environment.py  # OSMMapQualityEnvironment state machine
│   ├── graders.py      # 6-axis grader for all three task tiers
│   └── tasks.py        # Dynamic task generation with noise injection
├── Dockerfile          # gunicorn + uvicorn, port 7860
├── requirements.txt    # fastapi, uvicorn, pydantic
├── pyproject.toml      # entry point: server.app:app
├── openenv.yaml        # OpenEnv hackathon metadata
├── baseline.py         # Deterministic baseline agent script
├── inference.py        # Model inference helper
├── models.py           # Model loading utilities
├── README.md           # Project overview
├── BLOG.md             # This file
├── PROJECT_ANALYSIS.md # Detailed architecture notes
└── .gitignore
```

---

## What I Learned

Building a custom RL environment that an LLM calls over HTTP during training required solving several non-obvious problems:

- **Reward shaping matters more than model size.** A 1.5B model with a well-designed multi-component reward learned meaningful behavior within 50 steps.
- **Partial observability creates richer behavior.** Agents trained on fully-observable environments tend to take one action and stop. Partial observability forces multi-step reasoning.
- **Sequence bonuses drive correct ordering.** Without the sequence quality dimension in the hard grader (fix coordinates before setting address), the agent learned to set tags in arbitrary order.
- **Noise injection is essential for generalization.** Early experiments with static tasks showed the agent memorizing exact tag values. Noise injection eliminated this within a few training runs.

---

*Built by Dokka Vijay for the OpenEnv AI Hackathon (Meta x PyTorch x Scaler).*
