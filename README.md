---
title: OSM Map Quality Environment
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: "World-modeling RL environment for OSM data quality."
---

# OSM Map Quality Environment

![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-FFD21E?logo=huggingface)
![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-green)

**A world-modeling environment for geographic data quality assurance.**

Train AI agents to reason under partial observability, resolve conflicting data, and fix real-world OpenStreetMap issues — the way human mappers actually work.

## Architecture Highlights

This is not a static RL environment. It's a **world model** where agents must discover, reason, and adapt:

| Feature | Description |
|---|---|
| **Partial Observability** | Tags revealed progressively through actions. Agents start with limited data. |
| **Noisy & Conflicting Data** | Typos, stale values, contradictory fields — agents must reason about quality. |
| **Cascading Errors** | Fixing coordinates may reveal address inconsistencies. |
| **Confidence Calibration** | Overconfident wrong answers are penalized more heavily. |
| **Dynamic Generation** | Fresh task variations each episode. No two runs are identical. |
| **Multi-Dimensional Grading** | 6-axis scoring: completeness, consistency, efficiency, accuracy, merge, sequence. |

## Environment Tasks

| Task | Difficulty | Issues | Max Steps | Key Challenge |
|---|---|---|---|---|
| Missing Name Tag | Easy | 1 | 10 | Identify POI type, set appropriate name |
| Address Completion | Medium | 4 | 20 | Multiple fields, possible conflicting data |
| Duplicate Resolution | Hard | 6 | 30 | Invalid coords, tag conflicts, merge, sequence planning |

## API Interface

```bash
# Start episode
curl -X POST /reset -d '{"task_id":"task_hard"}'

# Take action
curl -X POST /step -d '{"action_type":"fix_coordinates","coordinates":{"lat":17.44,"lon":78.50},"confidence":0.9}'

# Get score
curl -X POST /grader -d '{"task_id":"task_hard"}'
```

## System Architecture

```
Agent → POST /reset → Partial Observation
     → POST /step  → Reward + Feedback + Tag Reveals + Cascading Discovery
     → POST /grader → 6-Dimensional Score (0.05 - 0.95)
```

## Training: GRPO with Grader-Optimized Rollouts

- **Model:** Qwen2.5-3B-Instruct (4-bit) + LoRA (r=32, 6 targets)
- **Method:** GRPO with 5-step grader-scored rollouts + positional bonuses
- **Result:** Agent learns multi-step reasoning and correct action sequences. Achieved a final mean reward of **1.043** (surpassing the 1.0 baseline via sequence bonuses).

## Links

- **Trained Model:** [HuggingFace Repo](https://huggingface.co/Arawn-1/osm-map-quality-agent)
- **Live Environment:** [HuggingFace Space](https://arawn-1-osm-env.hf.space)
- **API Docs:** [/docs](https://arawn-1-osm-env.hf.space/docs)
- **Health Check:** [/health](https://arawn-1-osm-env.hf.space/health)

---

Built by **Dokka Vijay** for the OpenEnv AI Hackathon.