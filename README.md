---
title: OSM Map Quality Environment
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: "World-modeling RL environment: partial observability, noisy data, cascading errors"
---

# 🌍 OSM Map Quality Environment

**A world-modeling environment for geographic data quality assurance.**

Train AI agents to reason under partial observability, resolve conflicting data, and fix real-world OpenStreetMap issues — the way human mappers actually work.

## 🧠 What Makes This Different

This is not a static RL environment. It's a **world model** where agents must discover, reason, and adapt:

| Innovation | Description |
|---|---|
| 👁 **Partial Observability** | Tags revealed progressively through actions. Agents start with limited data. |
| 🔀 **Noisy & Conflicting Data** | Typos, stale values, contradictory fields — agents must reason about quality. |
| ⛓ **Cascading Errors** | Fixing coordinates may reveal address inconsistencies. |
| 📊 **Confidence Calibration** | Overconfident wrong answers are penalized more heavily. |
| 🎲 **Dynamic Generation** | Fresh task variations each episode. No two runs are identical. |
| 📐 **Multi-Dimensional Grading** | 6-axis scoring: completeness, consistency, efficiency, accuracy, merge, sequence. |

## 🎯 Tasks

| Task | Difficulty | Issues | Max Steps | Key Challenge |
|---|---|---|---|---|
| Missing Name Tag | Easy | 1 | 10 | Identify POI type, set appropriate name |
| Address Completion | Medium | 4 | 20 | Multiple fields, possible conflicting data |
| Duplicate Resolution | Hard | 6 | 30 | Invalid coords, tag conflicts, merge, sequence planning |

## 🔌 API

```bash
# Start episode
curl -X POST /reset -d '{"task_id":"task_hard"}'

# Take action
curl -X POST /step -d '{"action_type":"fix_coordinates","coordinates":{"lat":17.44,"lon":78.50},"confidence":0.9}'

# Get score
curl -X POST /grader -d '{"task_id":"task_hard"}'
```

## 🏗 Architecture

```
Agent → POST /reset → Partial Observation
     → POST /step  → Reward + Feedback + Tag Reveals + Cascading Discovery
     → POST /grader → 6-Dimensional Score (0.05 - 0.95)
```

## 📊 Training: GRPO with Grader-Optimized Rollouts

- **Model:** Qwen2.5-3B-Instruct (4-bit) + LoRA (r=32, 6 targets)
- **Method:** GRPO with 5-step grader-scored rollouts + positional bonuses
- **Result:** Agent learns multi-step reasoning and correct action sequences

## 🔗 Links

- **Live Environment:** [HuggingFace Space](https://arawn-1-osm-env.hf.space)
- **API Docs:** [/docs](https://arawn-1-osm-env.hf.space/docs)
- **Health Check:** [/health](https://arawn-1-osm-env.hf.space/health)

---

Built by **Dokka Vijay** for the OpenEnv AI Hackathon.