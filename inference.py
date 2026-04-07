"""
inference.py - LLM-based agent for OSM Map Quality Environment.
Required by hackathon rules: uses OpenAI client, reads env vars.
Must be named inference.py and placed in repo root.
Emits structured [START], [STEP], [END] logs in PLAIN TEXT format.
"""
import os
import sys
import json
import requests
from openai import OpenAI

# --- Required environment variables ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

# --- OpenAI client (required by hackathon rules) ---
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

BENCHMARK = "osm-map-quality-env"
SUCCESS_SCORE_THRESHOLD = 0.5

# Tasks with their individual max steps
TASKS = [
    {"id": "task_easy",   "max_steps": 10},
    {"id": "task_medium", "max_steps": 20},
    {"id": "task_hard",   "max_steps": 30},
]

SYSTEM_PROMPT = """You are an OSM map data quality inspector agent.
You receive a JSON observation of a map feature with issues.
You must return a single JSON action object with these fields:
  action_type: one of [set_tag, remove_tag, fix_coordinates, merge_duplicate, flag_invalid, mark_complete]
  tag_key: string (optional, for set_tag/remove_tag)
  tag_value: string (optional, for set_tag)
  coordinates: {"lat": float, "lon": float} (optional, for fix_coordinates)
  confidence: float 0.0-1.0
Return ONLY valid JSON, no explanation, no markdown, no backticks."""


# --- Logging functions: PLAIN TEXT format as required by sample ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_str = "null" if error is None else str(error).replace(" ", "_")[:50]
    done_str = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = str(success).lower()
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def call_llm(observation: dict) -> dict:
    """Call LLM via OpenAI client and return parsed action dict."""
    obs_text = json.dumps(observation, indent=2)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current observation:\n{obs_text}\n\nWhat action do you take?"},
            ]
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        # Fallback action if LLM fails
        return {"action_type": "mark_complete", "confidence": 0.5}


def format_action_str(action: dict) -> str:
    """Format action dict as a readable string for [STEP] log."""
    atype = action.get("action_type", "unknown")
    if atype == "set_tag":
        return f"set_tag({action.get('tag_key','')},{action.get('tag_value','')})"
    elif atype == "remove_tag":
        return f"remove_tag({action.get('tag_key','')})"
    elif atype == "fix_coordinates":
        coords = action.get("coordinates", {})
        return f"fix_coordinates({coords.get('lat',0)},{coords.get('lon',0)})"
    elif atype == "merge_duplicate":
        return "merge_duplicate()"
    elif atype == "flag_invalid":
        return "flag_invalid()"
    elif atype == "mark_complete":
        return "mark_complete()"
    else:
        return f"{atype}()"


def run_task(task_id: str, max_steps: int) -> float:
    """Run one task episode and return final score."""
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        result = r.json()
        obs = result.get("observation", result)
        done = obs.get("done", False) if isinstance(obs, dict) else False

        for step in range(1, max_steps + 1):
            if done:
                break

            error = None
            action = {"action_type": "mark_complete", "confidence": 0.5}

            try:
                action = call_llm(obs if isinstance(obs, dict) else {})
            except Exception as e:
                error = str(e)

            action_str = format_action_str(action)

            try:
                r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
                r.raise_for_status()
                result = r.json()
                obs = result.get("observation", result)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            reward = min(max(reward, 0.0), 1.0)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Get final score from grader
        try:
            r = requests.post(f"{ENV_URL}/grader", json={"task_id": task_id}, timeout=30)
            r.raise_for_status()
            grader_result = r.json()
            score = float(grader_result.get("score", 0.0))
            score = min(max(score, 0.0), 1.0)
        except Exception:
            score = (sum(rewards) / len(rewards)) if rewards else 0.0
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    results = {}
    for task in TASKS:
        task_id = task["id"]
        max_steps = task["max_steps"]
        try:
            score = run_task(task_id, max_steps)
            results[task_id] = score
        except Exception as e:
            print(f"[DEBUG] {task_id} failed: {e}", flush=True)
            results[task_id] = 0.0

    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"[DEBUG] Average Score: {avg:.4f}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
