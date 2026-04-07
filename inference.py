"""
inference.py - LLM-based agent for OSM Map Quality Environment.
Required by hackathon rules: uses OpenAI client, reads env vars.
Must be named inference.py and placed in repo root.
Emits structured [START], [STEP], [END] logs as required.
"""
import os
import sys
import json
import requests
from openai import OpenAI

# --- Read required environment variables ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

# --- OpenAI client (required by hackathon rules) ---
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

TASKS = ["task_easy", "task_medium", "task_hard"]
BENCHMARK = "osm-map-quality-env"
MAX_STEPS = 30
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an OSM map data quality inspector agent.
You receive a JSON observation of a map feature with issues.
You must return a single JSON action object with these fields:
  action_type: one of [set_tag, remove_tag, fix_coordinates, merge_duplicate, flag_invalid, mark_complete]
  tag_key: string (optional, for set_tag/remove_tag)
  tag_value: string (optional, for set_tag)
  coordinates: {"lat": float, "lon": float} (optional, for fix_coordinates)
  confidence: float 0.0-1.0
Return ONLY valid JSON, no explanation, no markdown, no backticks."""


def log_start(task, env, model):
    print(json.dumps({"type": "[START]", "task": task, "env": env, "model": model}), flush=True)


def log_step(step, action, reward, done, error=None):
    print(json.dumps({"type": "[STEP]", "step": step, "action": action, "reward": reward, "done": done, "error": error}), flush=True)


def log_end(success, steps, score, rewards):
    print(json.dumps({"type": "[END]", "success": success, "steps": steps, "score": score, "rewards": rewards}), flush=True)


def call_llm(observation: dict) -> dict:
    obs_text = json.dumps(observation, indent=2)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current observation:\n{obs_text}\n\nWhat action do you take?"}
        ]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_task(task_id: str) -> float:
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        r.raise_for_status()
        result = r.json()
        obs = result.get("observation", result)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break
            error = None
            try:
                action = call_llm(obs)
            except Exception as e:
                error = str(e)
                action = {"action_type": "mark_complete", "confidence": 0.5}

            r = requests.post(f"{ENV_URL}/step", json=action)
            if r.status_code != 200:
                error = r.text
                break
            result = r.json()
            obs = result.get("observation", result)
            reward = result.get("reward", 0.0) or 0.0
            done = result.get("done", False)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=json.dumps(action), reward=reward, done=done, error=error)

            if done:
                break

        # Grade
        r = requests.post(f"{ENV_URL}/grader", json={"task_id": task_id})
        r.raise_for_status()
        grader_result = r.json()
        score = grader_result.get("score", 0.0)
        score = min(max(float(score), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    results = {}
    for task_id in TASKS:
        try:
            score = run_task(task_id)
            results[task_id] = score
        except Exception as e:
            print(f"[DEBUG] {task_id} failed: {e}", flush=True)
            results[task_id] = 0.0

    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"[DEBUG] Average Score: {avg:.4f}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
