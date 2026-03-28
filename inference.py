"""
inference.py — LLM-based agent for OSM Map Quality Environment.
Required by hackathon rules: uses OpenAI client, reads env vars.
Must be named inference.py and placed in repo root.
"""
import os
import sys
import json
import requests
from openai import OpenAI

# ─── Read required environment variables ──────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

# ─── OpenAI client (required by hackathon rules) ───────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """You are an OSM map data quality inspector agent.
You receive a JSON observation of a map feature with issues.
You must return a single JSON action object with these fields:
  action_type: one of [set_tag, remove_tag, fix_coordinates, merge_duplicate, flag_invalid, mark_complete]
  tag_key: string (optional, for set_tag/remove_tag)
  tag_value: string (optional, for set_tag)
  coordinates: {"lat": float, "lon": float} (optional, for fix_coordinates)
  confidence: float 0.0-1.0

Return ONLY valid JSON, no explanation, no markdown, no backticks.
Examples:
  {"action_type": "set_tag", "tag_key": "name", "tag_value": "Chai Point", "confidence": 0.9}
  {"action_type": "fix_coordinates", "coordinates": {"lat": 17.4449, "lon": 78.5011}, "confidence": 0.95}
  {"action_type": "merge_duplicate", "confidence": 0.8}
  {"action_type": "mark_complete", "confidence": 1.0}"""


def call_llm(observation: dict) -> dict:
    """Call the LLM with the current observation, get back an action."""
    obs_text = json.dumps(observation, indent=2)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Current observation:\n{obs_text}\n\nWhat action do you take?"}
        ]
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_task(task_id: str) -> float:
    """Run one full episode with the LLM agent. Returns final score."""
    # Reset the environment
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()

    for step in range(50):  # safety cap
        if obs.get("done", False):
            break

        try:
            action = call_llm(obs)
        except Exception as e:
            print(f"  LLM error at step {step}: {e}")
            action = {"action_type": "mark_complete", "confidence": 0.5}

        r = requests.post(f"{ENV_URL}/step", json=action)
        if r.status_code != 200:
            print(f"  Step error: {r.text}")
            break
        obs = r.json()

    # Grade the episode
    r = requests.post(f"{ENV_URL}/grader", json={"task_id": task_id})
    r.raise_for_status()
    result = r.json()
    return result.get("score", 0.0)


def main():
    print("=" * 55)
    print(" OSM Map Quality Env — LLM Agent Inference")
    print("=" * 55)
    print(f" Model:   {MODEL_NAME}")
    print(f" API:     {API_BASE_URL}")
    print(f" Env URL: {ENV_URL}")
    print("-" * 55)

    results = {}
    all_ok = True

    for task_id in TASKS:
        print(f"\nRunning {task_id}...")
        try:
            score = run_task(task_id)
            results[task_id] = score
            status = "PASS" if score >= 0.5 else "LOW"
            print(f"  [{status}] Score: {score:.4f}")
            if not (0.0 <= score <= 1.0):
                print(f"  [FAIL] Score out of range!")
                all_ok = False
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[task_id] = 0.0
            all_ok = False

    print("\n" + "=" * 55)
    avg = sum(results.values()) / len(results)
    print(f" Average Score: {avg:.4f}")
    print(f" All tasks completed: {all_ok}")
    print("=" * 55)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
