"""
inference.py - LLM-based agent for the OSM Map Quality Environment.
Required by hackathon rules:
  - Uses the OpenAI client
  - Reads configuration from environment variables
  - Must be named inference.py and placed at the repo root
  - Emits structured [START], [STEP], [END] log lines in plain text
"""
import os
import sys
import json
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

SUCCESS_THRESHOLD = 0.5
MAX_RETRIES = 3

TASKS = [
    {"id": "tag_completeness", "max_steps": 10},
    {"id": "geometry_validity", "max_steps": 10},
    {"id": "address_quality", "max_steps": 10},
]


def log_start(task_id, max_steps):
    print(f"[START] task_id={task_id} max_steps={max_steps}", flush=True)


def log_step(step, action, reward, done, error=None):
    parts = f"[STEP] step={step} action={action} reward={reward:.4f} done={done}"
    if error:
        parts += f" error={error}"
    print(parts, flush=True)


def log_end(success, steps, score, rewards):
    avg_r = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"[END] success={success} steps={steps} score={score:.4f} avg_reward={avg_r:.4f}",
        flush=True,
    )


def get_observation(task_id):
    try:
        r = requests.get(f"{ENV_URL}/observation", params={"task_id": task_id}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "observation": "Unable to fetch observation"}


def build_prompt(task_id, observation):
    obs_text = json.dumps(observation, indent=2) if isinstance(observation, dict) else str(observation)
    return [
        {
            "role": "system",
            "content": (
                "You are an expert OpenStreetMap data quality agent. "
                "Analyze the observation and return a single JSON action object. "
                "Valid action types: add_tag, fix_geometry, validate_address, complete_task. "
                "Response format: {\"action\": \"<type>\", \"params\": {<key>: <value>}}"
            ),
        },
        {
            "role": "user",
            "content": f"Task: {task_id}\n\nObservation:\n{obs_text}\n\nWhat action should be taken?",
        },
    ]


def call_llm(messages):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"[WARN] LLM attempt {attempt + 1} failed: {e}", flush=True)
    return "{\"action\": \"complete_task\", \"params\":{}}"


def parse_action(llm_output):
    try:
        # Try to extract JSON from the response
        start = llm_output.find("{")
        end = llm_output.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(llm_output[start:end])
    except Exception:
        pass
    return {"action": "complete_task", "params": {}}


def step_env(task_id, action_obj):
    try:
        r = requests.post(
            f"{ENV_URL}/step",
            json={"task_id": task_id, "action": action_obj},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"reward": 0.0, "done": True, "error": str(e)}


def run_task(task_id, max_steps):
    log_start(task_id, max_steps)
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, max_steps + 1):
            steps_taken = step
            reward = 0.0
            done = False
            error = None
            action_str = "complete_task"

            try:
                obs = get_observation(task_id)
                messages = build_prompt(task_id, obs)
                llm_out = call_llm(messages)
                action_obj = parse_action(llm_out)
                action_str = action_obj.get("action", "complete_task")

                result = step_env(task_id, action_obj)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                error = result.get("error")
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

        try:
            r = requests.post(f"{ENV_URL}/grader", json={"task_id": task_id}, timeout=30)
            r.raise_for_status()
            score = float(r.json().get("score", 0.0))
            score = min(max(score, 0.0), 1.0)
        except Exception:
            score = (sum(rewards) / len(rewards)) if rewards else 0.0
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    results = {}
    for task in TASKS:
        try:
            score = run_task(task["id"], task["max_steps"])
            results[task["id"]] = score
        except Exception as e:
            print(f"[ERROR] {task['id']} failed: {e}", flush=True)
            results[task["id"]] = 0.0

    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"[RESULT] average_score={avg:.4f}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
