"""
Baseline inference script for the OSM Map Quality Environment.
Runs a rule-based baseline agent on all 3 tasks and reports scores.
Usage: python baseline.py
"""
import sys
import os

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import OSMMapQualityEnvironment
from server.graders import grade


def make_action(action_type, tag_key=None, tag_value=None, coordinates=None):
    class Action:
        pass
    a = Action()
    a.action_type = action_type
    a.tag_key = tag_key
    a.tag_value = tag_value
    a.coordinates = coordinates
    a.confidence = 1.0
    return a


def run_task_easy(env):
    """Baseline for task_easy: set the name tag."""
    env.reset(task_id="task_easy")
    env.step(make_action("set_tag", tag_key="name", tag_value="Hyderabad Chai Cafe"))
    env.step(make_action("mark_complete"))
    snapshot = env.get_episode_snapshot()
    return grade("task_easy", snapshot)


def run_task_medium(env):
    """Baseline for task_medium: complete all 4 address tags."""
    env.reset(task_id="task_medium")
    env.step(make_action("set_tag", tag_key="addr:street",   tag_value="Jubilee Hills Road"))
    env.step(make_action("set_tag", tag_key="addr:city",     tag_value="Hyderabad"))
    env.step(make_action("set_tag", tag_key="addr:postcode", tag_value="500033"))
    env.step(make_action("set_tag", tag_key="addr:country",  tag_value="IN"))
    env.step(make_action("mark_complete"))
    snapshot = env.get_episode_snapshot()
    return grade("task_medium", snapshot)


def run_task_hard(env):
    """Baseline for task_hard: fix typo, coords, city, merge, street, website."""
    env.reset(task_id="task_hard")
    env.step(make_action("set_tag",         tag_key="name",       tag_value="Yashoda Hospital"))
    env.step(make_action("fix_coordinates", coordinates={"lat": 17.4449, "lon": 78.5011}))
    env.step(make_action("set_tag",         tag_key="addr:city",  tag_value="Secunderabad"))
    env.step(make_action("merge_duplicate"))
    env.step(make_action("set_tag",         tag_key="addr:street", tag_value="Alexander Road"))
    env.step(make_action("set_tag",         tag_key="website",     tag_value="https://yashodahospitals.com"))
    env.step(make_action("mark_complete"))
    snapshot = env.get_episode_snapshot()
    return grade("task_hard", snapshot)


def main():
    print("=" * 55)
    print(" OSM Map Quality Environment - Baseline Inference")
    print("=" * 55)

    env = OSMMapQualityEnvironment()
    results = {}

    runners = [
        ("task_easy",   run_task_easy,   "Easy:   Missing Name Tag Fix"),
        ("task_medium", run_task_medium, "Medium: Multi-Field Address Completion"),
        ("task_hard",   run_task_hard,   "Hard:   Duplicate & Conflicting Feature Resolution"),
    ]

    all_passed = True
    for task_id, runner, label in runners:
        try:
            score = runner(env)
            results[task_id] = score
            status = "PASS" if score >= 0.5 else "LOW"
            print(f"  [{status}] {label}")
            print(f"         Score: {score:.4f}")
            if score < 0.0 or score > 1.0:
                print(f"  [FAIL] Score out of range [0.0, 1.0]!")
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            results[task_id] = 0.0
            all_passed = False

    print("-" * 55)
    avg = sum(results.values()) / len(results)
    print(f"  Average Score: {avg:.4f}")
    print(f"  All scores in [0.0, 1.0]: {all_passed}")
    print("=" * 55)

    # Exit 0 = success for automated validation
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
