from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, Field
from typing import Optional, Dict, Any
from dataclasses import asdict
import time
import re
import os
import json
import uuid
import logging
from collections import defaultdict
from .environment import OSMMapQualityEnvironment
from .tasks import list_tasks
from .graders import grade

VALID_TASK_IDS = {"task_easy", "task_medium", "task_hard"}
VALID_ACTION_TYPES = {
    "set_tag", "remove_tag", "fix_coordinates",
    "merge_duplicate", "flag_invalid", "mark_complete",
}
TASK_DEFAULT_TAGS = {
    "task_easy": {
        "name": "Hyderabad Chai Cafe",
    },
    "task_medium": {
        "addr:street": "Jubilee Hills Road",
        "addr:city": "Hyderabad",
        "addr:postcode": "500033",
        "addr:country": "IN",
    },
    "task_hard": {
        "name": "Yashoda Hospital",
        "addr:city": "Secunderabad",
        "addr:street": "Alexander Road",
        "website": "https://yashodahospitals.com",
    },
}
DEFAULT_PREDICT_STRATEGY = os.environ.get("OSM_PREDICT_MODE", "auto").strip().lower() or "auto"
LOGGER = logging.getLogger("osm_env")


class RateLimiter:
    def __init__(self, requests_per_minute: int = 100):
        self.limit = requests_per_minute
        self.log: dict = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self.log[client_id] = [t for t in self.log[client_id] if now - t < 60]
        if len(self.log[client_id]) >= self.limit:
            return False
        self.log[client_id].append(now)
        return True


rate_limiter = RateLimiter()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def error_response(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            },
        },
    )


def sanitize_string(value: str, max_length: int = 500) -> str:
    if not value:
        return ""
    cleaned = re.sub(r'[<>"\\;{}]', "", str(value))
    return cleaned[:max_length]


env = OSMMapQualityEnvironment()
env.reset("task_easy")

app = FastAPI(
    title="OSM Map Quality Environment",
    description="OpenStreetMap data quality environment for AI agents.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        resp = error_response(
            status_code=429,
            code="rate_limit_exceeded",
            message="Rate limit exceeded. Max 100 requests per minute.",
            request_id=_request_id(request),
        )
        resp.headers["Retry-After"] = "60"
        return resp
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_response(
        status_code=422,
        code="validation_error",
        message=str(exc),
        request_id=_request_id(request),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code = "not_found" if exc.status_code == 404 else "http_error"
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
        request_id=_request_id(request),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    LOGGER.exception("Unhandled server error", exc_info=exc)
    return error_response(
        status_code=500,
        code="internal_server_error",
        message="Unexpected server error.",
        request_id=_request_id(request),
    )


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_easy")


class StepRequest(BaseModel):
    action_type: str
    tag_key: Optional[str] = None
    tag_value: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    confidence: float = 1.0


class PredictRequest(BaseModel):
    observation: Dict[str, Any] = Field(default_factory=dict)
    task_id: Optional[str] = Field(default=None, pattern=r"^task_(easy|medium|hard)$")
    strategy: str = Field(default=DEFAULT_PREDICT_STRATEGY, pattern=r"^(auto|local|rule)$")
    max_new_tokens: int = Field(default=128, ge=16, le=512)


def obs_to_dict(obs):
    try:
        return asdict(obs)
    except Exception:
        return obs.__dict__


def _clamp_confidence(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = 0.8
    return max(0.0, min(1.0, num))


def _normalize_action(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        candidate = {}
    action_type = candidate.get("action_type")
    if action_type not in VALID_ACTION_TYPES:
        return {"action_type": "flag_invalid", "confidence": 0.2}

    payload: Dict[str, Any] = {"action_type": action_type, "confidence": _clamp_confidence(candidate.get("confidence", 0.9))}
    if action_type in {"set_tag", "remove_tag"}:
        tag_key = sanitize_string(candidate.get("tag_key", ""), max_length=100)
        if not tag_key:
            return {"action_type": "flag_invalid", "confidence": 0.2}
        payload["tag_key"] = tag_key
        if action_type == "set_tag":
            payload["tag_value"] = sanitize_string(candidate.get("tag_value", ""), max_length=500)
    if action_type == "fix_coordinates":
        coords = candidate.get("coordinates") or {}
        try:
            lat = float(coords.get("lat"))
            lon = float(coords.get("lon"))
        except (TypeError, ValueError):
            lat, lon = 17.4449, 78.5011
        payload["coordinates"] = {"lat": max(-90.0, min(90.0, lat)), "lon": max(-180.0, min(180.0, lon))}
    return payload


def _extract_coordinates(observation: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    if not isinstance(observation, dict):
        return None, None
    coords = observation.get("coordinates")
    if isinstance(coords, dict):
        return coords.get("lat"), coords.get("lon")
    return observation.get("lat"), observation.get("lon")


def _needs_coordinate_fix(observation: Dict[str, Any]) -> bool:
    lat, lon = _extract_coordinates(observation)
    if lat is not None and not (-90 <= float(lat) <= 90):
        return True
    if lon is not None and not (-180 <= float(lon) <= 180):
        return True
    feedback = str(observation.get("feedback", "")).lower()
    return any(word in feedback for word in ("coordinate", "lat", "lon", "invalid location"))


def _rule_policy(observation: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    tags = observation.get("current_tags", {}) if isinstance(observation.get("current_tags"), dict) else {}
    issues_remaining = int(observation.get("issues_remaining", 0) or 0)
    step_count = int(observation.get("step_count", 0) or 0)
    feedback = str(observation.get("feedback", "")).lower()

    if issues_remaining <= 0:
        return {"action_type": "mark_complete", "confidence": 1.0}

    if task_id == "task_hard" and _needs_coordinate_fix(observation):
        return {
            "action_type": "fix_coordinates",
            "coordinates": {"lat": 17.4449, "lon": 78.5011},
            "confidence": 0.95,
        }

    for key, expected in TASK_DEFAULT_TAGS.get(task_id, {}).items():
        current = str(tags.get(key, "")).strip()
        if not current or current.lower() != expected.lower():
            return {
                "action_type": "set_tag",
                "tag_key": key,
                "tag_value": expected,
                "confidence": 0.92,
            }

    if task_id == "task_hard" and observation.get("secondary_feature"):
        # Hard task requires merge_duplicate eventually.
        if "duplicate" in feedback or step_count >= 2:
            return {"action_type": "merge_duplicate", "confidence": 0.9}

    if issues_remaining <= 1:
        return {"action_type": "mark_complete", "confidence": 0.9}
    return {"action_type": "flag_invalid", "confidence": 0.3}


class LocalInferenceEngine:
    def __init__(self):
        self.model_name = os.environ.get("OSM_LOCAL_MODEL", "").strip()
        self.device_preference = os.environ.get("OSM_LOCAL_DEVICE", "cuda").strip() or "cuda"
        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None

    def _prompt(self, observation: Dict[str, Any], task_id: str) -> str:
        safe_obs = json.dumps(observation, ensure_ascii=True, separators=(",", ":"))
        return (
            "You are an OSM quality agent. Return exactly ONE JSON action object.\n"
            "Valid action_type: set_tag, remove_tag, fix_coordinates, merge_duplicate, flag_invalid, mark_complete.\n"
            f"Task: {task_id}\n"
            f"Observation: {safe_obs}\n"
            "Output JSON only:"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None

    def _ensure_loaded(self) -> Optional[str]:
        if self._model is not None and self._tokenizer is not None:
            return None
        if self._load_error:
            return self._load_error
        if not self.model_name:
            self._load_error = "OSM_LOCAL_MODEL is not set."
            return self._load_error
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.device_preference == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
            LOGGER.info("Loaded local inference model '%s'", self.model_name)
        except Exception as exc:
            self._load_error = f"Failed to load local model '{self.model_name}': {exc}"
            LOGGER.warning(self._load_error)
        return self._load_error

    def predict(self, observation: Dict[str, Any], task_id: str, max_new_tokens: int) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        err = self._ensure_loaded()
        if err:
            return None, err
        try:
            import torch

            prompt = self._prompt(observation, task_id)
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            if next(self._model.parameters()).is_cuda:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            decoded = self._tokenizer.decode(output[0], skip_special_tokens=True)
            parsed = self._extract_json(decoded[len(prompt):] if decoded.startswith(prompt) else decoded)
            if not parsed:
                return None, "Local model returned non-JSON output."
            return _normalize_action(parsed), None
        except Exception as exc:
            return None, f"Local generation failed: {exc}"


local_engine = LocalInferenceEngine()


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "OSM Map Quality Environment API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "grader": "POST /grader",
            "baseline": "POST /baseline",
            "predict": "POST /predict",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "env": "osm-map-quality-env",
        "version": "2.0.0",
        "predict_strategy_default": DEFAULT_PREDICT_STRATEGY,
        "local_model_enabled": bool(local_engine.model_name),
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else None) or "task_easy"
    obs = env.reset(task_id=task_id)
    return {"observation": obs_to_dict(obs)}


@app.post("/step")
def step(req: StepRequest):
    class Action:
        pass

    action = Action()
    action.action_type = req.action_type
    action.tag_key = req.tag_key
    action.tag_value = req.tag_value
    action.coordinates = req.coordinates
    action.confidence = req.confidence
    obs = env.step(action)
    return {"observation": obs_to_dict(obs), "reward": obs.reward, "done": obs.done}


@app.get("/state")
def state():
    return asdict(env.state)


@app.get("/tasks")
def tasks():
    return {"tasks": list_tasks()}


@app.post("/grader")
def grader(req: Dict[str, str]):
    task_id = req.get("task_id", "task_easy")
    snapshot = env.get_episode_snapshot()
    score = grade(task_id, snapshot)
    return {"task_id": task_id, "score": score}


@app.post("/baseline")
def baseline():
    baseline_plans = {
        "task_easy": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Hyderabad Chai Cafe"},
            {"action_type": "mark_complete"},
        ],
        "task_medium": [
            {"action_type": "set_tag", "tag_key": "addr:street", "tag_value": "Jubilee Hills Road"},
            {"action_type": "mark_complete"},
        ],
        "task_hard": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Yashoda Hospital"},
            {"action_type": "mark_complete"},
        ],
    }
    results = {}
    for task_id, plan in baseline_plans.items():
        try:
            env.reset(task_id=task_id)
            for entry in plan:
                class Action:
                    pass

                act = Action()
                act.action_type = entry["action_type"]
                act.tag_key = entry.get("tag_key")
                act.tag_value = entry.get("tag_value")
                env.step(act)
            snapshot = env.get_episode_snapshot()
            score = grade(task_id, snapshot)
            results[task_id] = {"score": score, "status": "success"}
        except Exception as e:
            results[task_id] = {"score": 0.0, "status": "error", "error": str(e)}
    return {"baseline_scores": results}


@app.post("/predict", tags=["Inference"])
def predict(req: PredictRequest, request: Request):
    """
    Predict one action for any incoming observation.
    Strategy:
    - auto: try local model first (if configured), then fallback to rule policy
    - local: require local model
    - rule: deterministic fallback policy only
    """
    observation = req.observation or {}
    task_id = req.task_id or observation.get("task_id") or "task_easy"
    if task_id not in VALID_TASK_IDS:
        raise HTTPException(status_code=422, detail=f"Invalid task_id '{task_id}'. Allowed: {sorted(VALID_TASK_IDS)}")

    strategy = (req.strategy or DEFAULT_PREDICT_STRATEGY).lower()
    warnings = []
    action = None
    source = "rule_policy"

    if strategy in {"auto", "local"}:
        action, err = local_engine.predict(observation, task_id=task_id, max_new_tokens=req.max_new_tokens)
        if action:
            source = "local_model"
        elif strategy == "local":
            raise HTTPException(status_code=503, detail=err or "Local model inference unavailable.")
        else:
            warnings.append(err or "Local model inference unavailable.")

    if action is None:
        action = _normalize_action(_rule_policy(observation, task_id=task_id))
        source = "rule_policy"

    return {
        "ok": True,
        "request_id": _request_id(request),
        "task_id": task_id,
        "source": source,
        "action": action,
        "warnings": warnings,
    }


def main():
    """Entry point for running the server directly."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
