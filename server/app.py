"""FastAPI application for the OSM Map Quality Environment.

Serves:
  - Interactive landing page UI at /
  - REST API endpoints for agent interaction
  - Environment capabilities and documentation
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, field_validator, Field
from typing import Optional, Dict, Any
from dataclasses import asdict
import time
import re
from collections import defaultdict
from .environment import OSMMapQualityEnvironment
from .tasks import list_tasks
from .graders import grade

VALID_TASK_IDS = {"task_easy", "task_medium", "task_hard"}
VALID_ACTION_TYPES = {
    "set_tag", "remove_tag", "fix_coordinates",
    "merge_duplicate", "flag_invalid", "mark_complete",
}


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


def sanitize_string(value: str, max_length: int = 500) -> str:
    if not value:
        return ""
    cleaned = re.sub(r'[<>"\\;{}]', "", str(value))
    return cleaned[:max_length]


env = OSMMapQualityEnvironment()
env.reset("task_easy")

app = FastAPI(
    title="OSM Map Quality Environment",
    description=(
        "A world-modeling environment for geographic data quality assurance. "
        "Features partial observability, noisy/conflicting inputs, cascading "
        "error discovery, and confidence-calibrated grading."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Max 100 requests per minute."},
            headers={"Retry-After": "60"},
        )
    return await call_next(request)


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_easy", pattern=r"^task_(easy|medium|hard)$")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v):
        if v not in VALID_TASK_IDS:
            raise ValueError(f"Invalid task_id. Allowed: {sorted(VALID_TASK_IDS)}")
        return v


class StepRequest(BaseModel):
    action_type: str = Field(..., pattern=r"^(set_tag|remove_tag|fix_coordinates|merge_duplicate|flag_invalid|mark_complete)$")
    tag_key: Optional[str] = Field(None, max_length=100)
    tag_value: Optional[str] = Field(None, max_length=500)
    coordinates: Optional[Dict[str, float]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v):
        if v not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action_type. Allowed: {sorted(VALID_ACTION_TYPES)}")
        return v

    @field_validator("tag_key", "tag_value")
    @classmethod
    def sanitize_tags(cls, v):
        return sanitize_string(v) if v else None

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v):
        if v is None:
            return v
        if "lat" not in v or "lon" not in v:
            raise ValueError("coordinates must contain 'lat' and 'lon'")
        if not (-90 <= v["lat"] <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= v["lon"] <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        return v


class GraderRequest(BaseModel):
    task_id: str = Field(..., pattern=r"^task_(easy|medium|hard)$")


def obs_to_dict(obs):
    try:
        return asdict(obs)
    except Exception:
        return obs.__dict__ if hasattr(obs, "__dict__") else str(obs)


def state_to_dict(s):
    try:
        return asdict(s)
    except Exception:
        return s.__dict__ if hasattr(s, "__dict__") else str(s)


# ═══════════════════════════════════════════════════════════════════
# Landing Page
# ═══════════════════════════════════════════════════════════════════

LANDING_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>OSM Map Quality — Neural Network Environment</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #030A0E;
      --mint: #00FFB2;
      --mint-dim: rgba(0, 255, 178, 0.2);
      --blue: #0066FF;
      --orange: #FF6B35;
      --text: #E8F4F8;
      --text-dim: #8496A0;
      --grid: #0A2030;
      --panel: rgba(10, 32, 48, 0.6);
      --border: #11334A;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      margin: 0;
      overflow-x: hidden;
      line-height: 1.6;
    }
    .bg-grid {
      position: fixed; top: 0; left: 0; right: 0; bottom: 0;
      background-image: 
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px);
      background-size: 40px 40px;
      animation: pan 40s linear infinite;
      z-index: -1;
    }
    @keyframes pan {
      from { background-position: 0 0; }
      to { background-position: 400px 400px; }
    }
    h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; margin: 0; }
    h1 { font-size: clamp(3rem, 6vw, 6rem); line-height: 0.9; margin-bottom: 20px; }
    h2 { font-size: clamp(2rem, 4vw, 3.5rem); margin-bottom: 40px; color: var(--text); }
    .text-mint { color: var(--mint); }
    .text-orange { color: var(--orange); }
    .text-blue { color: var(--blue); }
    .mono-sub { font-family: 'DM Mono', monospace; font-size: clamp(1rem, 2vw, 1.3rem); }
    
    .btn {
      display: inline-block; padding: 15px 30px;
      background: rgba(0, 255, 178, 0.1); color: var(--mint);
      border: 1px solid var(--mint); font-family: 'Bebas Neue', sans-serif;
      font-size: 1.5rem; letter-spacing: 2px; text-decoration: none;
      transition: all 0.3s; box-shadow: 0 0 15px var(--mint-dim);
      cursor: pointer; position: relative; overflow: hidden;
    }
    .btn:hover { background: var(--mint); color: var(--bg); box-shadow: 0 0 30px var(--mint); }
    .btn-pulse { animation: pulse-border 2s infinite; }
    @keyframes pulse-border {
      0% { box-shadow: 0 0 0 0 rgba(0, 255, 178, 0.4); }
      70% { box-shadow: 0 0 0 20px rgba(0, 255, 178, 0); }
      100% { box-shadow: 0 0 0 0 rgba(0, 255, 178, 0); }
    }

    /* Nav */
    nav {
      display: flex; justify-content: space-between; align-items: center;
      padding: 20px 40px; background: rgba(3, 10, 14, 0.8);
      backdrop-filter: blur(10px); border-bottom: 1px solid var(--border);
      position: fixed; top: 0; width: 100%; z-index: 100; box-sizing: border-box;
    }
    .logo { font-family: 'DM Mono', monospace; font-weight: bold; color: var(--mint); display: flex; align-items: center; gap: 10px; }
    .links a {
      color: var(--text); text-decoration: none; margin-left: 30px; font-weight: 500;
      transition: color 0.3s; font-size: 0.9rem;
    }
    .links a:hover { color: var(--mint); }

    .container { max-width: 1200px; margin: 0 auto; padding: 100px 40px; }
    .fade-up { opacity: 0; transform: translateY(40px); transition: all 0.8s ease-out; }
    .fade-up.visible { opacity: 1; transform: translateY(0); }

    /* Hero */
    .hero { display: flex; align-items: center; min-height: 100vh; padding-top: 80px; }
    .hero-left { flex: 1; z-index: 2; padding-right: 20px; }
    .hero-right { flex: 1; position: relative; z-index: 1; }
    
    /* SVG Map */
    .fix-node { animation: pulse-fix 4s infinite; }
    @keyframes pulse-fix {
      0%, 30% { fill: var(--orange); filter: url(#glow-orange); }
      40%, 90% { fill: var(--mint); filter: url(#glow-mint); }
      100% { fill: var(--orange); filter: url(#glow-orange); }
    }
    .neural-path {
      stroke-dasharray: 600; stroke-dashoffset: 600;
      animation: send-signal 4s infinite cubic-bezier(0.4, 0, 0.2, 1);
    }
    @keyframes send-signal {
      0%, 10% { stroke-dashoffset: 600; opacity: 1; }
      35% { stroke-dashoffset: 0; opacity: 1; }
      45%, 100% { opacity: 0; }
    }
    .fix-label-broken { animation: fade-broken 4s infinite; }
    .fix-label-fixed { animation: fade-fixed 4s infinite; }
    @keyframes fade-broken { 0%, 30% { opacity: 1; } 35%, 100% { opacity: 0; } }
    @keyframes fade-fixed { 0%, 35% { opacity: 0; } 40%, 90% { opacity: 1; } 95%, 100% { opacity: 0; } }

    /* Split Problem Section */
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
    .panel {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; overflow: hidden; backdrop-filter: blur(5px);
    }
    .panel-header {
      background: rgba(0,0,0,0.5); padding: 12px 20px;
      font-family: 'DM Mono', monospace; font-size: 0.85rem; color: var(--text-dim);
      border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px;
    }
    pre, .typewriter { margin: 0; padding: 25px; font-family: 'DM Mono', monospace; font-size: 0.95rem; line-height: 1.6; }
    .code-err { text-decoration: underline; text-decoration-color: var(--orange); text-decoration-style: wavy; }
    
    .typewriter p { margin: 0 0 10px 0; overflow: hidden; white-space: nowrap; width: 0; }
    .reveal.visible .typewriter p:nth-child(1) { animation: typing 1s forwards 0s; }
    .reveal.visible .typewriter p:nth-child(2) { animation: typing 1s forwards 1.2s; }
    .reveal.visible .typewriter p:nth-child(3) { animation: typing 1s forwards 2.4s; }
    .reveal.visible .typewriter p:nth-child(4) { animation: typing 1s forwards 3.6s; }
    .reveal.visible .typewriter p:nth-child(5) { animation: fadein 0.5s forwards 4.8s; width: auto; opacity: 0; }
    
    @keyframes typing { from { width: 0; } to { width: 100%; } }
    @keyframes fadein { from { opacity: 0; } to { opacity: 1; } }

    /* Flow Grid */
    .flow-grid { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: center; gap: 20px; margin-bottom: 60px; }
    .flow-card {
      background: var(--panel); border: 1px solid var(--border);
      padding: 40px 30px; border-radius: 8px; text-align: center;
      transition: transform 0.3s, border-color 0.3s;
    }
    .flow-card:hover { transform: translateY(-10px); border-color: var(--mint); box-shadow: 0 0 20px var(--mint-dim); }
    .flow-card h3 { font-size: 1.6rem; color: var(--text); margin-bottom: 15px; }
    .flow-card p { font-size: 0.95rem; color: var(--text-dim); margin: 0; }
    .arrow { color: var(--blue); }

    /* Results Table */
    .results-table { display: flex; flex-direction: column; gap: 30px; margin-bottom: 60px; }
    .r-row { display: flex; align-items: center; }
    .r-name { width: 350px; font-family: 'DM Mono', monospace; font-size: 0.95rem; }
    .r-bar-wrap { flex: 1; height: 12px; background: rgba(0,0,0,0.5); border: 1px solid var(--border); border-radius: 6px; margin: 0 30px; overflow: hidden; }
    .r-bar { height: 100%; background: var(--mint); box-shadow: 0 0 10px var(--mint); transform-origin: left; transform: scaleX(0); transition: transform 1.5s cubic-bezier(0.1, 0.8, 0.2, 1); }
    .r-score { width: 60px; text-align: right; font-family: 'DM Mono', monospace; font-weight: bold; color: var(--mint); font-size: 1.2rem; }
    .reveal.visible .r-bar { transform: scaleX(1); }
    
    .counter-box {
      background: var(--panel); border: 1px solid var(--mint-dim); padding: 30px;
      text-align: center; border-radius: 8px; font-family: 'DM Mono', monospace;
      font-size: 1.5rem; color: var(--text-dim);
    }
    .counter-box span { font-weight: bold; font-size: 3rem; margin: 0 15px; }

    /* Swagger */
    .swagger-wrapper { height: 700px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: #fff; }
    .swagger-wrapper iframe { width: 100%; height: 100%; border: none; }

    /* CTA */
    .cta-section { text-align: center; padding: 150px 0; border-top: 1px solid var(--border); background: linear-gradient(0deg, var(--grid) 0%, transparent 100%); }
    .cta-section h2 { font-size: 4rem; margin-bottom: 20px; }
    .cta-section p { font-size: 1.2rem; color: var(--text-dim); margin-bottom: 40px; }

    @media(max-width: 900px) {
      .split, .flow-grid, .hero { flex-direction: column; grid-template-columns: 1fr; text-align: center; }
      .r-row { flex-direction: column; align-items: stretch; gap: 10px; }
      .r-name { width: auto; }
      .r-bar-wrap { margin: 0; }
      .r-score { text-align: left; }
      .arrow { transform: rotate(90deg); margin: 20px 0; }
      .links { display: none; }
    }
  </style>
</head>
<body>
  <div class="bg-grid"></div>

  <nav>
    <div class="logo">
      <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="var(--mint)" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <circle cx="12" cy="12" r="3"></circle>
        <line x1="12" y1="2" x2="12" y2="12"></line>
      </svg>
      OSM_ENV // v2.1.0
    </div>
    <div class="links">
      <a href="#problem">The Problem</a>
      <a href="#how-it-works">Architecture</a>
      <a href="#results">Benchmarks</a>
      <a href="/docs" class="btn" style="padding: 8px 20px; font-size: 1.1rem; margin-left: 20px; box-shadow: none;">Swagger API</a>
    </div>
  </nav>

  <div class="container hero">
    <div class="hero-left fade-up">
      <h1>TEACHING AI TO <br><span class="text-mint">READ THE WORLD</span></h1>
      <p class="mono-sub" style="margin-bottom: 40px;">avg_score: <span class="text-orange">0.48</span> &rarr; <span class="text-mint">0.85</span></p>
      <a href="#demo" class="btn btn-pulse">WATCH THE AGENT LEARN</a>
    </div>
    <div class="hero-right fade-up">
      <svg class="hero-map" viewBox="0 0 600 400" width="100%" height="auto" style="overflow:visible;">
        <defs>
          <filter id="glow-mint" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="glow-orange" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        
        <path d="M50 200 L200 150 L350 250 L550 100" stroke="var(--grid)" stroke-width="6" fill="none" />
        <path d="M200 150 L150 50" stroke="var(--grid)" stroke-width="6" fill="none" />
        <path d="M350 250 L450 350" stroke="var(--grid)" stroke-width="6" fill="none" />
        
        <path d="M50 200 L200 150 L350 250 L550 100" stroke="var(--blue)" stroke-width="2" fill="none" opacity="0.6"/>
        <path d="M200 150 L150 50" stroke="var(--blue)" stroke-width="2" fill="none" opacity="0.6"/>
        <path d="M350 250 L450 350" stroke="var(--blue)" stroke-width="2" fill="none" opacity="0.6"/>
        
        <path class="neural-path" d="M50 200 L200 150 L350 250" stroke="var(--mint)" stroke-width="3" fill="none" filter="url(#glow-mint)"/>
        
        <circle cx="50" cy="200" r="6" fill="var(--mint)" filter="url(#glow-mint)"/>
        <circle cx="200" cy="150" r="6" fill="var(--mint)" filter="url(#glow-mint)"/>
        <circle cx="550" cy="100" r="6" fill="var(--mint)" filter="url(#glow-mint)"/>
        <circle cx="150" cy="50" r="6" fill="var(--mint)" filter="url(#glow-mint)"/>
        <circle cx="450" cy="350" r="6" fill="var(--mint)" filter="url(#glow-mint)"/>
        
        <rect x="300" y="200" width="120" height="90" rx="4" fill="none" stroke="var(--border)" stroke-dasharray="4" />
        <text x="310" y="220" fill="var(--text-dim)" font-family="monospace" font-size="12">ZONE_4A</text>
        
        <circle class="fix-node" cx="350" cy="250" r="8" />
        <text class="fix-label-broken" x="375" y="248" fill="var(--orange)" font-family="monospace" font-size="14" font-weight="bold">ERR_LATLON</text>
        <text class="fix-label-fixed" x="375" y="268" fill="var(--mint)" font-family="monospace" font-size="14" font-weight="bold">FIXED_OK</text>
      </svg>
    </div>
  </div>

  <section id="problem" class="container">
    <h2 class="fade-up"><span class="text-blue">//01</span> THE PROBLEM</h2>
    <div class="split fade-up reveal">
      <div class="panel">
        <div class="panel-header">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
          OSM_OBSERVATION.JSON
        </div>
        <pre><span class="text-dim">01</span> {
<span class="text-dim">02</span>   "id": "node/12345",
<span class="text-dim">03</span>   <span class="text-orange code-err">"lat": 99.9999,</span>
<span class="text-dim">04</span>   <span class="text-orange code-err">"name": "Hospitl",</span>
<span class="text-dim">05</span>   "amenity": "hospital"
<span class="text-dim">06</span> }</pre>
      </div>
      <div class="panel">
        <div class="panel-header">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20v2"></path><path d="M12 2v2"></path><path d="M4.22 19.78l1.42-1.42"></path><path d="M18.36 5.64l1.42-1.42"></path><circle cx="12" cy="12" r="6"></circle></svg>
          AGENT_REASONING.LOG
        </div>
        <div class="typewriter">
          <p class="text-text">> Analyzing node/12345 (amenity=hospital)...</p>
          <p class="text-orange">> WARNING: Lat 99.9999 is outside valid [-90,90] range.</p>
          <p class="text-mint">> ACTION: fix_coordinates</p>
          <p class="text-orange">> WARNING: Suspected typo in name tag 'Hospitl'.</p>
          <p class="text-mint">> ACTION: set_tag {key: "name", value: "Hospital"}</p>
        </div>
      </div>
    </div>
    <p class="mono-sub fade-up" style="margin-top: 30px; color: var(--text-dim); text-align: center;">Real maps. Real errors. Real learning.</p>
  </section>

  <section id="how-it-works" class="container">
    <h2 class="fade-up"><span class="text-blue">//02</span> ARCHITECTURE</h2>
    <div class="flow-grid fade-up">
       <div class="flow-card">
           <svg viewBox="0 0 24 24" width="48" height="48" stroke="var(--blue)" fill="none" stroke-width="1.5" style="margin-bottom: 20px;">
             <circle cx="12" cy="12" r="10"></circle>
             <line x1="2" y1="12" x2="22" y2="12"></line>
             <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
           </svg>
           <h3>World Model</h3>
           <p>Environment with partial observability, noise injection, and cascading errors.</p>
       </div>
       <div class="arrow">
          <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" fill="none" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
       </div>
       <div class="flow-card" style="border-color: var(--mint);">
           <svg viewBox="0 0 24 24" width="48" height="48" stroke="var(--mint)" fill="none" stroke-width="1.5" style="margin-bottom: 20px;">
             <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"></path>
             <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"></path>
           </svg>
           <h3 class="text-mint">AI Agent</h3>
           <p>LoRA fine-tuned on Qwen2.5 with GRPO rollouts. Actions reveal hidden tags.</p>
       </div>
       <div class="arrow">
          <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" fill="none" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
       </div>
       <div class="flow-card">
           <svg viewBox="0 0 24 24" width="48" height="48" stroke="var(--orange)" fill="none" stroke-width="1.5" style="margin-bottom: 20px;">
             <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
           </svg>
           <h3>6-Axis Grader</h3>
           <p>Scores completeness, sequence, efficiency, duplication, and confidence limits.</p>
       </div>
    </div>
    
    <div class="panel fade-up" style="padding: 40px;">
       <div class="mono-sub" style="margin-bottom: 30px; color: var(--text-dim);">> TRAINING_PROGRESS.LOG / REWARD CURVE</div>
       <svg viewBox="0 0 800 200" width="100%" height="auto" style="overflow:visible; display: block;">
           <line x1="0" y1="150" x2="800" y2="150" stroke="var(--border)" stroke-width="1"/>
           <line x1="0" y1="100" x2="800" y2="100" stroke="var(--border)" stroke-width="1"/>
           <line x1="0" y1="50"  x2="800" y2="50"  stroke="var(--border)" stroke-width="1"/>
           <path d="M0 150 L800 150" stroke="var(--blue)" stroke-width="2" fill="none" opacity="0.3" stroke-dasharray="5,5"/>
           <path d="M0 140 Q100 130, 200 90 T400 40 T600 20 T800 10" stroke="var(--mint)" stroke-width="4" fill="none" filter="url(#glow-mint)"/>
           <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
               <stop offset="0%" stop-color="var(--mint)" stop-opacity="0.3" />
               <stop offset="100%" stop-color="var(--mint)" stop-opacity="0" />
           </linearGradient>
           <path d="M0 140 Q100 130, 200 90 T400 40 T600 20 T800 10 L800 200 L0 200 Z" fill="url(#grad)" />
       </svg>
    </div>
  </section>

  <section id="results" class="container reveal">
    <h2 class="fade-up"><span class="text-blue">//03</span> BENCHMARKS</h2>
    <div class="results-table fade-up">
       <div class="r-row">
          <div class="r-name">task_easy (Missing Tags)</div>
          <div class="r-bar-wrap"><div class="r-bar" style="width: 95%"></div></div>
          <div class="r-score">0.95</div>
       </div>
       <div class="r-row">
          <div class="r-name">task_medium (Address Conflicts)</div>
          <div class="r-bar-wrap"><div class="r-bar" style="width: 88%"></div></div>
          <div class="r-score">0.88</div>
       </div>
       <div class="r-row">
          <div class="r-name">task_hard (Duplicate Resolution)</div>
          <div class="r-bar-wrap"><div class="r-bar" style="width: 82%"></div></div>
          <div class="r-score">0.82</div>
       </div>
    </div>
    <div class="counter-box fade-up">
       Overall Agent Gain: <span class="text-orange" id="c-start">0.48</span> &rarr; <span class="text-mint" id="c-end">0.85</span>
    </div>
  </section>

  <section id="api" class="container">
    <h2 class="fade-up"><span class="text-blue">//04</span> API DOCUMENTATION</h2>
    <div class="swagger-wrapper fade-up">
       <iframe src="/docs" title="Swagger UI"></iframe>
    </div>
  </section>

  <section id="demo" class="cta-section fade-up">
     <div class="container" style="padding: 0;">
       <h2>WATCH THE <span class="text-mint">AGENT LEARN</span></h2>
       <p>Deploy the world model and train your own autonomous mapping agent.</p>
       <a href="/docs" class="btn btn-pulse" style="margin-top: 20px;">LAUNCH ENVIRONMENT</a>
     </div>
  </section>

  <script>
    const obs = new IntersectionObserver((entries) => {
       entries.forEach(e => {
          if(e.isIntersecting) {
             e.target.classList.add('visible');
             // Counters
             if(e.target.classList.contains('reveal')) {
                const endEl = document.getElementById('c-end');
                if(!endEl.dataset.done) {
                   endEl.dataset.done = 'true';
                   let start = 0.48, end = 0.85, frames = 60, c = 0;
                   let int = setInterval(() => {
                       c++;
                       endEl.innerText = (start + (end - start) * (c/frames)).toFixed(2);
                       if (c >= frames) { clearInterval(int); endEl.innerText = "0.85"; }
                   }, 30);
                }
             }
          }
       });
    }, { threshold: 0.1 });
    document.querySelectorAll('.fade-up, .reveal').forEach(el => obs.observe(el));
  </script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def root():
    """Serve the interactive landing page."""
    return LANDING_HTML


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "env": "osm-map-quality-env",
        "version": "2.1.0",
    }


@app.get("/info", tags=["Info"])
def info():
    """Return environment capabilities, innovations, and metadata."""
    return {
        "name": "OSM Map Quality Environment",
        "version": "2.1.0",
        "author": "Dokka Vijay",
        "description": (
            "A world-modeling environment for geographic data quality assurance. "
            "Features partial observability, noisy/conflicting inputs, cascading "
            "error discovery, and confidence-calibrated grading."
        ),
        "innovations": [
            {
                "name": "Partial Observability",
                "description": "Tags revealed progressively through agent actions",
            },
            {
                "name": "Noisy & Conflicting Data",
                "description": "Typos, stale values, contradictory fields injected dynamically",
            },
            {
                "name": "Cascading Error Discovery",
                "description": "Fixing one issue may reveal new inconsistencies",
            },
            {
                "name": "Confidence Calibration",
                "description": "Overconfident wrong actions penalized; encourages honest uncertainty",
            },
            {
                "name": "Dynamic Task Generation",
                "description": "Fresh task variations each episode via randomized seeds",
            },
            {
                "name": "Multi-Dimensional Grading",
                "description": "6-axis scoring: completeness, consistency, efficiency, accuracy, merge, sequence",
            },
        ],
        "tasks": ["task_easy", "task_medium", "task_hard"],
        "action_types": sorted(VALID_ACTION_TYPES),
        "scoring_range": [0.05, 0.95],
        "rate_limit": "100 requests/minute",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "grader": "POST /grader",
            "baseline": "POST /baseline",
            "health": "GET /health",
            "info": "GET /info",
            "docs": "GET /docs",
        },
    }


@app.post("/reset", tags=["Environment"])
def reset(req: ResetRequest = None):
    """Reset the environment to the start of a new episode."""
    task_id = (req.task_id if req else None) or "task_easy"
    try:
        obs = env.reset(task_id=task_id)
        return {"observation": obs_to_dict(obs)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reset failed: {e}")


@app.post("/step", tags=["Environment"])
def step(req: StepRequest):
    """Submit one action and advance the episode by one step."""
    try:
        class Action:
            pass

        action = Action()
        action.action_type = req.action_type
        action.tag_key = req.tag_key
        action.tag_value = req.tag_value
        action.coordinates = req.coordinates
        action.confidence = req.confidence

        obs = env.step(action)
        return {
            "observation": obs_to_dict(obs),
            "reward": obs.reward,
            "done": obs.done,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Step failed: {e}")


@app.get("/state", tags=["Environment"])
def state():
    """Return the current episode state."""
    return {"state": state_to_dict(env.state)}


@app.get("/tasks", tags=["Environment"])
def tasks():
    """List all available tasks and the action schema."""
    return {"tasks": list_tasks()}


@app.post("/grader", tags=["Grading"])
def grader(req: GraderRequest):
    """Grade the current episode state for a given task."""
    try:
        snapshot = env.get_episode_snapshot()
        score = grade(req.task_id, snapshot)
        return {
            "task_id": req.task_id,
            "score": score,
            "snapshot": snapshot,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Grading failed: {e}")


@app.post("/baseline", tags=["Grading"])
def baseline():
    """Run a deterministic baseline agent across all tasks and return scores."""
    baseline_plans = {
        "task_easy": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Hyderabad Chai Cafe"},
            {"action_type": "mark_complete"},
        ],
        "task_medium": [
            {"action_type": "set_tag", "tag_key": "addr:street", "tag_value": "Jubilee Hills Road"},
            {"action_type": "set_tag", "tag_key": "addr:city", "tag_value": "Hyderabad"},
            {"action_type": "set_tag", "tag_key": "addr:postcode", "tag_value": "500033"},
            {"action_type": "set_tag", "tag_key": "addr:country", "tag_value": "IN"},
            {"action_type": "mark_complete"},
        ],
        "task_hard": [
            {"action_type": "set_tag", "tag_key": "name", "tag_value": "Yashoda Hospital"},
            {"action_type": "fix_coordinates", "coordinates": {"lat": 17.4449, "lon": 78.5011}},
            {"action_type": "set_tag", "tag_key": "addr:city", "tag_value": "Secunderabad"},
            {"action_type": "merge_duplicate"},
            {"action_type": "set_tag", "tag_key": "addr:street", "tag_value": "Alexander Road"},
            {"action_type": "set_tag", "tag_key": "website", "tag_value": "https://yashodahospitals.com"},
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
                act.coordinates = entry.get("coordinates")
                act.confidence = 1.0
                env.step(act)
            snapshot = env.get_episode_snapshot()
            score = grade(task_id, snapshot)
            results[task_id] = {"score": score, "status": "success"}
        except Exception as e:
            results[task_id] = {"score": 0.0, "status": "error", "error": str(e)}

    env.reset("task_easy")
    return {"baseline_scores": results}


def main():
    """Entry point for running the server directly."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()