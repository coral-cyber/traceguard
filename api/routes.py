"""
API routes for TraceGuard.

Key fixes vs original:
- ContainmentEngine pulled from app.state (singleton, not re-instantiated per request)
- attack_mode is instance-level on ReviewerAgent (no cross-request bleed)
- Admin endpoints: quarantine release, engine stats
- Proper 422 handling on bad input
"""
import asyncio, uuid
from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from typing import Literal
import json

from agents.coder    import CoderAgent
from agents.reviewer import ReviewerAgent, ATTACK_MODES
from agents.tester   import TesterAgent
from agents.deployer import DeployerAgent
from agents.base     import AgentInput
from overseer.interceptor import Interceptor
from db.store import get_metrics, get_run_traces

router = APIRouter()


# ── Request models ─────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    spec: str = Field(..., min_length=5, max_length=2000)
    attack_mode: Literal["none", "obvious", "collusion", "ssrf_social", "comment_inject"] = "none"


# ── Pipeline ───────────────────────────────────────────────────────────────────

@router.post("/run")
async def run_pipeline(req: PipelineRequest, request: Request):
    containment = request.app.state.containment   # singleton from lifespan
    run_id      = str(uuid.uuid4())
    interceptor = Interceptor(run_id=run_id, containment=containment)

    # Instance-level attack — no class mutation, no cross-request bleed
    attack_mode = None if req.attack_mode == "none" else req.attack_mode
    agents = [
        CoderAgent(),
        ReviewerAgent(attack_mode=attack_mode),
        TesterAgent(),
        DeployerAgent(),
    ]

    payload      = {"content": req.spec}
    pipeline_log = []

    for agent in agents:
        if containment.is_quarantined(agent.name):
            pipeline_log.append({
                "agent":  agent.name,
                "status": "QUARANTINED",
                "skipped": True,
            })
            continue

        # Intercept input BEFORE agent runs
        pre = await interceptor.intercept(agent.name, "input", payload)
        if not pre["allowed"]:
            pipeline_log.append({
                "agent":  agent.name,
                "status": "BLOCKED_INPUT",
                "threat": pre["threat"],
                "event":  pre["containment"],
            })
            break

        # Run agent
        try:
            out = await agent.run(AgentInput(run_id=run_id, content=payload["content"]))
        except Exception as e:
            pipeline_log.append({"agent": agent.name, "status": "ERROR", "error": str(e)})
            break

        # Intercept output BEFORE handing to next agent
        post = await interceptor.intercept(agent.name, "output", {"content": out.content})
        if not post["allowed"]:
            pipeline_log.append({
                "agent":          agent.name,
                "status":         "BLOCKED_OUTPUT",
                "threat":         post["threat"],
                "event":          post["containment"],
                "output_preview": out.content[:200],
            })
            break

        pipeline_log.append({
            "agent":          agent.name,
            "status":         "OK",
            "threat_score":   post["threat"]["score"],
            "threat_flags":   post["threat"]["flags"],
            "output_preview": out.content[:300],
        })
        payload = {"content": out.content}

    return {
        "run_id":      run_id,
        "pipeline":    pipeline_log,
        "attack_mode": req.attack_mode,
        "completed":   all(s["status"] == "OK" for s in pipeline_log if not s.get("skipped")),
    }


# ── Metrics & traces ───────────────────────────────────────────────────────────

@router.get("/metrics")
async def metrics():
    return await get_metrics()

@router.get("/traces/{run_id}")
async def traces(run_id: str):
    rows = await get_run_traces(run_id)
    return [
        {
            "agent":        r.agent_name,
            "action":       r.action_type,
            "threat_score": r.threat_score,
            "flags":        r.threat_flags,
            "contained":    r.contained,
            "timestamp":    r.timestamp.isoformat(),
        }
        for r in rows
    ]


# ── Admin ──────────────────────────────────────────────────────────────────────

@router.get("/admin/containment")
async def containment_stats(request: Request):
    return request.app.state.containment.stats()

@router.post("/admin/release/{agent}")
async def release_quarantine(agent: str, request: Request):
    await request.app.state.containment.release_quarantine(agent)
    return {"released": agent}

@router.get("/attack-modes")
async def attack_modes():
    return {"modes": list(ATTACK_MODES.keys())}


# ── SSE stream ─────────────────────────────────────────────────────────────────

@router.get("/stream")
async def stream(request: Request):
    """
    Server-Sent Events — dashboard subscribes here.
    Each THREAT_CONTAINED event is pushed within milliseconds of detection.
    """
    bus = request.app.state.event_bus

    async def generator():
        yield {"data": json.dumps({"event": "CONNECTED"})}
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(bus.get(), timeout=25)
                yield {"data": json.dumps(event)}
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"event": "PING"})}

    return EventSourceResponse(generator())
