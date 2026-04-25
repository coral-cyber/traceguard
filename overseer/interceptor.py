# overseer/interceptor.py
import uuid, time
from db.models import Trace, ContainmentEvent
from db.store import save_trace, save_containment
from overseer.scorer import score
from overseer.policy import get_containment_cfg
from overseer.remediator import Remediator


class Interceptor:
    """
    Sits between every agent handoff.
    Called twice per agent: once on input, once on output.
    """

    def __init__(self, run_id: str, containment):
        self.run_id      = run_id
        self.containment = containment   # ContainmentEngine singleton from app.state

    async def intercept(self, agent_name: str, action_type: str, payload: dict) -> dict:
        t0       = time.perf_counter()
        text     = payload.get("content", "")
        result   = score(text)
        cfg      = get_containment_cfg()
        trace_id = str(uuid.uuid4())
        elapsed  = int((time.perf_counter() - t0) * 1000)

        threat = {
            "score":  result.score,
            "flags":  result.flags,
            "reason": result.reason,
        }

        is_threat = result.score >= cfg["quarantine_threshold"]
        contained_event = None

        # Save trace to DB
        await save_trace(Trace(
            run_id       = self.run_id,
            agent_name   = agent_name,
            action_type  = action_type,
            payload      = {"preview": text[:300]},
            threat_score = result.score,
            threat_flags = result.flags,
            contained    = is_threat,
        ))

        if is_threat:
            contained_event = await self.containment.contain(
                run_id   = self.run_id,
                trace_id = trace_id,
                agent    = agent_name,
                result   = threat,
                elapsed  = elapsed,
            )
            remediator = Remediator()
            remediation = remediator.remediate(agent_name, payload, result)

            if remediation["safe"]:
               payload["content"] = remediation["content"]

        return {
            "allowed":     remediation["safe"],
            "threat":      threat,
            "containment": contained_event,
            "remediation": remediation,
            "latency_ms":  result.latency_ms,
        }