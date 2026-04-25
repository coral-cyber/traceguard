"""
Containment Engine — zero-human, sub-second response.

Fixes:
1. asyncio.Lock per agent — no more read-modify-write races under concurrent requests
2. Quarantine state persisted to DB — survives restarts
3. ContainmentEngine is instantiated ONCE per app (not per request) via lifespan
4. Event bus is bounded and never blocks the hot path
"""
import asyncio, time, uuid
from datetime import datetime
from collections import defaultdict

from db.models import ContainmentEvent
from db.store import save_containment, load_quarantine_state
from overseer.policy import get_containment_cfg


class ContainmentEngine:
    """
    One instance lives for the lifetime of the FastAPI app.
    All state is protected by per-agent locks — safe under asyncio concurrency.
    """

    def __init__(self, event_bus: asyncio.Queue | None = None):
        self._event_bus = event_bus

        # Per-agent lock: only one coroutine mutates an agent's state at a time
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Protected by _locks[agent]
        self._repeat_counts: dict[str, int] = defaultdict(int)
        self._quarantined:   set[str]       = set()

    async def restore_state(self):
        """
        Call once at startup. Loads quarantined agents from DB so
        a restart doesn't silently un-quarantine a bad actor.
        """
        quarantined = await load_quarantine_state()
        self._quarantined.update(quarantined)
        if quarantined:
            print(f"[ContainmentEngine] Restored quarantine: {quarantined}")

    async def contain(
        self,
        run_id:  str,
        trace_id: str,
        agent:   str,
        result:  dict,
        elapsed: int,
    ) -> dict:
        cfg = get_containment_cfg()

        async with self._locks[agent]:
            # Now safe — no other coroutine can read-modify-write this agent's state
            self._repeat_counts[agent] += 1
            count = self._repeat_counts[agent]

            action = "rollback"
            if (
                cfg.get("quarantine_agent_on_repeat")
                and count >= cfg.get("repeat_threshold", 2)
                and agent not in self._quarantined
            ):
                action = "quarantine"
                self._quarantined.add(agent)

        ev = ContainmentEvent(
            run_id      = run_id,
            trace_id    = trace_id,
            action      = action,
            reason      = result.get("reason", ""),
            evidence    = result,
            resolved_ms = elapsed,
        )
        await save_containment(ev)

        event = {
            "event":      "THREAT_CONTAINED",
            "run_id":     run_id,
            "agent":      agent,
            "action":     action,
            "score":      result["score"],
            "flags":      result["flags"],
            "reason":     result["reason"],
            "repeat_count": count if 'count' in dir() else self._repeat_counts[agent],
            "elapsed_ms": elapsed,
            "timestamp":  datetime.utcnow().isoformat(),
        }

        if self._event_bus is not None:
            try:
                # Non-blocking put — if bus is full, drop the event rather than block hot path
                self._event_bus.put_nowait(event)
            except asyncio.QueueFull:
                pass   # Dashboard misses one event; pipeline is unaffected

        return event

    def is_quarantined(self, agent: str) -> bool:
        # Read-only check, no lock needed (set reads are atomic in CPython,
        # and worst case we re-check after acquire in contain())
        return agent in self._quarantined

    def stats(self) -> dict:
        return {
            "quarantined":    list(self._quarantined),
            "repeat_counts":  dict(self._repeat_counts),
        }

    async def release_quarantine(self, agent: str):
        """Manual override — useful for tests and admin API."""
        async with self._locks[agent]:
            self._quarantined.discard(agent)
            self._repeat_counts[agent] = 0
