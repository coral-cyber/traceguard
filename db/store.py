from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, distinct
from .models import Base, Trace, ContainmentEvent
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./traceguard.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def save_trace(trace: Trace):
    async with AsyncSessionLocal() as s:
        s.add(trace)
        await s.commit()
        await s.refresh(trace)

async def save_containment(ev: ContainmentEvent):
    async with AsyncSessionLocal() as s:
        s.add(ev)
        await s.commit()

async def load_quarantine_state() -> set[str]:
    """
    Returns agents that have ever been quarantined (not just rolled back).
    Used at startup to restore containment state across restarts.
    """
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(distinct(ContainmentEvent.agent_name))
            .where(ContainmentEvent.action == "quarantine")
        )
        return {row[0] for row in result.fetchall() if row[0]}

async def get_run_traces(run_id: str) -> list[Trace]:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(Trace).where(Trace.run_id == run_id).order_by(Trace.timestamp)
        )
        return result.scalars().all()

async def get_metrics() -> dict:
    threshold = float(os.getenv("THREAT_THRESHOLD", 0.65))
    async with AsyncSessionLocal() as s:
        total      = await s.scalar(select(func.count()).select_from(Trace)) or 0
        threats    = await s.scalar(
            select(func.count()).select_from(Trace).where(Trace.threat_score >= threshold)
        ) or 0
        contained  = await s.scalar(select(func.count()).select_from(ContainmentEvent)) or 0
        avg_score  = await s.scalar(select(func.avg(Trace.threat_score)).select_from(Trace)) or 0.0
        return {
            "total_traces":    total,
            "threats_detected": threats,
            "containments":    contained,
            "avg_threat_score": round(float(avg_score), 4),
            "detection_rate":  round(threats / max(total, 1), 4),
        }
