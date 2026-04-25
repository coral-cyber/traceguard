from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Boolean
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class Trace(Base):
    __tablename__ = "traces"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id       = Column(String, index=True)
    agent_name   = Column(String)
    action_type  = Column(String)   # "input" | "output"
    payload      = Column(JSON)
    threat_score = Column(Float, default=0.0)
    threat_flags = Column(JSON,  default=list)
    contained    = Column(Boolean, default=False)
    timestamp    = Column(DateTime, default=datetime.utcnow)

class ContainmentEvent(Base):
    __tablename__ = "containment_events"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id       = Column(String, index=True)
    trace_id     = Column(String)
    agent_name   = Column(String, index=True)   # ← was missing, needed for quarantine restore
    action       = Column(String)               # "rollback" | "quarantine"
    reason       = Column(String)
    evidence     = Column(JSON)
    resolved_ms  = Column(Integer)
    timestamp    = Column(DateTime, default=datetime.utcnow)
