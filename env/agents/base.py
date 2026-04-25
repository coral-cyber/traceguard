"""
base.py — synchronous agent base classes.
All agents run sync. No async/await, no OpenAI dependency.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel


class AgentInput(BaseModel):
    run_id:  str = "local"
    content: str
    meta:    dict = {}


class AgentOutput(BaseModel):
    agent:   str
    content: str
    meta:    dict = {}

    def __str__(self) -> str:
        return self.content


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, inp: AgentInput) -> AgentOutput:
        ...
