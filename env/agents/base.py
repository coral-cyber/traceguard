from abc import ABC, abstractmethod
from pydantic import BaseModel

class AgentInput(BaseModel):
    run_id:  str
    content: str
    meta:    dict = {}

class AgentOutput(BaseModel):
    agent:   str
    content: str
    meta:    dict = {}

class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, inp: AgentInput) -> AgentOutput:
        ...