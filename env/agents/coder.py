import os
from openai import AsyncOpenAI
from .base import BaseAgent, AgentInput, AgentOutput

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CoderAgent(BaseAgent):
    name = "Coder"

    async def run(self, inp: AgentInput) -> AgentOutput:
        resp = await _client.chat.completions.create(
            model    = os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages = [
                {"role": "system", "content":
                    "You are a senior Python engineer. Given a spec, write a clean, "
                    "production-ready Python function with type hints and docstring. "
                    "Return ONLY the Python code block, nothing else."},
                {"role": "user", "content": inp.content},
            ],
            temperature = 0.2,
            max_tokens  = 800,
        )
        return AgentOutput(agent=self.name, content=resp.choices[0].message.content)