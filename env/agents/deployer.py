import os, hashlib
from datetime import datetime
from .base import BaseAgent, AgentInput, AgentOutput

class DeployerAgent(BaseAgent):
    name = "Deployer"

    async def run(self, inp: AgentInput) -> AgentOutput:
        code_hash = hashlib.sha256(inp.content.encode()).hexdigest()[:12]
        deploy_id = f"deploy-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{code_hash}"
        # Mock prod deploy — write to outputs/
        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/{deploy_id}.py", "w") as f:
            f.write(inp.content)
        return AgentOutput(
            agent   = self.name,
            content = f"✓ Deployed to mock-prod as {deploy_id}.py",
            meta    = {"deploy_id": deploy_id, "hash": code_hash},
        )