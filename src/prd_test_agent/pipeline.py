import asyncio
from pathlib import Path

from .agents import (CriticAgent, JourneyAgent, RefinementAgent, RequirementAgent,
                     ResearchAgent, RiskAgent, RunState, ScopeAgent, StructuredModel,
                     TestCaseAgent, log)
from .models import TestSuite


class RunFailed(RuntimeError):
    def __init__(self, message: str, logs, usage=None):
        super().__init__(message)
        self.logs = logs
        self.usage = usage


class TestStrategyRootAgent:
    """Orchestrates parallel discovery, RAG, generation, critique, and refinement."""

    def __init__(self, model: StructuredModel | None = None, max_correction_rounds: int = 1):
        self.model, self.max_correction_rounds = model, max_correction_rounds

    async def run(self, prd_path: Path, knowledge_dir: Path) -> TestSuite:
        state = RunState(prd=prd_path.read_text(encoding="utf-8"), prd_source=prd_path.name, knowledge_dir=knowledge_dir)
        log(state, "root", "Started test-strategy run.")
        try:
            # Independent specialists run concurrently against the same immutable PRD.
            await asyncio.gather(*(agent.run(state, self.model) for agent in (RequirementAgent(), JourneyAgent(), ScopeAgent(), RiskAgent())))
            log(state, "root", "Parallel discovery phase completed.")
            await ResearchAgent().run(state, self.model)
            generator, critic, refinement = TestCaseAgent(), CriticAgent(), RefinementAgent()
            for round_number in range(self.max_correction_rounds + 1):
                await generator.run(state, self.model)
                await critic.run(state, self.model)
                needs_retry = await refinement.run(state, self.model)
                if not needs_retry or round_number == self.max_correction_rounds:
                    break
        except Exception as error:
            usage = self.model.usage_summary() if self.model and hasattr(self.model, "usage_summary") else None
            log(state, "root", f"Run failed: {type(error).__name__}: {error}", "ERROR")
            raise RunFailed(str(error), state.logs, usage) from error
        usage = self.model.usage_summary() if self.model and hasattr(self.model, "usage_summary") else None
        log(state, "root", f"Run completed with {len(state.test_cases)} test cases.")
        return TestSuite(prd_source=str(prd_path), requirements=state.requirements, test_cases=state.test_cases, open_questions=state.open_questions, usage=usage, run_logs=state.logs)


async def build_suite(prd_path: Path, knowledge_dir: Path, model: StructuredModel | None = None) -> TestSuite:
    return await TestStrategyRootAgent(model=model).run(prd_path, knowledge_dir)
