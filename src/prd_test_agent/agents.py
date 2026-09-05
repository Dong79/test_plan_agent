"""Specialist agents used by the root orchestrator."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Evidence, Requirement, RunLogEntry, TestCase, TestStep


class RequirementList(BaseModel):
    requirements: list[Requirement]


class TextList(BaseModel):
    items: list[str]


class GeneratedCases(BaseModel):
    test_cases: list[TestCase]
    open_questions: list[str] = []


class CritiqueIssue(BaseModel):
    severity: str
    test_case_id: str | None = None
    message: str = Field(max_length=240)


class Critique(BaseModel):
    issues: list[CritiqueIssue] = Field(default_factory=list, max_length=12)


class StructuredModel:
    async def generate(self, *, instructions: str, input_text: str, output_type: type[BaseModel]) -> BaseModel:
        raise NotImplementedError


@dataclass
class RunState:
    prd: str
    prd_source: str
    knowledge_dir: Path
    requirements: list[Requirement] = field(default_factory=list)
    journeys: list[str] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    retrieved: list[Evidence] = field(default_factory=list)
    test_cases: list[TestCase] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    critique: Critique = field(default_factory=Critique)
    trace: list[str] = field(default_factory=list)
    logs: list[RunLogEntry] = field(default_factory=list)


def log(state: RunState, agent: str, message: str, level: str = "INFO") -> None:
    state.logs.append(RunLogEntry.create(agent=agent, message=message, level=level))  # type: ignore[arg-type]


def bullets(prd: str) -> list[tuple[str, str]]:
    heading, results = "", []
    for line in prd.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()
        elif line.strip().startswith("- "):
            results.append((heading, line.strip()[2:].strip()))
    return results


class RequirementAgent:
    name = "requirements"

    async def run(self, state: RunState, model: StructuredModel | None) -> None:
        log(state, self.name, "Started requirement extraction.")
        if model:
            result = await model.generate(instructions="Extract atomic, testable PRD requirements. Mark only content under an Out of scope heading as out_of_scope. Use exact PRD excerpts as evidence. Do not infer requirements.", input_text=state.prd, output_type=RequirementList)
            state.requirements = result.requirements  # type: ignore[assignment]
        else:
            state.requirements = [Requirement(id=f"REQ-{i:03}", text=item, kind="out_of_scope" if "out of scope" in heading else "functional", evidence=Evidence(source=state.prd_source, excerpt=item)) for i, (heading, item) in enumerate(bullets(state.prd), 1)]
        state.trace.append(self.name)
        log(state, self.name, f"Extracted {len(state.requirements)} requirements.")


class JourneyAgent:
    name = "journeys"

    async def run(self, state: RunState, model: StructuredModel | None) -> None:
        log(state, self.name, "Started critical-user-journey extraction.")
        if model:
            result = await model.generate(instructions="Extract critical user journeys from this PRD as short end-to-end statements. Do not invent details.", input_text=state.prd, output_type=TextList)
            state.journeys = result.items  # type: ignore[assignment]
        else:
            state.journeys = [item for _, item in bullets(state.prd) if any(x in item.lower() for x in ("user can", "customer can", "signed-in"))]
        state.trace.append(self.name)
        log(state, self.name, f"Identified {len(state.journeys)} critical user journeys.")


class ScopeAgent:
    name = "scope"

    async def run(self, state: RunState, model: StructuredModel | None) -> None:
        log(state, self.name, "Started scope-boundary extraction.")
        if model:
            result = await model.generate(instructions="Extract scope boundaries, exclusions, and constraints as a short list. Quote only supported statements.", input_text=state.prd, output_type=TextList)
            state.scope = result.items  # type: ignore[assignment]
        else:
            state.scope = [item for heading, item in bullets(state.prd) if "out of scope" in heading or "cannot" in item.lower()]
        state.trace.append(self.name)
        log(state, self.name, f"Identified {len(state.scope)} scope boundaries.")


class RiskAgent:
    name = "risks"

    async def run(self, state: RunState, model: StructuredModel | None) -> None:
        log(state, self.name, "Started risk analysis.")
        if model:
            result = await model.generate(instructions="Identify test-risk areas explicitly supported by this PRD: destructive actions, authorization, timing, defaults, and negative constraints. Do not invent risks.", input_text=state.prd, output_type=TextList)
            state.risks = result.items  # type: ignore[assignment]
        else:
            flags = ("delete", "alert", "notification", "default", "only", "must not")
            state.risks = [item for _, item in bullets(state.prd) if any(flag in item.lower() for flag in flags)]
        state.trace.append(self.name)
        log(state, self.name, f"Identified {len(state.risks)} PRD-backed risk signals.")


class LocalRetriever:
    def search(self, directory: Path, query: str, limit: int = 3) -> list[Evidence]:
        terms = set(re.findall(r"[a-z]{3,}", query.lower()))
        docs = [(path.name, path.read_text(encoding="utf-8")) for path in directory.glob("*.md")]
        ranked = sorted(docs, key=lambda d: len(terms & set(re.findall(r"[a-z]{3,}", d[1].lower()))), reverse=True)
        return [Evidence(source=name, excerpt=text[:800]) for name, text in ranked[:limit] if text]


class ResearchAgent:
    name = "research"

    async def run(self, state: RunState, model: StructuredModel | None) -> None:
        log(state, self.name, "Started local RAG retrieval.")
        state.retrieved = LocalRetriever().search(state.knowledge_dir, " ".join(state.journeys + state.risks))
        state.trace.append(self.name)
        log(state, self.name, f"Retrieved {len(state.retrieved)} guidance documents from the local knowledge base.")


class TestCaseAgent:
    name = "test_case_generation"

    async def run(self, state: RunState, model: StructuredModel | None) -> None:
        log(state, self.name, "Started test-case generation.")
        requirements = "\n".join(f"{r.id}: {r.text}" for r in state.requirements if r.kind != "out_of_scope")
        guidance = "\n".join(f"[{e.source}] {e.excerpt}" for e in state.retrieved)
        feedback = "\n".join(issue.message for issue in state.critique.issues)
        if model:
            result = await model.generate(instructions=("Generate execution-ready test cases only for stated requirements. Every case must cite an existing requirement ID and its exact PRD evidence. Use retrieved material only as test-design guidance; it cannot add product behavior. Create observable expected results. Return unresolved ambiguity in open_questions.\nPrior critic feedback:\n" + (feedback or "None")), input_text=f"REQUIREMENTS:\n{requirements}\n\nRETRIEVED GUIDANCE:\n{guidance}", output_type=GeneratedCases)
            state.test_cases, state.open_questions = result.test_cases, result.open_questions  # type: ignore[assignment]
        else:
            cases, questions = [], []
            for req in (r for r in state.requirements if r.kind != "out_of_scope"):
                priority = "P0" if any(x in req.text.lower() for x in ("delete", "must not", "only")) else "P1"
                cases.append(TestCase(id=f"TC-{len(cases)+1:03}", title=req.text[:100], priority=priority, type="functional", preconditions=[], steps=[TestStep(action=f"Perform: {req.text}", expected_result=f"The system satisfies this requirement: {req.text}")], requirement_ids=[req.id], evidence=[req.evidence]))
                if any(x in req.text.lower() for x in ("alert", "daily", "weekly", "immediate")):
                    questions.append(f"What test clock or event trigger should be used to verify {req.id}?")
            state.test_cases, state.open_questions = cases, questions
        state.trace.append(self.name)
        log(state, self.name, f"Generated {len(state.test_cases)} test cases and {len(state.open_questions)} open questions.")


class CriticAgent:
    name = "critic"

    async def run(self, state: RunState, model: StructuredModel | None) -> None:
        log(state, self.name, "Started quality review.")
        valid_ids = {r.id for r in state.requirements if r.kind != "out_of_scope"}
        if model:
            result = await model.generate(instructions="Act as a strict test-reviewer. Identify unsupported claims, missing requirement links, vague expected results, duplicate cases, and untested testable requirements. Return issues only. Return at most 12 highest-impact issues; combine repeated patterns; each message must be concise (under 240 characters).", input_text=f"REQUIREMENTS: {[r.model_dump() for r in state.requirements]}\nCASES: {[c.model_dump() for c in state.test_cases]}", output_type=Critique)
            state.critique = result  # type: ignore[assignment]
        else:
            issues = []
            for case in state.test_cases:
                unknown = set(case.requirement_ids) - valid_ids
                if unknown:
                    issues.append(CritiqueIssue(severity="error", test_case_id=case.id, message=f"Unknown requirement IDs: {sorted(unknown)}"))
                if not case.evidence:
                    issues.append(CritiqueIssue(severity="error", test_case_id=case.id, message="Missing PRD evidence."))
            covered = {item for case in state.test_cases for item in case.requirement_ids}
            for req_id in valid_ids - covered:
                issues.append(CritiqueIssue(severity="warning", message=f"No test case covers {req_id}."))
            state.critique = Critique(issues=issues)
        state.trace.append(self.name)
        log(state, self.name, f"Completed quality review with {len(state.critique.issues)} issue(s).", "WARNING" if state.critique.issues else "INFO")


class RefinementAgent:
    name = "refinement"

    async def run(self, state: RunState, model: StructuredModel | None) -> bool:
        log(state, self.name, "Started refinement decision.")
        invalid = {issue.test_case_id for issue in state.critique.issues if issue.severity == "error" and issue.test_case_id}
        state.test_cases = [case for case in state.test_cases if case.id not in invalid]
        state.trace.append(self.name)
        log(state, self.name, f"Removed {len(invalid)} invalid test cases; {'requesting another generation pass' if state.critique.issues else 'no correction required' }.", "WARNING" if state.critique.issues else "INFO")
        return bool(state.critique.issues)
