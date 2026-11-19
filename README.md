# PRD Test-Case Agent

A compact, inspectable implementation of a PRD-to-test-case **multi-agent** system. It follows the useful ideas from the reference project while keeping the design local and vendor-neutral:

1. Parse a PRD into source chunks.
2. Run independent extractors for requirements, user journeys, scope, and risks.
3. Retrieve only relevant reference material from a small local knowledge base.
4. Generate structured test cases with traceability to PRD evidence.
5. Validate output with Pydantic and reject unsupported claims.

## Architecture

```text
                         ┌─ Requirement agent ─┐
PRD ── Root agent ───────┼─ Journey agent ─────┼──► Research/RAG agent
                         ├─ Scope agent ───────┤            │
                         └─ Risk agent ────────┘            ▼
                                      (parallel)     Test-case agent
                                                               │
                                              ┌────────────────┘
                                              ▼
                                     Critic agent → Refinement agent
                                              │          │
                                              └─ retry once ─┘
```

`TestStrategyRootAgent` runs the four independent discovery agents in parallel, then RAG, generation, critique, and a bounded correction loop. Every agent reads and writes the typed `RunState` contract in `agents.py`.

The default mode is deterministic, so you can study the orchestration without an API key. The optional DeepSeek provider makes every specialist a structured-output LLM agent; it uses DeepSeek's Responses API, JSON Schema output, and Pydantic validation.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m prd_test_agent.main --prd sample_prd.md --knowledge-dir knowledge --out test_cases.json
```

To run the LLM-agent version after setting an API key:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
python -m prd_test_agent.main --provider deepseek --model deepseek-v4-flash --prd sample_prd.md --knowledge-dir knowledge --out test_cases.json
```

Use `--max-output-tokens 16000` to set a generated-token cap shared across every DeepSeek agent call in one CLI run. The final JSON includes measured input/output/total token usage and number of model calls. The allocator is role-aware: generation receives a larger allowance than short extraction tasks. Input tokens are reported but cannot be strictly capped before a request because the provider meters them after it receives the prompt.

## Web UI

The Streamlit UI is the recommended way to use the project. It supports PRD upload/paste, optional reference-document upload, DeepSeek or demo mode, a per-run output-token cap, measured token usage, reviewable test cases, a requirement-to-case matrix, open questions, JSON download, and a downloadable per-agent run log.

```powershell
streamlit run app.py
```

The UI uses an entered API key only for the active request; it does not write it to a project file.

The command writes a validated JSON test suite. Each test case includes `requirement_ids` and `evidence`, which make review and debugging much easier than free-form LLM output. The critic rejects unknown requirement IDs and missing evidence; the root agent allows at most one correction round to keep cost and latency bounded.

## Where RAG belongs

Put approved test standards, product constraints, and historical test examples in `knowledge/`. The retriever uses a transparent lexical score for learning. In production, replace `LocalRetriever` with embeddings + a vector database, retain metadata (`source`, `version`, `owner`), and require citations.

Do **not** use RAG as a substitute for unclear product requirements. Mark ambiguity as a question or an assumption, then ask a human to resolve it.

## Production path

- Add an LLM adapter that requests JSON matching `TestSuite` exactly.
- Use structured outputs/tool calling; never parse prose with regex.
- Add authentication, tenant isolation, prompt-injection filtering, and document access control before accepting untrusted PRDs.
- Evaluate with a labeled PRD set: requirement coverage, unsupported-claim rate, duplicate rate, and reviewer acceptance rate.
