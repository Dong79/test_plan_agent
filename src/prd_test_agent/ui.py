"""Streamlit review UI for the PRD test-strategy multi-agent system."""
import asyncio
import tempfile
from pathlib import Path

import streamlit as st

from .llm import DeepSeekResponsesModel
from .pipeline import RunFailed, build_suite


def _coverage(suite) -> tuple[int, int, float]:
    testable = {requirement.id for requirement in suite.requirements if requirement.kind != "out_of_scope"}
    covered = {requirement_id for case in suite.test_cases for requirement_id in case.requirement_ids} & testable
    return len(covered), len(testable), 100 * len(covered) / len(testable) if testable else 0.0


def _case_rows(suite) -> list[dict[str, str]]:
    return [{
        "ID": case.id,
        "Priority": case.priority,
        "Type": case.type,
        "Title": case.title,
        "Requirements": ", ".join(case.requirement_ids),
        "Steps": str(len(case.steps)),
    } for case in suite.test_cases]


def _save_upload(upload, destination: Path) -> None:
    destination.write_bytes(upload.getvalue())


def main() -> None:
    st.set_page_config(page_title="Test Strategy Agent", page_icon="🧪", layout="wide")
    st.title("Test Strategy Agent")
    st.caption("Turn an unstructured PRD into traceable, reviewable test cases with a multi-agent workflow.")

    with st.sidebar:
        st.header("Run settings")
        provider = st.selectbox("Mode", ["DeepSeek", "Deterministic demo"], help="DeepSeek runs specialist LLM agents; demo mode uses local deterministic logic.")
        model = st.selectbox("Model", ["deepseek-v4-flash", "deepseek-v4-pro"], disabled=provider != "DeepSeek")
        api_key = st.text_input("DeepSeek API key", type="password", disabled=provider != "DeepSeek", help="Used only for this request; it is not written to disk.")
        max_output_tokens = st.select_slider("Run output-token limit", options=[8000, 16000, 32000], value=16000, disabled=provider != "DeepSeek", help="A shared cap for generated tokens across all specialist agents in this run. 16k is recommended for a normal PRD; input tokens are measured and reported but cannot be hard-capped client-side.")
        st.divider()
        st.markdown("**Agent flow**")
        st.caption("Requirements, journeys, scope, and risks → RAG → test generation → critic → refinement")

    left, right = st.columns([3, 2])
    with left:
        uploaded_prd = st.file_uploader("Upload PRD (.md or .txt)", type=["md", "txt"])
    with right:
        knowledge_files = st.file_uploader("Optional test guidance (.md)", type=["md"], accept_multiple_files=True)

    pasted_prd = st.text_area("Or paste PRD text", height=220, placeholder="# Feature name\n\n## Requirements\n- A signed-in user can …")
    run_clicked = st.button("Generate test strategy", type="primary", use_container_width=True)

    if run_clicked:
        if provider == "DeepSeek" and not api_key:
            st.error("Enter a DeepSeek API key, or switch to Deterministic demo.")
            return
        if not uploaded_prd and not pasted_prd.strip():
            st.error("Upload a PRD or paste its text.")
            return
        live_usage = st.empty()

        def show_usage(summary) -> None:
            live_usage.info(f"Live token usage — {summary.total_tokens:,} total ({summary.input_tokens:,} input + {summary.output_tokens:,} output) across {summary.calls} agent calls.")

        suite = None
        with tempfile.TemporaryDirectory(prefix="prd-test-agent-") as workdir:
            root = Path(workdir)
            knowledge_dir = root / "knowledge"
            knowledge_dir.mkdir()
            if uploaded_prd:
                prd_path = root / uploaded_prd.name
                _save_upload(uploaded_prd, prd_path)
            else:
                prd_path = root / "pasted_prd.md"
                prd_path.write_text(pasted_prd, encoding="utf-8")
            for guidance in knowledge_files or []:
                _save_upload(guidance, knowledge_dir / guidance.name)
            model_client = DeepSeekResponsesModel(model, api_key, max_output_tokens, show_usage) if provider == "DeepSeek" else None
            try:
                with st.spinner("Specialist agents are analyzing the PRD…"):
                    suite = asyncio.run(build_suite(prd_path, knowledge_dir, model_client))
            except RunFailed as error:
                st.session_state["failed_logs"] = error.logs
                st.session_state["failed_usage"] = error.usage
                st.session_state.pop("suite", None)
                st.error("Generation failed. Open the Failed run log below for the provider status and failing agent.")
                live_usage.empty()
            except Exception as error:
                st.session_state["failed_logs"] = []
                st.session_state["failed_usage"] = None
                st.error("Generation failed before the agent workflow could start.")
                st.exception(error)
                return
        if suite is not None:
            st.session_state["suite"] = suite
            st.session_state.pop("failed_logs", None)
            st.session_state.pop("failed_usage", None)
        live_usage.empty()

    suite = st.session_state.get("suite")
    if not suite:
        failed_logs = st.session_state.get("failed_logs", [])
        if failed_logs:
            st.divider()
            st.subheader("Failed run log")
            st.dataframe([entry.model_dump() for entry in failed_logs], use_container_width=True, hide_index=True)
            st.download_button("Download failed run logs", "\n".join(entry.model_dump_json() for entry in failed_logs), file_name="failed_run_logs.jsonl", mime="application/x-ndjson")
            failed_usage = st.session_state.get("failed_usage")
            if failed_usage:
                st.caption(f"Tokens consumed before failure: {failed_usage.total_tokens:,} ({failed_usage.input_tokens:,} input + {failed_usage.output_tokens:,} output).")
        return
    covered, total, percentage = _coverage(suite)
    st.divider()
    st.subheader("Test strategy review")
    metrics = st.columns(4)
    metrics[0].metric("Test cases", len(suite.test_cases))
    metrics[1].metric("Testable requirements", total)
    metrics[2].metric("Requirement coverage", f"{percentage:.0f}%")
    metrics[3].metric("Open questions", len(suite.open_questions))
    if suite.usage:
        usage = suite.usage
        token_metrics = st.columns(4)
        token_metrics[0].metric("Tokens used", f"{usage.total_tokens:,}")
        token_metrics[1].metric("Input tokens", f"{usage.input_tokens:,}")
        token_metrics[2].metric("Output tokens", f"{usage.output_tokens:,}")
        token_metrics[3].metric("Agent calls", usage.calls, help=f"Generated-token cap: {usage.max_output_tokens:,}")

    overview, cases, traceability, questions, logs = st.tabs(["Overview", "Test cases", "Traceability", "Open questions", "Run logs"])
    with overview:
        st.dataframe(_case_rows(suite), use_container_width=True, hide_index=True)
        st.download_button("Download validated JSON", suite.model_dump_json(indent=2), file_name="test_strategy.json", mime="application/json")
    with cases:
        for case in suite.test_cases:
            with st.expander(f"{case.id} · {case.priority} · {case.title}"):
                if case.preconditions:
                    st.markdown("**Preconditions:** " + "; ".join(case.preconditions))
                for index, step in enumerate(case.steps, start=1):
                    st.markdown(f"**{index}. Action:** {step.action}  \n**Expected:** {step.expected_result}")
                st.caption("Evidence: " + "; ".join(f"{e.source} — {e.excerpt}" for e in case.evidence))
    with traceability:
        rows = []
        for requirement in suite.requirements:
            linked = [case.id for case in suite.test_cases if requirement.id in case.requirement_ids]
            rows.append({"Requirement": requirement.id, "Text": requirement.text, "Scope": requirement.kind, "Test cases": ", ".join(linked) or "Not covered"})
        st.dataframe(rows, use_container_width=True, hide_index=True)
    with questions:
        if suite.open_questions:
            for question in suite.open_questions:
                st.warning(question)
        else:
            st.success("No unresolved questions were returned.")
    with logs:
        if suite.run_logs:
            log_rows = [entry.model_dump() for entry in suite.run_logs]
            st.dataframe(log_rows, use_container_width=True, hide_index=True)
            st.download_button("Download run logs", "\n".join(entry.model_dump_json() for entry in suite.run_logs), file_name="run_logs.jsonl", mime="application/x-ndjson")
        else:
            st.info("No run logs are available for this result.")


if __name__ == "__main__":
    main()
