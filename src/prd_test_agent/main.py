import argparse
import asyncio
import os
from pathlib import Path

from .llm import DeepSeekResponsesModel
from .pipeline import build_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a validated test suite from a Markdown PRD.")
    parser.add_argument("--prd", required=True, type=Path)
    parser.add_argument("--knowledge-dir", required=True, type=Path)
    parser.add_argument("--out", default="test_cases.json", type=Path)
    parser.add_argument("--provider", choices=("deterministic", "deepseek"), default="deterministic")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-output-tokens", type=int, default=8000, help="Maximum generated tokens shared by all agents in one run.")
    args = parser.parse_args()
    if args.provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        parser.error("DEEPSEEK_API_KEY is required when --provider deepseek is selected.")
    model = DeepSeekResponsesModel(args.model, max_run_output_tokens=args.max_output_tokens) if args.provider == "deepseek" else None
    suite = asyncio.run(build_suite(args.prd, args.knowledge_dir, model))
    args.out.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {len(suite.test_cases)} validated test cases to {args.out}")


if __name__ == "__main__":
    main()
