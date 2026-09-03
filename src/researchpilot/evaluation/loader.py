"""Load and validate ResearchPilot evaluation JSONL files."""

from pathlib import Path

from pydantic import ValidationError

from researchpilot.evaluation.models import EvalCase, SuiteName


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Read a JSONL file and report validation errors with line numbers."""

    if not path.is_file():
        raise FileNotFoundError(f"测评文件不存在：{path}")

    cases: list[EvalCase] = []
    seen_case_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                case = EvalCase.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行校验失败：\n{exc}"
                ) from exc

            if case.case_id in seen_case_ids:
                raise ValueError(
                    f"{path} 第 {line_number} 行 case_id 重复：{case.case_id}"
                )

            seen_case_ids.add(case.case_id)
            cases.append(case)

    if not cases:
        raise ValueError(f"测评文件中没有案例：{path}")

    return cases


def load_suite(evals_dir: Path, suite: SuiteName) -> list[EvalCase]:
    """Load one suite or both suites, preserving file order."""

    paths: list[Path] = []
    if suite in {"qa", "all"}:
        paths.append(evals_dir / "qa_cases.jsonl")
    if suite in {"experiment", "all"}:
        paths.append(evals_dir / "experiment_cases.jsonl")

    cases: list[EvalCase] = []
    seen_case_ids: set[str] = set()

    for path in paths:
        for case in load_eval_cases(path):
            if case.case_id in seen_case_ids:
                raise ValueError(f"多个测评文件之间 case_id 重复：{case.case_id}")
            seen_case_ids.add(case.case_id)
            cases.append(case)

    return cases
