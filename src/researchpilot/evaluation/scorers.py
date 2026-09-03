"""Deterministic scorers for retrieval, citations, and response structure."""

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from researchpilot.evaluation.models import CaseMetrics, EvalCase, GoldEvidence

SUMMARY_CLAIM_FIELDS = (
    "research_tasks",
    "methods",
    "datasets",
    "baselines",
    "experiment_settings",
    "metrics",
    "results",
    "limitations",
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _integer_ids(values: Any) -> set[int]:
    return {
        value
        for value in _list(values)
        if isinstance(value, int) and not isinstance(value, bool)
    }


def collect_cited_ids(endpoint: str, response: dict[str, Any]) -> set[int]:
    """Collect every evidence ID referenced by the generated answer."""

    cited_ids: set[int] = set()

    if endpoint == "qa":
        answer = _dict(response.get("answer"))
        cited_ids.update(_integer_ids(answer.get("overview_evidence_ids")))
        for claim in _list(answer.get("claims")):
            cited_ids.update(_integer_ids(_dict(claim).get("evidence_ids")))
        return cited_ids

    analysis = _dict(response.get("analysis"))
    cited_ids.update(_integer_ids(analysis.get("overall_evidence_ids")))

    for comparison in _list(analysis.get("comparisons")):
        cited_ids.update(_integer_ids(_dict(comparison).get("evidence_ids")))

    for summary_value in _list(analysis.get("document_summaries")):
        summary = _dict(summary_value)
        for field_name in SUMMARY_CLAIM_FIELDS:
            for claim in _list(summary.get(field_name)):
                cited_ids.update(_integer_ids(_dict(claim).get("evidence_ids")))

    return cited_ids


def _actual_sufficient(endpoint: str, response: dict[str, Any]) -> bool | None:
    container_name = "answer" if endpoint == "qa" else "analysis"
    value = _dict(response.get(container_name)).get("sufficient_evidence")
    return value if isinstance(value, bool) else None


def _page_range(evidence: dict[str, Any]) -> set[int]:
    start = evidence.get("page_start")
    end = evidence.get("page_end")

    if isinstance(start, int):
        if not isinstance(end, int) or end < start:
            end = start
        return set(range(start, end + 1))

    pages: set[int] = set()
    for location in _list(evidence.get("source_locations")):
        page_no = _dict(location).get("page_no")
        if isinstance(page_no, int):
            pages.add(page_no)
    return pages


def _normalise_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", "", value)


def _matching_evidences(
    target: GoldEvidence,
    evidences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_pages = set(target.acceptable_pages)
    matches: list[dict[str, Any]] = []

    for evidence in evidences:
        if str(evidence.get("document_id", "")) != target.document_id:
            continue
        if target_pages and not (_page_range(evidence) & target_pages):
            continue
        matches.append(evidence)

    return matches


def _anchor_hit(target: GoldEvidence, matches: Iterable[dict[str, Any]]) -> bool:
    if not target.anchor_terms:
        return True

    terms = [_normalise_text(term) for term in target.anchor_terms]
    for evidence in matches:
        text = _normalise_text(str(evidence.get("text", "")))
        if any(term and term in text for term in terms):
            return True
    return False


def _score_structure(
    case: EvalCase,
    response: dict[str, Any],
) -> tuple[float | None, list[str]]:
    if case.endpoint != "experiment_analysis":
        return None, []

    # Partial/none cases intentionally may return sparse summaries.
    if not case.expected_sufficient_evidence:
        return None, []

    analysis = _dict(response.get("analysis"))
    summaries = {
        str(summary.get("document_id", "")): summary
        for item in _list(analysis.get("document_summaries"))
        if (summary := _dict(item))
    }

    checks = 0
    passed = 0
    missing: list[str] = []

    for document_id in case.expected_document_coverage:
        summary = summaries.get(document_id)
        checks += 1
        if summary is None:
            missing.append(f"document_summary:{document_id}")
            continue
        passed += 1

        for field_name in case.required_summary_fields:
            checks += 1
            if _list(summary.get(field_name)):
                passed += 1
            else:
                missing.append(f"{document_id}.{field_name}")

    if case.required_comparison_aspects:
        checks += 1
        if _list(analysis.get("comparisons")):
            passed += 1
        else:
            missing.append("comparisons")

    return (passed / checks if checks else None), missing


def score_response(case: EvalCase, response: dict[str, Any]) -> CaseMetrics:
    """Score one successful API response without another model call."""

    evidences = [
        evidence
        for value in _list(response.get("evidences"))
        if (evidence := _dict(value))
    ]
    valid_ids = {
        evidence_id
        for evidence in evidences
        if isinstance((evidence_id := evidence.get("evidence_id")), int)
    }
    cited_ids = collect_cited_ids(case.endpoint, response)
    invalid_ids = cited_ids - valid_ids

    actual_sufficient = _actual_sufficient(case.endpoint, response)
    sufficiency_correct = (
        actual_sufficient == case.expected_sufficient_evidence
        if actual_sufficient is not None
        else False
    )

    citation_validity = (
        (len(cited_ids) - len(invalid_ids)) / len(cited_ids) if cited_ids else 1.0
    )
    citation_presence_ok = not (actual_sufficient is True and not cited_ids)

    if case.expected_document_coverage:
        retrieved_document_ids = {
            str(evidence.get("document_id", "")) for evidence in evidences
        }
        expected_document_ids = set(case.expected_document_coverage)
        missing_document_ids = sorted(expected_document_ids - retrieved_document_ids)
        document_coverage = len(expected_document_ids & retrieved_document_ids) / len(
            expected_document_ids
        )
    else:
        missing_document_ids = []
        document_coverage = None

    missed_targets: list[int] = []
    page_hits = 0
    anchor_hits = 0
    for index, target in enumerate(case.gold_evidence, start=1):
        matches = _matching_evidences(target, evidences)
        if matches:
            page_hits += 1
            if _anchor_hit(target, matches):
                anchor_hits += 1
        else:
            missed_targets.append(index)

    target_count = len(case.gold_evidence)
    gold_page_recall = page_hits / target_count if target_count else None
    anchor_evidence_recall = anchor_hits / target_count if target_count else None

    structure_completeness, missing_structure_items = _score_structure(
        case,
        response,
    )

    return CaseMetrics(
        request_ok=True,
        actual_sufficient_evidence=actual_sufficient,
        sufficient_evidence_correct=sufficiency_correct,
        citation_validity=citation_validity,
        citation_presence_ok=citation_presence_ok,
        cited_evidence_ids=sorted(cited_ids),
        invalid_citation_ids=sorted(invalid_ids),
        document_coverage=document_coverage,
        missing_document_ids=missing_document_ids,
        gold_page_recall=gold_page_recall,
        anchor_evidence_recall=anchor_evidence_recall,
        missed_gold_evidence_indexes=missed_targets,
        structure_completeness=structure_completeness,
        missing_structure_items=missing_structure_items,
    )


def case_passed(metrics: CaseMetrics) -> bool:
    """Apply a transparent v1 quality gate to one case."""

    if not metrics.request_ok:
        return False
    if metrics.sufficient_evidence_correct is not True:
        return False
    if metrics.citation_validity != 1.0:
        return False
    if metrics.citation_presence_ok is not True:
        return False
    if metrics.document_coverage is not None and metrics.document_coverage < 1.0:
        return False
    if metrics.gold_page_recall is not None and metrics.gold_page_recall < 0.5:
        return False
    return not (
        metrics.structure_completeness is not None
        and metrics.structure_completeness < 1.0
    )
