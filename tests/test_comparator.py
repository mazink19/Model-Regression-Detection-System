from app.comparator import compare_runs
from app.schemas import EvaluationRun, EvaluationCaseResult


def test_detects_regression_and_improvement():

    baseline = EvaluationRun(
        total_cases=2,
        passed=1,
        failed=1,
        accuracy=0.5,
        cases=[
            EvaluationCaseResult(
                case_id="case_001",
                expected_category="billing",
                actual_category="billing",
                expected_summary="test",
                actual_summary="test",
                passed=True,
            ),
            EvaluationCaseResult(
                case_id="case_002",
                expected_category="account",
                actual_category="technical",
                expected_summary="test",
                actual_summary="test",
                passed=False,
            ),
        ],
    )

    current = EvaluationRun(
        total_cases=2,
        passed=1,
        failed=1,
        accuracy=0.5,
        cases=[
            EvaluationCaseResult(
                case_id="case_001",
                expected_category="billing",
                actual_category="technical",
                expected_summary="test",
                actual_summary="test",
                passed=False,
            ),
            EvaluationCaseResult(
                case_id="case_002",
                expected_category="account",
                actual_category="account",
                expected_summary="test",
                actual_summary="test",
                passed=True,
            ),
        ],
    )

    result = compare_runs(baseline, current)

    assert result.status == "pass"
    assert result.accuracy_delta == 0.0

    assert len(result.regressions) == 1
    assert result.regressions[0].case_id == "case_001"

    assert len(result.improvements) == 1
    assert result.improvements[0].case_id == "case_002"