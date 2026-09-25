from app.regression import evaluate_regression
from app.schemas import CategoryComparison, EvaluationRun, RegressionStatus, CaseChange


def compare_runs(
    baseline: EvaluationRun,
    current: EvaluationRun,
) -> RegressionStatus:

    regression_status = evaluate_regression(
        baseline_accuracy=baseline.accuracy,
        current_accuracy=current.accuracy,
    )

    baseline_cases = {case.case_id: case for case in baseline.cases}

    current_cases = {
        case.case_id: case
        for case in current.cases
    }

    regressions = []
    improvements = []

    for case_id in baseline_cases.keys():

        baseline_case = baseline_cases[case_id]
        current_case = current_cases[case_id]

        # PASS -> FAIL
        if baseline_case.passed and not current_case.passed:
            regressions.append(
                CaseChange(
                    case_id=case_id,
                    expected_category=current_case.expected_category,
                    baseline_category=baseline_case.actual_category,
                    current_category=current_case.actual_category,
                )
            )

        # FAIL -> PASS
        elif not baseline_case.passed and current_case.passed:
            improvements.append(
                CaseChange(
                    case_id=case_id,
                    expected_category=current_case.expected_category,
                    baseline_category=baseline_case.actual_category,
                    current_category=current_case.actual_category,
                )
            )
        categories = ["billing", "technical", "account", "general"]

    category_comparisons = []

    for category in categories:

        baseline_category_cases = [
            case
            for case in baseline.cases
            if case.expected_category == category
        ]

        current_category_cases = [
            case
            for case in current.cases
            if case.expected_category == category
        ]

        baseline_correct = sum(
            case.passed
            for case in baseline_category_cases
        )

        current_correct = sum(
            case.passed
            for case in current_category_cases
        )

        baseline_accuracy = (
            baseline_correct / len(baseline_category_cases)
            if baseline_category_cases
            else 0.0
        )

        current_accuracy = (
            current_correct / len(current_category_cases)
            if current_category_cases
            else 0.0
        )

        category_comparisons.append(
            CategoryComparison(
                category=category,
                baseline_accuracy=baseline_accuracy,
                current_accuracy=current_accuracy,
                accuracy_delta=current_accuracy - baseline_accuracy,
            )
        )


    return RegressionStatus(
        status=regression_status.status,
        accuracy_delta=regression_status.accuracy_delta,
        regressions=regressions,
        improvements=improvements,
        category_comparisons = category_comparisons
    )