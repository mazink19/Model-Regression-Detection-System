import json

from app.llm import classify_email
from app.schemas import EvaluationCaseResult, EvaluationRun
from app.storage import save_run

# DATASET_PATH  = "dataset/golden_dataset_small.json"
DATASET_PATH  = "dataset/golden_dataset.json"

def load_dataset():
    with open(DATASET_PATH, "r") as file:
        return json.load(file)
    
def evaluate() -> EvaluationRun:
    dataset = load_dataset()
    case_results = []
    print("Running Evaluation..\n")
    for case in  dataset:
        result = classify_email(case["email"])    
        expected = case["expected_category"]
        actual = result.category

        is_correct = expected == actual
        case_result = EvaluationCaseResult(
            case_id=case["id"],
            expected_category=expected,
            actual_category=actual,
            expected_summary=case["expected_summary"],
            actual_summary=result.summary,  
            passed=is_correct
        )
        case_results.append(case_result)
        status = "PASS" if case_result.passed else "FAIL"
        print(
            f"{case_result.case_id}  {status} "
            f"(expected={case_result.expected_category}, "
            f"actual={case_result.actual_category})"
        )

        if not case_result.passed:
            print(f"  Email: {case['email']}")
            print(f"  Expected category: {case_result.expected_category}")
            print(f"  Actual category:   {case_result.actual_category}")
            print(f"  Summary: {case_result.actual_summary}")
            print()

    total = len(case_results)
    passed = sum(case_result.passed for case_result in case_results)
    failed = total - passed
    accuracy = passed / total if total > 0 else 0.0

    evaluation_run = EvaluationRun(
        total_cases=total,
        passed=passed,
        failed=failed,
        accuracy=accuracy,
        cases=case_results
    )
    print("--------------------------------")
    print(f"Total cases: {evaluation_run.total_cases}")
    print(f"Passed: {evaluation_run.passed}")
    print(f"Failed: {evaluation_run.failed}")
    print(f"Category accuracy: {evaluation_run.accuracy * 100:.1f}%")
    print("--------------------------------")
    return evaluation_run

if __name__ == "__main__":
    evaluation_run = evaluate()
    path = save_run(evaluation_run,)

    print(f"\nEvaluation run saved to: {path}")