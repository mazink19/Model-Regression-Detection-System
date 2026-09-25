from app.comparator import compare_runs
from app.evaluator import evaluate
from app.storage import load_run
from app.report import generate_html_report

BASELINE_PATH = "baseline/baseline.json"


def main():
    print("Running CI evaluation...\n")

    baseline = load_run(BASELINE_PATH)

    current = evaluate()

    result = compare_runs(baseline=baseline,current=current,)
    report_path = generate_html_report(
    baseline=baseline,
    current=current,
    result=result,
                )

    print(f"HTML report: {report_path}")

    print("\n========== CI REGRESSION CHECK ==========")

    print(f"Baseline accuracy: {baseline.accuracy * 100:.1f}%")
    print(f"Current accuracy:  {current.accuracy * 100:.1f}%")
    print(f"Accuracy delta:    {result.accuracy_delta * 100:+.1f}%")
    print(f"Status:            {result.status.upper()}")

    print(f"\nRegressions: {len(result.regressions)}")
    print(f"Improvements: {len(result.improvements)}")

    if result.regressions:
        print("\nRegressions:")
        for case in result.regressions:
            print(
                f"  {case.case_id}: "
                f"{case.baseline_category} -> "
                f"{case.current_category}"
            )

    print("\n==========================================")

    if result.status == "critical":
        print("CI CHECK FAILED")
        raise SystemExit(1)

    print("CI CHECK PASSED")


if __name__ == "__main__":
    main()