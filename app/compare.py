from app.comparator import compare_runs
from app.storage import load_run
from app.report import generate_html_report

baseline = load_run("results/run_001.json")
current = load_run("results/run_002.json")

result = compare_runs(baseline, current)
print("\n========== REGRESSION REPORT ==========")

print(f"Baseline accuracy: {baseline.accuracy * 100:.1f}%")
print(f"Current accuracy:  {current.accuracy * 100:.1f}%")
print(f"Accuracy delta:    {result.accuracy_delta * 100:.1f}%")

print(f"\nStatus: {result.status}")

print("\nRegressions (PASS -> FAIL):")

if result.regressions:
    for case in result.regressions:
        print(
            f"  {case.case_id}: "
            f"{case.baseline_category} -> {case.current_category}"
        )
else:
    print("  None")

print("\nImprovements (FAIL -> PASS):")

if result.improvements:
    for case in result.improvements:
        print(
            f"  {case.case_id}: "
            f"{case.baseline_category} -> {case.current_category}"
        )
else:
    print("  None")

print("\nCategory Accuracy:")

for category in result.category_comparisons:
    print(
        f"  {category.category}: "
        f"{category.baseline_accuracy * 100:.1f}% -> "
        f"{category.current_accuracy * 100:.1f}% "
        f"(delta: {category.accuracy_delta * 100:+.1f}%)"
    )    
report_path = generate_html_report(
    baseline=baseline,
    current=current,
    result=result,
)

print(f"\nHTML report generated: {report_path}")    