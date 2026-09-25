from pathlib import Path

from app.schemas import RegressionStatus, EvaluationRun


REPORTS_DIR = Path("reports")


def generate_html_report(
    baseline: EvaluationRun,
    current: EvaluationRun,
    result: RegressionStatus,
) -> Path:

    REPORTS_DIR.mkdir(exist_ok=True)

    report_path = REPORTS_DIR / "evaluation_report.html"

    regressions_html = ""

    for case in result.regressions:
        regressions_html += f"""
        <tr>
            <td>{case.case_id}</td>
            <td>{case.expected_category}</td>
            <td>{case.baseline_category}</td>
            <td>{case.current_category}</td>
        </tr>
        """

    if not regressions_html:
        regressions_html = """
        <tr>
            <td colspan="4">No regressions detected.</td>
        </tr>
        """

    improvements_html = ""

    for case in result.improvements:
        improvements_html += f"""
        <tr>
            <td>{case.case_id}</td>
            <td>{case.expected_category}</td>
            <td>{case.baseline_category}</td>
            <td>{case.current_category}</td>
        </tr>
        """

    if not improvements_html:
        improvements_html = """
        <tr>
            <td colspan="4">No improvements detected.</td>
        </tr>
        """

    categories_html = ""

    for category in result.category_comparisons:
        categories_html += f"""
        <tr>
            <td>{category.category}</td>
            <td>{category.baseline_accuracy * 100:.1f}%</td>
            <td>{category.current_accuracy * 100:.1f}%</td>
            <td>{category.accuracy_delta * 100:+.1f}%</td>
        </tr>
        """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLM Evaluation Report</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }}

        .container {{
            max-width: 1000px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
        }}

        h1, h2 {{
            margin-bottom: 15px;
        }}

        .summary {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
            flex: 1;
        }}

        .status {{
            font-size: 24px;
            font-weight: bold;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}

        th {{
            background: #eee;
        }}
    </style>
</head>

<body>

<div class="container">

    <h1>LLM Evaluation Report</h1>

    <div class="summary">

        <div class="card">
            <strong>Baseline Accuracy</strong>
            <div>{baseline.accuracy * 100:.1f}%</div>
        </div>

        <div class="card">
            <strong>Current Accuracy</strong>
            <div>{current.accuracy * 100:.1f}%</div>
        </div>

        <div class="card">
            <strong>Accuracy Delta</strong>
            <div>{result.accuracy_delta * 100:+.1f}%</div>
        </div>

        <div class="card">
            <strong>Status</strong>
            <div class="status">{result.status.upper()}</div>
        </div>

    </div>


    <h2>Category Accuracy</h2>

    <table>
        <thead>
            <tr>
                <th>Category</th>
                <th>Baseline</th>
                <th>Current</th>
                <th>Delta</th>
            </tr>
        </thead>

        <tbody>
            {categories_html}
        </tbody>
    </table>


    <h2>Regressions (PASS → FAIL)</h2>

    <table>
        <thead>
            <tr>
                <th>Case</th>
                <th>Expected</th>
                <th>Baseline</th>
                <th>Current</th>
            </tr>
        </thead>

        <tbody>
            {regressions_html}
        </tbody>
    </table>


    <h2>Improvements (FAIL → PASS)</h2>

    <table>
        <thead>
            <tr>
                <th>Case</th>
                <th>Expected</th>
                <th>Baseline</th>
                <th>Current</th>
            </tr>
        </thead>

        <tbody>
            {improvements_html}
        </tbody>
    </table>

</div>

</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")

    return report_path