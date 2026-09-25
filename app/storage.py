import json
from pathlib import Path

from app.schemas import EvaluationRun


RESULTS_DIR = Path("results")


def get_next_run_id() -> str:
    RESULTS_DIR.mkdir(exist_ok=True)

    existing_runs = list(RESULTS_DIR.glob("run_*.json"))

    if not existing_runs:
        return "run_001"

    numbers = []

    for file in existing_runs:
        try:
            number = int(file.stem.split("_")[1])
            numbers.append(number)
        except (IndexError, ValueError):
            continue

    next_number = max(numbers, default=0) + 1

    return f"run_{next_number:03d}"


def save_run(run: EvaluationRun) -> Path:
    run_id = get_next_run_id()

    file_path = RESULTS_DIR / f"{run_id}.json"

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            run.model_dump(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return file_path

def load_run(file_path: str) -> EvaluationRun:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return EvaluationRun.model_validate(data)