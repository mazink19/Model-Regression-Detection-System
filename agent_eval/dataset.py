from pathlib import Path

import yaml
from pydantic import ValidationError

from agent_eval.models import TestCase


def load_dataset(path: str | Path) -> list[TestCase]:
    """Load one YAML file or all YAML files in a directory with useful errors."""
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.glob("*.yaml")) + sorted(source.glob("*.yml"))
    if not files:
        raise ValueError(f"No YAML dataset files found at {source}")
    cases: list[TestCase] = []
    ids: set[str] = set()
    for file in files:
        try:
            rows = yaml.safe_load(file.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("top level must be a list of cases")
            for index, row in enumerate(rows):
                try:
                    case = TestCase.model_validate(row)
                except ValidationError as exc:
                    raise ValueError(f"{file}, case {index + 1}: {exc}") from exc
                if case.id in ids:
                    raise ValueError(f"{file}, case {index + 1}: duplicate case id {case.id!r}")
                ids.add(case.id)
                cases.append(case)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {file}: {exc}") from exc
    return cases
