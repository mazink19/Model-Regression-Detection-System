import json
import sqlite3
from pathlib import Path

from agent_eval.models import AgentRun


class RunStore:
    def __init__(self, path: str | Path = "results/agent_eval.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, agent_name TEXT NOT NULL, run_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS baselines (name TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id));
            """)

    def _connect(self):
        return sqlite3.connect(self.path)

    def save(self, run: AgentRun) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", (run.id, run.created_at, run.agent_name, run.model_dump_json()))

    def load(self, run_id: str) -> AgentRun:
        with self._connect() as db:
            row = db.execute("SELECT run_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            raise KeyError(f"Unknown run: {run_id}")
        return AgentRun.model_validate_json(row[0])

    def latest(self) -> AgentRun:
        with self._connect() as db:
            row = db.execute("SELECT run_json FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        if not row:
            raise KeyError("No evaluation runs have been saved")
        return AgentRun.model_validate_json(row[0])

    def set_baseline(self, name: str, run_id: str) -> None:
        with self._connect() as db:
            if not db.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone():
                raise KeyError(f"Unknown run: {run_id}")
            db.execute("INSERT INTO baselines(name, run_id) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET run_id=excluded.run_id", (name, run_id))

    def baseline(self, name: str) -> AgentRun:
        with self._connect() as db:
            row = db.execute("SELECT run_id FROM baselines WHERE name = ?", (name,)).fetchone()
        if not row:
            raise KeyError(f"Unknown baseline: {name}")
        return self.load(row[0])
