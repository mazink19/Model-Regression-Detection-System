from pydantic_settings import BaseSettings


class EvaluationSettings(BaseSettings):
    warning_threshold: float = 0.03
    critical_threshold: float = 0.08


settings = EvaluationSettings()