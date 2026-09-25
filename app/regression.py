from app.config import settings
from app.schemas import RegressionStatus


def evaluate_regression(baseline_accuracy: float,
                        current_accuracy: float,) -> RegressionStatus:
    delta = current_accuracy - baseline_accuracy
    regression_amount = -delta
    if regression_amount > settings.critical_threshold:
        status = "critical"
    elif regression_amount > settings.warning_threshold:
        status = "warning"
    else:
        status = "pass"

    return RegressionStatus(
        status = status,
        accuracy_delta = delta
    )    
