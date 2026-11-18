
import pytest
from pydantic import ValidationError

from app.schemas import EvaluationResult


def test_evaluation_result_accepts_valid_category():
    result = EvaluationResult(
        evaluation="Good",
        improvement="Could expand on the explanation.",
    )

    assert result.evaluacion == "Good"
    assert result.improvement == "Could expand on the explanation."


def test_evaluation_result_rejects_invalid_category():
    with pytest.raises(ValidationError):
        EvaluationResult(
            evaluation="Excelent",
            improvement="No improvement.",
        )
