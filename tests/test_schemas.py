
import pytest
from pydantic import ValidationError

from app.schemas import EvaluationResult


def test_evaluation_result_accepts_valid_category():
    result = EvaluationResult(
        evaluation="Good",
        improve="Could expand on the explanation.",
    )

    assert result.evaluation == "Good"
    assert result.improve == "Could expand on the explanation."


def test_evaluation_result_rejects_invalid_category():
    with pytest.raises(ValidationError):
        EvaluationResult(
            evaluation="Excelent",
            improve="No improvement.",
        )
