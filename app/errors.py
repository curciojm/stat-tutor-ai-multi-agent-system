from app.schemas import LLMError, LLMErrorType


def classify_error(error: Exception) -> LLMError:
    status_code = getattr(error, "status_code", None)

    if status_code == 401:
        return LLMError(
            LLMErrorType.KEY,
            "Provider credentials are invalid",
        )

    if status_code == 429:
        return LLMError(
            LLMErrorType.RATE_LIMIT,
            "Request limit reached",
        )

    return LLMError(
        LLMErrorType.UNKNOWN,
        "Unknown error, please try later",
    )