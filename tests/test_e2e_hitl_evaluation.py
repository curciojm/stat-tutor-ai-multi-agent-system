
import time

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 180
POLL_INTERVAL_SECONDS = 2


@pytest.mark.e2e
def test_evaluation_with_human_approve():
    """
    Verifies the actual workflow:
    request -> supervisor -> human approval -> evaluator.
    Requires the API, worker, Redis, Pinecone, and LLM providers to be running.
    """

    query = """
    I want you to evaluate the following student's answer.

    Question: What is correlation?

    Student's answer:
    "Correlation consists of calculating an average between two different groups
    to determine which group has the higher average."

    Indicate whether the answer is correct and briefly justify
    your evaluation.
    """

    response = httpx.post(
        f"{BASE_URL}/process",
        json={"query": query},
        timeout=15,
    )
    response.raise_for_status()

    job_id = response.json()["job_id"]
    assert job_id

    deadline = time.monotonic() + TIMEOUT_SECONDS
    state = None
    approvals_send = 0

    while time.monotonic() < deadline:
        response = httpx.get(
            f"{BASE_URL}/status/{job_id}",
            timeout=15,
        )
        response.raise_for_status()
        task = response.json()
        state = task["status"]

        if state == "waiting_approval" and approvals_send == 0:
            response = httpx.post(
                f"{BASE_URL}/approve/{job_id}",
                json={
                    "decision": "approve",
                    "feedback": (
                        "Continue evaluating the answer. "
                        "Add an example involving anxiety levels "
                        "and the frequency of words recalled."
                    ),
                },
                timeout=15,
            )
            response.raise_for_status()
            approvals_send += 1

        elif state == "completed":
            break

        elif state == "failed":
            pytest.fail(
                f"The task failed. Detail: {task.get('error')}"
            )
            response.raise_for_status()
            approvals_send += 1

        elif state == "completed":
            break

        elif state == "failed":
            pytest.fail(
                f"The task failed. Detail: {task.get('error')}"
            )

        time.sleep(POLL_INTERVAL_SECONDS)

    assert state == "completed", (
        f"La tarea no terminó dentro de {TIMEOUT_SECONDS} segundos. "
        f"Último estado: {state}"
    )

    assert approvals_send >= 1, ("The task end it with out human approval.")

    result = task.get("result")
    assert result, "The task end it with out result."

    if isinstance(result, str):
        text_result = result
    else:
        text_result = str(result)

    assert "Evaluation:" in text_result, (
        "The final result does not contain the expected classification. "
        f"Received result: {text_result}"
    )
