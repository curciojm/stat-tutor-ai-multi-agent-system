import os
import time
import uuid

import httpx
import pytest

BASE_URL = os.getenv(
    "TEST_BASE_URL",
    "http://127.0.0.1:8000",
)

TIMEOUT_SECONDS = 180
POLL_INTERVAL_SECONDS = 2


@pytest.mark.e2e
def test_query_complete_end_to_end():
    """
    Test the actual workflow:
    FastAPI -> Redis -> Worker -> LangGraph/RAG -> response.
    Requires run.sh to be running and the credentials
    for external services to be configured.
    """

    query = (
        "Explain what regression is and how to calculate."
    )

    with httpx.Client(base_url=BASE_URL,timeout=30) as client:

        response = client.post(
            "/process",
            json={"query": query},
        )

        assert response.status_code == 200, (
            f"Error in task initialization: {response.text}"
        )

        task = response.json()

        assert task.get("job_id")
        assert task["status"] == "pending"

        job_id = task["job_id"]

        deadline = time.monotonic() + TIMEOUT_SECONDS
        last_status = None

        while time.monotonic() < deadline:
            status_response = client.get(
                f"/status/{job_id}",
            )

            assert status_response.status_code == 200, (
                f"Error consultando estado: "
                f"{status_response.text}"
            )

            task_status = status_response.json()
            last_status = task_status["status"]

            if last_status == "completed":
                break

            if last_status == "failed":
                pytest.fail(
                    "Process failed: "
                    f"{task_status.get('error')}"
                )

            if last_status == "waiting_approval":
                pytest.fail(
                    "The task is waiting for human approval; "
                    "this test expected an automatic response."
                )

            time.sleep(POLL_INTERVAL_SECONDS)

        else:
            pytest.fail(
                f"The task {job_id} didn't end in "
                f"{TIMEOUT_SECONDS} seconds. "
                f"Last state: {last_status}"
            )

        final_response = client.get(
            f"/status/{job_id}",
        )

        assert final_response.status_code == 200

        result = final_response.json()

        
    assert result["status"] == "completed"

    response = result.get("result")

    assert response, "The task end it with out response."

    if isinstance(response, str):
        text_response = response
    elif isinstance(response, list):
        textos = [
            bloque["text"]
            for bloque in response
            if isinstance(bloque, dict)
            and bloque.get("type") == "text"
            and isinstance(bloque.get("text"), str)
        ]
        text_response = "\n".join(textos)
    else:
        pytest.fail(
            f"Unexpected response format: "
            f"{type(response).__name__}"
        )

    assert text_response.strip(), ("The task end it with out a useful text response.")
    assert result.get("error") is None

