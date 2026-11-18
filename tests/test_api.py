import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_create_task():
    fake_redis = AsyncMock()

    with patch("app.main.redis_client", fake_redis):
        response = client.post(
            "/process",
            json={
                "query": "Explain what mean is."
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "job_id" in data
    assert data["status"] == "pending"

    fake_redis.set.assert_awaited_once()
    fake_redis.rpush.assert_awaited_once()

    args = fake_redis.rpush.await_args.args

    assert args[0] == "multiagent_tasks"
    assert args[1] == data["job_id"]

    set_args = fake_redis.set.await_args.args

    assert set_args[0].startswith("task_status:")

    task_data = json.loads(set_args[1])

    assert task_data["job_id"] == data["job_id"]
    assert task_data["query"] == "Explain what mean is."
    assert task_data["status"] == "pending"

def test_get_status_returns_actual_task():
    job_id = "test-job-123"

    task_data = {
        "job_id": job_id,
        "query": "Explain what mean is.",
        "status": "processing",
        "result": None,
        "error": None,
    }

    fake_redis = AsyncMock()
    fake_redis.get.return_value = json.dumps(task_data)

    with patch("app.main.redis_client", fake_redis):
        response = client.get(f"/status/{job_id}")

    assert response.status_code == 200
    assert response.json() == task_data

    fake_redis.get.assert_awaited_once_with(
        f"task_status:{job_id}"
    )


def test_get_status_return_404_if_task_doesnt_exist():
    job_id = "job-inexistente"

    fake_redis = AsyncMock()
    fake_redis.get.return_value = None

    with patch("app.main.redis_client", fake_redis):
        response = client.get(f"/status/{job_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_approve_task_approval_valid():
    job_id = "test-job-123"

    task_data = {
        "job_id": job_id,
        "query": "Explain what mean is.",
        "status": "waiting_approval",
        "result": None,
        "error": None,
    }

    fake_redis = AsyncMock()
    fake_redis.get.return_value = json.dumps(task_data)

    with patch("app.main.redis_client", fake_redis):
        response = client.post(
            f"/approve/{job_id}",
            json={
                "decision": "approve",
                "feedback": "The answer is correct",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": job_id,
        "status": "approval_submitted",
    }

    fake_redis.set.assert_awaited_once()

    set_args = fake_redis.set.await_args.args

    assert set_args[0] == f"approval_data:{job_id}"

    approval_data = json.loads(set_args[1])

    assert approval_data == {
        "decision": "approve",
        "feedback": "The answer is correct",
    }

    fake_redis.rpush.assert_awaited_once_with(
        "multiagent_approvals",
        job_id,
    )


def test_approve_task_reject_task_if_approve_isnt_expected():
    job_id = "test-job-123"

    task_data = {
        "job_id": job_id,
        "query": "Explain what mean is",
        "status": "processing",
        "result": None,
        "error": None,
    }

    fake_redis = AsyncMock()
    fake_redis.get.return_value = json.dumps(task_data)

    with patch("app.main.redis_client", fake_redis):
        response = client.post(
            f"/approve/{job_id}",
            json={
                "decision": "approve",
                "feedback": "",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Job is not waiting for approval"
    )

    fake_redis.set.assert_not_awaited()
    fake_redis.rpush.assert_not_awaited()