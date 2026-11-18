import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import app.main as main


@pytest.mark.asyncio
async def test_process_crea_tarea_y_encola_sin_redis_real():
    mock_redis = AsyncMock()

    with patch.object(main, "redis_client", mock_redis):
        transport = httpx.ASGITransport(app=main.app)

        async with httpx.AsyncClient(transport=transport,base_url="http://test") as client:
            response = await client.post(
                "/process",
                json={"query": "What is a hypothesis test?"},
            )

    assert response.status_code == 200

    body = response.json()
    job_id = body["job_id"]

    assert body["status"] == "pending"

    mock_redis.set.assert_awaited_once()
    mock_redis.rpush.assert_awaited_once_with(
        "multiagent_tasks",
        job_id,
    )

    saved_key, saved_value = mock_redis.set.await_args.args
    saved_task = json.loads(saved_value)

    assert saved_key == f"task_status:{job_id}"
    assert saved_task == {
        "job_id": job_id,
        "query": "What is a hypothesis test?",
        "status": "pending",
        "result": None,
        "error": None,
    }

@pytest.mark.asyncio
async def test_status_returns_task_saved_in_redis_mock():
    job_id = "test-job-123"

    task_data = {
        "job_id": job_id,
        "query": "What is a hypothesis test?",
        "status": "processing",
        "result": None,
        "error": None,
    }

    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps(task_data)

    with patch.object(main, "redis_client", mock_redis):
        transport = httpx.ASGITransport(app=main.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get(f"/status/{job_id}")

    assert response.status_code == 200
    assert response.json() == task_data

    mock_redis.get.assert_awaited_once_with(
        f"task_status:{job_id}"
    )

@pytest.mark.asyncio
async def test_status_returns_404_if_task_dont_exist_mock():
    job_id = "job-inexistente"

    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    with patch.object(main, "redis_client", mock_redis):
        transport = httpx.ASGITransport(app=main.app)

        async with httpx.AsyncClient(transport=transport,base_url="http://test") as client:
            response = await client.get(f"/status/{job_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}

    mock_redis.get.assert_awaited_once_with(f"task_status:{job_id}")

@pytest.mark.asyncio
async def test_approve_save_desicion_and_enqueues_redis_mock():
    job_id = "test-job-approval"

    task_data = {
        "job_id": job_id,
        "query": "Explain what is a hypothesis test",
        "status": "waiting_approval",
        "result": "Agent explanation",
        "error": None,
    }

    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps(task_data)

    with patch.object(main, "redis_client", mock_redis):
        transport = httpx.ASGITransport(app=main.app)

        async with httpx.AsyncClient(transport=transport,base_url="http://test") as client:
            response = await client.post(
                f"/approve/{job_id}",
                json={
                    "decision": "approve",
                    "feedback": "The explanation is correct",
                },
            )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": job_id,
        "status": "approval_submitted",
    }

    mock_redis.get.assert_awaited_once_with(
        f"task_status:{job_id}"
    )

    mock_redis.set.assert_awaited_once_with(
        f"approval_data:{job_id}",
        json.dumps({
            "decision": "approve",
            "feedback": "The explanation is correct",
        }),
    )

    mock_redis.rpush.assert_awaited_once_with(
        "multiagent_approvals",
        job_id,
    )

@pytest.mark.asyncio
async def test_approve_returns_404_if_task_doesnt_exist_mock():
    job_id = "job-inexistente"

    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    with patch.object(main, "redis_client", mock_redis):
        transport = httpx.ASGITransport(app=main.app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/approve/{job_id}",
                json={
                    "decision": "approve",
                    "feedback": "The explanation is correct",
                },
            )

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}

    mock_redis.get.assert_awaited_once_with(f"task_status:{job_id}")
    mock_redis.set.assert_not_awaited()
    mock_redis.rpush.assert_not_awaited()

@pytest.mark.asyncio
async def test_approve_reject_task_if_it_is_out_of_state_mock():
    job_id = "test-job-pending"

    task_data = {
        "job_id": job_id,
        "query": "Explain what is a hypothesis test",
        "status": "pending",
        "result": None,
        "error": None,
    }

    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps(task_data)

    with patch.object(main, "redis_client", mock_redis):
        transport = httpx.ASGITransport(app=main.app)

        async with httpx.AsyncClient(transport=transport,base_url="http://test") as client:
            response = await client.post(
                f"/approve/{job_id}",
                json={
                    "decision": "approve",
                    "feedback": "The explanation is correct",
                },
            )

    assert response.status_code == 400
    assert response.json() == {"detail": "Job is not waiting for approval"}

    mock_redis.get.assert_awaited_once_with(f"task_status:{job_id}")
    mock_redis.set.assert_not_awaited()
    mock_redis.rpush.assert_not_awaited()

@pytest.mark.asyncio
async def test_process_fails_if_redis_cant_save():
    mock_redis = AsyncMock()
    mock_redis.set.side_effect = ConnectionError(
        "Redis no disponible"
    )

    with patch.object(main, "redis_client", mock_redis):
        transport = httpx.ASGITransport(
            app=main.app,
            raise_app_exceptions=False,
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/process",
                json={"query": "What is a hypothesis test?"},
            )
    
    assert response.status_code == 500
    mock_redis.set.assert_awaited_once()
    mock_redis.rpush.assert_not_awaited()