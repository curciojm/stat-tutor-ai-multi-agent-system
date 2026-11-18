import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.worker import main_worker, approval_processing


@pytest.mark.asyncio
async def test_process_approval_marks_failed_if_graph_fails():

    job_id = "test-job-123"

    task_data = {
        "job_id": job_id,
        "query": "Evaluate a response about the mean.",
        "status": "waiting_approval",
        "result": None,
        "error": None,
    }

    approval_data = {
        "decision": "approve",
        "feedback": "",
    }

    fake_redis = AsyncMock()

    fake_redis.get.side_effect = [
        json.dumps(task_data),
        json.dumps(approval_data),
    ]

    fake_app = AsyncMock()
    fake_app.ainvoke.side_effect = Exception(
        "Simulated graph error"
    )

    with patch(
        "app.worker.redis_client",
        fake_redis,
    ):
        await approval_processing(
            job_id,
            fake_app,
        )

    calls_set = fake_redis.set.await_args_list

    last_call = calls_set[-1]

    key_guardada = last_call.args[0]
    data_saved = json.loads(
        last_call.args[1]
    )

    assert key_guardada == f"task_status:{job_id}"
    assert data_saved["status"] == "failed"
    assert data_saved["result"] is None
    assert data_saved["error"] == "Evaluate a response about the mean."

    fake_app.ainvoke.assert_awaited_once()

@pytest.mark.asyncio
async def test_main_worker_completes_task_successfully():

    job_id = "test-job-success"

    task_data = {
        "job_id": job_id,
        "query": "Explain what is the mean.",
        "status": "pending",
        "result": None,
        "error": None,
    }

    response = "The mean is the arithmetic average."


    fake_redis = AsyncMock()
    fake_redis.lpop.side_effect = [
        job_id,
        None,
        None,
    ]
    fake_redis.get.return_value = json.dumps(task_data)

    fake_app = AsyncMock()
    fake_app.ainvoke.return_value = {
        "messages": [
            HumanMessage(content=response)
        ]
    }

    fake_graph = type(
        "FakeGraph",
        (),
        {"compile": lambda self, checkpointer: fake_app},
    )()

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class StopWorker(Exception):
        pass

    with (
        patch("app.worker.redis_client", fake_redis),
        patch("app.worker.graph", fake_graph),
        patch(
            "app.worker.AsyncRedisSaver.from_conn_string",
            return_value=FakeSaver(),
        ),
        patch("app.worker.serialize_trace", return_value=[]),
        patch("app.worker.save_trace"),
        patch(
            "app.worker.asyncio.sleep",
            side_effect=StopWorker,
        ),
    ):
        with pytest.raises(StopWorker):
            await main_worker()

    fake_app.ainvoke.assert_awaited_once()

    assert fake_redis.set.await_count == 2

    last_set = fake_redis.set.await_args_list[-1]
    saved_key = last_set.args[0]
    final_state = json.loads(last_set.args[1])

    assert saved_key == f"task_status:{job_id}"
    assert final_state["status"] == "completed"
    assert final_state["result"] == response
    assert final_state["error"] is None

@pytest.mark.asyncio
async def test_main_worker_leaves_task_waiting_for_approval():

    job_id = "test-job-approval"

    task_data = {
        "job_id": job_id,
        "query": "Evaluate my answer about the mean.",
        "status": "pending",
        "result": None,
        "error": None,
    }

    fake_redis = AsyncMock()
    fake_redis.lpop.side_effect = [
        job_id,
        None,
        None,
    ]
    fake_redis.get.return_value = json.dumps(task_data)

    fake_app = AsyncMock()
    fake_app.ainvoke.return_value = {
        "__interrupt__": ["Waiting for human approval"]
    }

    fake_graph = type(
        "FakeGraph",
        (),
        {"compile": lambda self, checkpointer: fake_app},
    )()

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class StopWorker(Exception):
        pass

    with (
        patch("app.worker.redis_client", fake_redis),
        patch("app.worker.graph", fake_graph),
        patch(
            "app.worker.AsyncRedisSaver.from_conn_string",
            return_value=FakeSaver(),
        ),
        patch("app.worker.asyncio.sleep", side_effect=StopWorker),
    ):
        with pytest.raises(StopWorker):
            await main_worker()

    fake_app.ainvoke.assert_awaited_once()

    last_set = fake_redis.set.await_args_list[-1]
    fianl_state = json.loads(last_set.args[1])

    assert fianl_state["status"] == "waiting_approval"
    assert fianl_state["result"] is None
    assert fianl_state["error"] is None

@pytest.mark.asyncio
async def test_procesar_aprobacion_completa_tarea_exitosamente():

    job_id = "test-job-approved"

    task_data = {
        "job_id": job_id,
        "query": "Evaluate answer about the mean.",
        "status": "waiting_approval",
        "result": None,
        "error": None,
    }

    approval_data = {
        "decision": "approve",
        "feedback": "The answer is correct.",
    }

    fake_redis = AsyncMock()

    fake_redis.get.side_effect = [
        json.dumps(task_data),
        json.dumps(approval_data),
    ]

    response = "The mean is the arithmetic average."

    fake_app = AsyncMock()
    fake_app.ainvoke.return_value = {
        "messages": [
            HumanMessage(content=response)
        ]
    }

    with (
        patch("app.worker.redis_client", fake_redis),
        patch(
            "app.worker.save_human_event",
            new_callable=AsyncMock,
        ) as save_event,
        patch(
            "app.worker.get_human_events",
            new_callable=AsyncMock,
            return_value=[
                {
                    "tipo": "HumanApproval",
                    "decision": "approve",
                    "feedback": "The answer is correct.",
                }
            ],
        ),
        patch("app.worker.serialize_trace", return_value=[]),
        patch("app.worker.save_trace") as save_trace,
    ):
        await approval_processing(job_id, fake_app)

    save_event.assert_awaited_once_with(
        job_id=job_id,
        decision="approve",
        feedback="The answer is correct.",
    )

    fake_app.ainvoke.assert_awaited_once()

    args, kwargs = fake_app.ainvoke.await_args

    assert args[0].resume == {
        "decision": "approve",
        "feedback": "The answer is correct.",
    }
    assert kwargs["config"]["configurable"]["thread_id"] == job_id

    save_trace.assert_called_once()

    last_set = fake_redis.set.await_args_list[-1]
    final_state = json.loads(last_set.args[1])

    assert last_set.args[0] == f"task_status:{job_id}"
    assert final_state["status"] == "completed"
    assert final_state["result"] == response
    assert final_state["error"] is None


    fake_redis.delete.assert_any_await(f"approval_data:{job_id}")
    fake_redis.delete.assert_any_await(f"human_events:{job_id}")

@pytest.mark.asyncio
async def test_process_approval_does_nothing_if_job_does_not_exist():

    job_id = "job-non-existent"

    fake_redis = AsyncMock()
    fake_redis.get.return_value = None

    fake_app = AsyncMock()

    with patch("app.worker.redis_client", fake_redis):
        await approval_processing(job_id, fake_app)

    fake_redis.get.assert_awaited_once_with(
        f"task_status:{job_id}"
    )

    fake_app.ainvoke.assert_not_awaited()
    fake_redis.set.assert_not_awaited()

@pytest.mark.asyncio
async def test_process_approval_does_nothing_if_no_approval_is_pending():

    job_id = "test-job-no-approval"

    task_data = {
        "job_id": job_id,
        "query": "Explain the mean.",
        "status": "waiting_approval",
        "result": None,
        "error": None,
    }

    fake_redis = AsyncMock()
    fake_redis.get.side_effect = [
        json.dumps(task_data),
        None,
    ]

    fake_app = AsyncMock()

    with patch("app.worker.redis_client", fake_redis):
        await approval_processing(job_id, fake_app)

    assert fake_redis.get.await_count == 2

    fake_app.ainvoke.assert_not_awaited()
    fake_redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_approval_returns_to_waiting_for_approval():

    job_id = "test-job-second-approval"

    task_data = {
        "job_id": job_id,
        "query": "Evaluate answer about the mean.",
        "status": "waiting_approval",
        "result": None,
        "error": None,
    }

    approval_data = {
        "decision": "reject",
        "feedback": "The answer needs review.",
    }

    fake_redis = AsyncMock()

    fake_redis.get.side_effect = [
        json.dumps(task_data),
        json.dumps(approval_data),
    ]

    fake_app = AsyncMock()
    fake_app.ainvoke.return_value = {
        "__interrupt__": ["Needs another approvement"]
    }

    with (
        patch("app.worker.redis_client", fake_redis),
        patch(
            "app.worker.save_human_event",
            new_callable=AsyncMock,
        ) as save_event,
    ):
        await approval_processing(job_id, fake_app)

    save_event.assert_awaited_once_with(
        job_id=job_id,
        decision="reject",
        feedback="The answer needs review.",
    )

    last_set = fake_redis.set.await_args_list[-1]
    final_state = json.loads(last_set.args[1])

    assert last_set.args[0] == f"task_status:{job_id}"
    assert final_state["status"] == "waiting_approval"
    assert final_state["result"] is None
    assert final_state["error"] is None

    fake_redis.delete.assert_awaited_once_with(f"approval_data:{job_id}")

    fake_app.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_approval_fails_if_human_event_cannot_be_saved():

    job_id = "test-job-error-event"

    task_data = {
        "job_id": job_id,
        "query": "Evaluate an answer about the mean.",
        "status": "waiting_approval",
        "result": None,
        "error": None,
    }

    approval_data = {
        "decision": "approve",
        "feedback": "The answer is correct.",
    }

    fake_redis = AsyncMock()
    fake_redis.get.side_effect = [
        json.dumps(task_data),
        json.dumps(approval_data),
    ]

    fake_app = AsyncMock()

    with (
        patch("app.worker.redis_client", fake_redis),
        patch(
            "app.worker.save_human_event",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Redis no disponible"),
        ) as save_event,
    ):
        await approval_processing(job_id, fake_app)

    save_event.assert_awaited_once_with(
        job_id=job_id,
        decision="approve",
        feedback="The answer is correct.",
    )

    last_set = fake_redis.set.await_args_list[-1]
    final_state = json.loads(last_set.args[1])

    assert last_set.args[0] == f"task_status:{job_id}"
    assert final_state["status"] == "failed"
    assert final_state["result"] is None
    assert final_state["error"] == "Redis no disponible"

    fake_app.ainvoke.assert_not_awaited()

    fake_redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_worker_marks_task_as_failed_if_graph_fails():

    job_id = "test-job-worker-error"

    task_data = {
        "job_id": job_id,
        "query": "Explain what is the mean.",
        "status": "pending",
        "result": None,
        "error": None,
    }

    fake_redis = AsyncMock()
    fake_redis.lpop.side_effect = [
        job_id,
        None,
        None,
    ]
    fake_redis.get.return_value = json.dumps(task_data)

    fake_app = AsyncMock()
    fake_app.ainvoke.side_effect = Exception(
        "Error simulado del graph"
    )

    fake_graph = type(
        "FakeGraph",
        (),
        {"compile": lambda self, checkpointer: fake_app},
    )()

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class StopWorker(Exception):
        pass

    with (
        patch("app.worker.redis_client", fake_redis),
        patch("app.worker.graph", fake_graph),
        patch(
            "app.worker.AsyncRedisSaver.from_conn_string",
            return_value=FakeSaver(),
        ),
        patch(
            "app.worker.asyncio.sleep",
            side_effect=StopWorker,
        ),
    ):
        with pytest.raises(StopWorker):
            await main_worker()

    fake_app.ainvoke.assert_awaited_once()

    assert fake_redis.set.await_count == 2

    last_set = fake_redis.set.await_args_list[-1]
    final_state = json.loads(last_set.args[1])

    assert last_set.args[0] == f"task_status:{job_id}"
    assert final_state["status"] == "failed"
    assert final_state["result"] is None
    assert final_state["error"] == "Error simulating the graph"


@pytest.mark.asyncio
async def test_main_worker_ignores_task_that_does_not_exist_in_redis():

    job_id = "test-job-non-existent"

    fake_redis = AsyncMock()
    fake_redis.lpop.side_effect = [
        job_id,
        None, 
        None,
    ]
    fake_redis.get.return_value = None

    fake_app = AsyncMock()

    fake_graph = type(
        "FakeGraph",
        (),
        {"compile": lambda self, checkpointer: fake_app},
    )()

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class StopWorker(Exception):
        pass

    with (
        patch("app.worker.redis_client", fake_redis),
        patch("app.worker.graph", fake_graph),
        patch(
            "app.worker.AsyncRedisSaver.from_conn_string",
            return_value=FakeSaver(),
        ),
        patch(
            "app.worker.asyncio.sleep",
            side_effect=StopWorker,
        ),
    ):
        with pytest.raises(StopWorker):
            await main_worker()

    fake_redis.get.assert_awaited_once_with(f"task_status:{job_id}")

    fake_app.ainvoke.assert_not_awaited()
    fake_redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_worker_ignores_approval_for_nonexistent_job():

    job_id = "test-approval-job-non-existent"

    fake_redis = AsyncMock()

    fake_redis.lpop.side_effect = [
        None,
        job_id,
        None,
        None,
    ]

    fake_redis.get.return_value = None

    fake_app = AsyncMock()

    fake_graph = type(
        "FakeGraph",
        (),
        {"compile": lambda self, checkpointer: fake_app},
    )()

    class FakeSaver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class StopWorker(Exception):
        pass

    with (
        patch("app.worker.redis_client", fake_redis),
        patch("app.worker.graph", fake_graph),
        patch(
            "app.worker.AsyncRedisSaver.from_conn_string",
            return_value=FakeSaver(),
        ),
        patch(
            "app.worker.asyncio.sleep",
            side_effect=StopWorker,
        ),
    ):
        with pytest.raises(StopWorker):
            await main_worker()

    fake_redis.get.assert_awaited_once_with(f"task_status:{job_id}")

    fake_app.ainvoke.assert_not_awaited()
    fake_redis.set.assert_not_awaited()
