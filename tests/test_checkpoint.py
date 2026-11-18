
import os
from uuid import uuid4

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from redis import asyncio as aioredis
from typing_extensions import TypedDict

load_dotenv()

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379",
)


class CheckpointState(TypedDict):
    mensaje: str


@pytest_asyncio.fixture
async def redis_client():
    """Connects with redis for integration test"""
    client = aioredis.from_url(
        REDIS_URL,
        decode_responses=True,
    )

    try:
        await client.ping()
        yield client
    except Exception as exc:
        pytest.fail(
            f"Connection failed with Redis at {REDIS_URL}: {exc}"
        )
    finally:
        await client.aclose()


def crear_grafo():
    """A graph to test persistence."""

    builder = StateGraph(CheckpointState)

    def save_message(state: CheckpointState):
        return {"message": state["message"]}

    builder.add_node("save", save_message)
    builder.add_edge(START, "save")
    builder.add_edge("save", END)

    return builder


@pytest.mark.asyncio
@pytest.mark.integration
async def test_checkpoint_persitence_after_reopen_checkpoint(redis_client):
    thread_id = f"test-checkpoint-{uuid4()}"
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    builder = crear_grafo()

    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)

        await app.ainvoke(
            {"message": "state persisted in Redis"},
            config=config,
        )

        estado = await app.aget_state(config)

        assert estado.values["message"] == (
            "state persisted in Redis"
        )

    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)

        estado_recuperado = await app.aget_state(config)

        assert estado_recuperado.values["message"] == ("state persisted in Redis")
        assert (estado_recuperado.config["configurable"]["thread_id"] == thread_id)
        assert estado_recuperado.created_at is not None
