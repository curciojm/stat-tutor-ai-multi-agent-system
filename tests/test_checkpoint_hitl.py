
import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command

from app.graph_config import graph

load_dotenv()

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379",
)

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_checkpoint_hitl_restablish_after_reopen_checkpointer():
    thread_id = f"test-hitl-{uuid4()}"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }

    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        resultado = await app.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Evaluate my answer about the mean: "
                            "it is the sum of the values divided "
                            "by the number of observations."
                        ),
                    }
                ],
                "job_id": thread_id,
                "human_review": True,
            },
            config=config,
        )

        assert resultado.get("__interrupt__"), ("El graph no quedó interrumpido como se esperaba.")

        estado = await app.aget_state(config)
        assert estado.next, "El checkpoint no indica nodos pendientes."

    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        estado_recuperado = await app.aget_state(config)

        assert estado_recuperado.next, ("No se recuperaron los nodos pendientes.")

        resultado_final = await app.ainvoke(
            Command(
                resume={
                    "decision": "approve",
                    "feedback": "Aprobación de prueba.",
                }
            ),
            config=config,
        )

        assert not resultado_final.get("__interrupt__"), ("El graph volvió a interrumpirse.")

        estado_final = await app.aget_state(config)
        assert not estado_final.next, ("El graph no terminó después de la aprobación.")
