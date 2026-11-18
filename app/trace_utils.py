import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage


def extract_text(message) -> str:
    """Normalized message content."""
    content = message.content

    if isinstance(content, str):
        return message

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content if isinstance(block, dict)
        )

    return str(content)

def serializar_traza(message) -> list[dict]:
    """Transform langraph messages to json."""
    traza = []

    for message in messages:
        entrada = {
            "type": type(message).__name__,
            "content": extract_text(message),
        }

        if getattr(message, "name", None):
            entrada["agente"] = message.name

        if isinstance(message, AIMessage) and message.tool_calls:
            entrada["tool_calls"] = [
                {
                    "nombre": tool_call["name"],
                    "argumentos": tool_call["args"],
                }
                for tool_call in message.tool_calls
            ]

        if isinstance(message, ToolMessage):
            entrada["herramienta"] = message.name

        traza.append(entrada)

    return traza

def guardar_traza(trace: list[dict], thread_id: str) -> None:
    """Save ReAct trace from a conversation in json format."""
    path = Path("traces") / f"{thread_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            trace,
            file,
            ensure_ascii=False,
            indent=2,
        )