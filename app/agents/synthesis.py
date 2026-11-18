from langchain_core.messages import AIMessage, HumanMessage

from app.errors import classify_error
from app.logging_config import logger
from app.models import get_model
from app.schemas import AgentState


async def node_synthesis(state: AgentState) -> dict:

    text_contributions = "\n".join(
        f"- [{c['agent']}] {c['contribution']}"
        for c in state.get("contributions", [])
    )

    question = next(
        message.content
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )


    prompt = (
        f"Original question: {question}\n\n"
        f"Team contributions:\n{text_contributions}\n\n"
        "Write a clear and concise final answer by combining "
        "the team's relevant contributions. "
        "Do not mention the agent architecture or internal process."
    )

    last_error = None

    for provider in ["openai", "anthropic", "gemini"]:
        try:
            logger.info(
                f"Trying to initialize an LLM provider for the synthesis: {provider}"
            )

            llm = get_model(provider)
            final_response = await llm.ainvoke(prompt)

            logger.info(
                f"LLM Provider successfully used by synthesis: {provider}"
            )

            return {
                "messages": [
                    AIMessage(
                        content=final_response.content,
                        name="synthesis",
                    )
                ]
            }

        except Exception as e:
            last_error = e
            error = classify_error(e)

            logger.warning(
                f"The provider {provider} failed to synthesis: {error}"
            )

            continue

    raise classify_error(last_error)