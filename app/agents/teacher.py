from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from app.errors import classify_error
from app.logging_config import logger
from app.models import get_model
from app.schemas import AgentState
from app.tools import search_concept, search_source

TEACHER_PROMPT = """
You are a university professor specializing in research
methodology and statistics.

Your role is to answer questions using the available knowledge
base through the provided tools.

1. If the user requests a definition or explanation of a
   concept, use 'buscar_concepto'.

2. If the user asks where to find information, which source
   to consult, or on which page a topic appears, use 'buscar_fuente'.

3. Do not make up information when the answer depends on the
   available documents.

4. If the tools do not provide sufficient information,
   state this clearly.

5. When a query involves more than one concept, retrieve
   information about each concept before responding.

6. Respond in a clear, natural, and educational manner.

TOOL USAGE RULES:

- Use only one tool at a time.
- Wait for a tool's result before requesting another one.
- Do not request simultaneous calls to different tools.
- Do not call 'search_concept' more than once during the same execution.
- Use the information obtained from a tool before making
  another request.
"""

async def node_teacher(state: AgentState) -> dict:
    task = state["messages"][-1].content

    last_error = None

    for provider in ["openai", "anthropic", "gemini"]:
        try:
            logger.info(
                "Trying to initialize an LLM provider for the teacher: %s",
                provider,
            )

            agent_teacher = create_agent(
                model=get_model(provider),
                tools=[search_concept, search_source],
                system_prompt=TEACHER_PROMPT,
            )

            result = await agent_teacher.ainvoke(
                {"messages": [HumanMessage(content=task)]}
            )

            response = result["messages"][-1].content

            logger.info(
                "LLM Provider successfully used by evaluator: %s",
                provider,
            )

            return {
                "messages": [
                    AIMessage(
                        content=response,
                        name="teacher",
                    )
                ],
                "contribuciones": [
                    {
                        "agente": "teacher",
                        "contribution": response,
                    }
                ],
                "steps": state.get("steps", 0) + 1,
            }

        except Exception as e:
            last_error = e

            error = classify_error(e)

            logger.warning(
                "The provider %s failed to teacher: %s",
                provider,
                error.message,
            )

            continue

    logger.error(
        "All LLM failed to teacher"
    )

    raise classify_error(last_error)