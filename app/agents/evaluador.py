from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from app.errors import classify_error
from app.logging_config import logger
from app.models import get_model
from app.schemas import AgentState
from app.tools import search_concept, search_source, evaluate_concept

EVALUATOR_PROMPT = """
You are an agent specialized in evaluating students' answers
about research methodology and statistics concepts.

YOUR MAIN TASK:

Evaluate the student's explanation using the available tools
and construct a final response addressed to the student.

MANDATORY WORKFLOW:

1. First, use 'evaluate_concept' to evaluate the user's answer.

2. Preserve the result returned by 'evaluate_concept'.
   The possible categories are exactly:
   "Poor", "Incomplete", "Good", and "Very Good".

3. If the evaluation is "Poor", "Incomplete", or "Good",
   use 'buscar_fuente' to locate where the student can study
   the concept or the information they need.

4. If you need to explain the correct formulation or identify
   missing conceptual information, use 'buscar_concepto'.

5. Do not use a tool if the available information is already
   sufficient to construct the response.

FINAL RESPONSE:

The final response must integrate the information obtained from
the tools and be addressed to the student.

The first line of the response must always explicitly state
the evaluation returned by 'evaluate_concept', using exactly
one of the following categories:

Evaluation: Poor
Evaluation: Incomplete
Evaluation: Good
Evaluation: Very Good

After stating the evaluation:

- Explain what is correct or incorrect in the student's answer.
- Indicate what should be improved or provide a correct formulation.
- When applicable, include the source and page numbers obtained
  through 'search_source'.
- If 'search_concept' was used, integrate that information
  into the explanation.

IMPORTANT RULES:

- Never replace the evaluation returned by 'evaluate_concept'
  with the result of another tool.
- Never respond solely with the result of 'buscar_fuente'
  or 'buscar_concepto'.
- Do not modify or invent the category returned by
  'evaluate_concept'.
- Do not invent sources, page numbers, or bibliographic information.
- If the evaluation is "Poor", clearly state that the answer
  does not correctly address the evaluated concept.
- If the evaluation is "Incomplete", identify the conceptual
  elements missing from the explanation.
- If the evaluation is "Good", indicate which aspects are correct
  and what could be added or clarified.
- If the evaluation is "Very Good", indicate that the explanation
  is adequate and mention only minor improvements, if applicable.
- If there is insufficient information to evaluate the answer,
  state this clearly.

TOOL USAGE RULES:

- Use only one tool at a time.
- Wait for a tool's result before requesting another one.
- Do not request simultaneous calls to different tools.
- Do not call 'buscar_concepto' more than once during the same execution.
- Use the information obtained from a tool before making
  another request.

HUMAN REVIEW:

If the task includes a review or feedback provided by a human supervisor:

- Consider the feedback as additional information for reconsidering
  the evaluation.
- Re-evaluate the student's answer using 'evaluate_concept'.
- Do not assume that the category implicitly suggested by the
  feedback is necessarily correct.
- The final evaluation must always be the category returned by
  'evaluate_concept'.
- Use the human feedback to identify aspects of the answer
  that need to be reviewed.
"""

async def node_evaluator(state: AgentState) -> dict:

    task = state["messages"][-1].content

    feedback = state.get("human_feedback")

    if feedback:
        task = f"""
{task}

HUMAN REVIEW OF THE PREVIOUS EVALUATION:

A human supervisor reviewed the previous evaluation and provided the following feedback:

{feedback}

You must reconsider the student's answer taking this feedback into account.
Use the available tools again to perform a new evaluation.
"""

    last_error = None

    for provider in ["openai", "anthropic", "gemini"]:
        try:
            logger.info(
                "Trying to initialize an LLM provider for the evaluator: %s",
                provider,
            )

            agent_evaluator = create_agent(
                model=get_model(provider),
                tools=[
                    evaluate_concept,
                    search_source,
                    search_concept,
                ],
                system_prompt=EVALUATOR_PROMPT,
            )

            result = await agent_evaluator.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=task)
                    ]
                }
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
                        name="evaluator",
                    )
                ],
                "contributions": [
                    {
                        "agent": "evaluator",
                        "contribution": response,
                    }
                ],
                "steps": state.get("steps", 0) + 1,
            }

        except Exception as e:
            last_error = e

            error = classify_error(e)

            logger.warning(
                "The provider %s failed to evaluator: %s",
                provider,
                error.message,
            )

            continue

    logger.error(
        "All LLM failed to evaluator"
    )

    raise classify_error(last_error)