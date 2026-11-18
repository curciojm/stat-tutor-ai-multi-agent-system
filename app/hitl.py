from langgraph.types import interrupt


def human_approval(state):
    desition = interrupt(
        {
            "message": "Requieres human approval to execute the evaluation.",
            "job_id": state["job_id"],
        }
    )

    approve = desition["desition"] == "approve"

    return {
        "approval": approve,
        "human_feedback": desition.get("feedback", ""),
        "human_review": not approve,
    }