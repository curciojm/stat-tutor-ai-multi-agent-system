from langgraph.types import interrupt


def human_approval(state):
    decision = interrupt(
        {
            "message": "Requieres human approval to execute the evaluation.",
            "job_id": state["job_id"],
        }
    )

    approve = decision["decision"] == "approve"

    return {
        "approval": approve,
        "human_feedback": decision.get("feedback", ""),
        "human_review": not approve,
    }