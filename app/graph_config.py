from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agents.evaluador import node_evaluator
from app.agents.teacher import node_teacher
from app.agents.synthesis import node_synthesis
from app.agents.supervisor import node_supervisor
from app.hitl import human_approval
from app.schemas import AgentState


def routing(state: AgentState) -> Literal["teacher", "evaluator", "synthesis"]:

    if state["next_agent"] == "FINISH":
        return "synthesis"

    return state["next_agent"]


def routing_approval(state: AgentState) -> Literal["evaluator"]:
    return "evaluator"

def routing_after_evaluator(state: AgentState) -> Literal["supervisor", "human_approval"]:

    if state.get("human_review", False):
        return "human_approval"

    return "supervisor"


graph = StateGraph(AgentState)

graph.add_node("supervisor", node_supervisor)
graph.add_node("teacher", node_teacher)
graph.add_node("evaluator", node_evaluator)
graph.add_node("synthesis", node_synthesis)
graph.add_node("human_approval", human_approval)


graph.add_edge(START, "supervisor")


graph.add_conditional_edges(
    "supervisor",
    routing,
    {
        "teacher": "teacher",
        "evaluator": "human_approval",
        "synthesis": "synthesis",
    },
)


graph.add_edge("teacher", "supervisor")


graph.add_conditional_edges(
    "human_approval",
    routing_approval,
    {
        "evaluator": "evaluator",
    },
)

graph.add_conditional_edges(
    "evaluator",
    routing_after_evaluator,
    {
        "supervisor": "supervisor",
        "human_approval": "human_approval",
    },
)

graph.add_edge("synthesis", END)