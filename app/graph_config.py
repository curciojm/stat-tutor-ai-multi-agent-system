from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agents.evaluador import node_evaluator
from app.agents.teacher import node_teacher
from app.agents.synthesis import node_synthesis
from app.agents.supervisor import node_supervisor
from app.hitl import human_approval
from app.schemas import AgentState


def ruting(state: AgentState) -> Literal["teacher", "evaluator", "synthesis"]:

    if state["next_agent"] == "FINISH":
        return "synthesis"

    return state["next_agent"]


def ruting_approval(state: AgentState) -> Literal["evaluator"]:
    return "evaluator"

def ruting_after_evaluator(state: AgentState) -> Literal["supervisor", "human_approval"]:

    if state.get("human_review", False):
        return "human_approval"

    return "supervisor"


grafo = StateGraph(AgentState)

grafo.add_node("supervisor", node_supervisor)
grafo.add_node("teacher", node_teacher)
grafo.add_node("evaluator", node_evaluator)
grafo.add_node("synthesis", node_synthesis)
grafo.add_node("human_approval", human_approval)


grafo.add_edge(START, "supervisor")


grafo.add_conditional_edges(
    "supervisor",
    ruting,
    {
        "teacher": "teacher",
        "evaluator": "human_approval",
        "synthesis": "synthesis",
    },
)


grafo.add_edge("teacher", "supervisor")


grafo.add_conditional_edges(
    "human_approval",
    ruting_approval,
    {
        "evaluator": "evaluator",
    },
)

grafo.add_conditional_edges(
    "evaluevaluatorador",
    ruting_after_evaluator,
    {
        "supervisor": "supervisor",
        "human_approval": "human_approval",
    },
)

grafo.add_edge("synthesis", END)