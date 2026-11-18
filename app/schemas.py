from enum import Enum
from operator import add
from typing import Annotated, Dict, List, Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# Agent

class AgentState(MessagesState):


    job_id: Optional[str]

    approval: Optional[bool]

    human_feedback: Optional[str]

    human_review: bool

    next_agent: Optional[str]

    contribution: Annotated[List[Dict[str, str]], add]

    steps: int

    task_completed: bool

    reviews: int

class ApprovalRequest(BaseModel):
    desition: Literal["approve", "reject"]
    feedback: str = ""

class DesitionSupervisor(BaseModel):

    next: Literal["teacher", "evaluator", "FINISH"] = Field(
        description=(
            "Next agent to invoce, or FINISH if the task is completed."
        )
    )

    reason: str = Field(
        description="Brief justification of the desition."
    )

class ConceptResult(BaseModel):

    content: str = Field(
        description="Relevant content of the document retrieval."
    )


class SourceResult(BaseModel):

    source: str = Field(
        description="Bibliographic source."
    )

    page: int = Field(
        description="Page number of the source."
    )

class EvaluationResult(BaseModel):

    evaluation: Literal["Wrong", "Incomplete", "Good", "Very good"] = Field(
        description="User knowledge evaluation."
    )

    improve: str = Field(
        description=(
            "How the answer could be improved. "
            "Si es Muy bien, indicar que no necesita mejoras."
        )
    )

# Redis
class TaskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    waiting_approval = "waiting_approval"
    completed = "completed"
    failed = "failed"

class TaskResponse(BaseModel):
    job_id: str
    status: TaskStatus

# Errors
class LLMErrorType(str, Enum):
    """Error types."""

    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"
    KEY = "key"


class LLMError(Exception):
    """Allows to classificate the error and send a message to the user."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)