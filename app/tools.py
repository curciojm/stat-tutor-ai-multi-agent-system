from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity

from app.db_config import EMBEDDINGS
from app.errors import classify_error
from app.logging_config import logger
from app.retriever import hybrid_retriever
from app.schemas import ConceptResult, EvaluationResult, SourceResult


@tool
async def search_concept(consulta: str) -> list[ConceptResult]:
    """
    Search for theoretical information about a concept in statistics
    or research methodology.

    Use this tool when the user requests a definition
    or explanation of a concept.

    Return only the relevant content.
    """
    try:
        logger.info("Executing search_concept: %s", consulta)

        docs = await hybrid_retriever.ainvoke(consulta)

        return [
            ConceptResult(
                content=doc.page_content,
            )
            for doc in docs[:5]
        ]

    except Exception as e:
        logger.exception(
            "ERROR in search_concept"
        )

        error = classify_error(e)

        logger.error(
            "Error while search_concept was executing: %s",
            error.message,
        )

        raise error

@tool
async def search_source(tema: str) -> list[SourceResult]:
    """
    Search the knowledge base to locate relevant information
    about a topic in statistics or research methodology.

    Use this tool when the user asks where to find
    information, which source to consult, or on which page a topic appears.

    Return the corresponding source and page number.
    """
    try:
        logger.info("Executing search_source: %s", tema)

        docs = await hybrid_retriever.ainvoke(tema)

        return [
            SourceResult(
                source=doc.metadata.get("source", "unknown"),
                page=int(doc.metadata.get("pagina", 0)),
            )
            for doc in docs[:5]
        ]

    except Exception as e:
        error = classify_error(e)
        logger.error(
            "Error while search_source was executing: %s",
            error.message,
        )
        raise error

THRESHOLD_INCOMPLETE = 0.55
THRESHOLD_GOOD = 0.70
THRESHOLD_VERY_GOOD = 0.90

@tool
async def evaluate_concept(tema: str, user_response: str) -> EvaluationResult:
    """
    Evaluate the user's answer about a concept in statistics
    or research methodology using semantic similarity with information
    retrieved from the knowledge base.

    The evaluation uses heuristic thresholds established
    for educational purposes:

        0.00 - 0.54 → Poor
        0.55 - 0.69 → Incomplete
        0.70 - 0.89 → Good
        0.90 - 1.00 → Very Good

    These thresholds have not been empirically calibrated, and
    semantic similarity alone does not constitute a definitive
    measure of the conceptual quality of an answer.
    """
    try:
        logger.info(
            "Executing evaluate_concept: %s",
            tema,
        )

        docs = await hybrid_retriever.ainvoke(tema)

        embedding_response = await EMBEDDINGS.aembed_query(
            user_response
        )

        embedding_docs = await EMBEDDINGS.aembed_documents(
            [doc.page_content for doc in docs]
        )

        similarity = cosine_similarity(
            [embedding_response],
            embedding_docs
        )[0]

        similarity = float(max(similarity))
        logger.info(
            "The semantic similarity score was: %.3f",
            similarity,
        )

        if similarity < THRESHOLD_INCOMPLETE:
            evaluation = "Poor"
        elif similarity < THRESHOLD_GOOD:
            evaluation = "Incomplete"
        elif similarity < THRESHOLD_VERY_GOOD:
            evaluation = "Good"
        else:
            evaluation = "Very good"

        return EvaluationResult(
            evaluation=evaluation,
            improve=(
                f"The semantic similarity score was "
                f"{similarity:.2f}."
            ),
        )
    
    except Exception as e:
        logger.exception(
            "ERROR evaluate_concept"
    )

        error = classify_error(e)

        logger.error(
            "Error while evaluate_concept was executing: %s",
            error.message,
        )

        raise error