import logging
from functools import lru_cache

from openai import LengthFinishReasonError, OpenAI

from app.config import get_settings
from app.schemas import GroundedAnswer

logger = logging.getLogger(__name__)

INSUFFICIENT = "Insufficient information."

SYSTEM_PROMPT = f"""You answer questions using ONLY the context provided by the user. Each context passage is labeled with its chunk id in square brackets, like [a1b2c3d4e5f60718].

Rules (hard constraints):
1. Use ONLY the provided context. Do not use outside knowledge, and do not guess or infer beyond what the passages state.
2. If the answer is not in the context, respond with exactly: "{INSUFFICIENT}" and nothing else.
3. Cite every claim with the [chunk-id] of the passage that supports it, e.g. "The trial began in October [a1b2c3d4e5f60718]." A claim with no supporting passage must not be made.
4. Respond with JSON matching the schema: "answer" is your cited answer; "referenced_chunks" lists the chunk ids you cited (an empty list when the answer is "{INSUFFICIENT}")."""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


class GenerationError(Exception):
    pass


class MissingApiKeyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = get_settings().openai_api_key
    if not api_key:
        raise MissingApiKeyError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def generate(query: str, chunks: list[dict], client: OpenAI | None = None) -> GroundedAnswer:
    client = client or get_openai_client()
    settings = get_settings()
    model = settings.llm_model
    # gpt-5 models only accept the default temperature
    temperature = {} if model.startswith("gpt-5") else {"temperature": 0}
    try:
        completion = client.chat.completions.parse(
            model=model,
            **temperature,
            max_completion_tokens=settings.max_output_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {query}"},
            ],
            response_format=GroundedAnswer,
        )
    except LengthFinishReasonError:
        raise GenerationError("the answer was cut off by the output token limit")
    message = completion.choices[0].message
    if message.parsed is None:
        raise GenerationError(f"no valid GroundedAnswer returned (refusal={message.refusal!r})")
    return drop_unknown_citations(message.parsed, {c["chunk_id"] for c in chunks})


def drop_unknown_citations(answer: GroundedAnswer, context_ids: set[str]) -> GroundedAnswer:
    kept = [cid for cid in answer.referenced_chunks if cid in context_ids]
    dropped = [cid for cid in answer.referenced_chunks if cid not in context_ids]
    if dropped:
        logger.warning("dropped referenced_chunks not in retrieved context: %s", dropped)
    return answer.model_copy(update={"referenced_chunks": kept})
