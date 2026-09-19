from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from app.config import get_settings


class DefaultTemperatureChatOpenAI(ChatOpenAI):
    """ragas always sends a tiny temperature; the judge model only accepts its default."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("temperature", None)
        return payload


def get_judge_llm() -> LangchainLLMWrapper:
    s = get_settings()
    return LangchainLLMWrapper(
        DefaultTemperatureChatOpenAI(model=s.judge_model, api_key=s.openai_api_key, max_retries=2, timeout=120)
    )


def get_judge_embeddings() -> LangchainEmbeddingsWrapper:
    s = get_settings()
    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=s.judge_embedding_model, api_key=s.openai_api_key)
    )
