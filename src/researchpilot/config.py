from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# 让 Hugging Face 等第三方库也能读取 .env 中的环境变量
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "researchpilot_chunks_v1"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 4


    # openai_api_key: SecretStr | None = None
    # openai_model: str = "gpt-5-mini"

    llm_provider: str = "deepseek"
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"

    llm_structured_max_attempts: int = 2
    llm_temperature: float = 0.0

    eval_judge_model: str | None = None
    eval_judge_max_output_tokens: int = 1800

    eval_judge_temperature: float = 0.0

    rag_top_k: int = 6
    rag_max_output_tokens: int = 1200
    rag_table_top_k_per_document: int = 2

    rag_multi_query_candidate_k: int = 6
    rag_rrf_k: int = 60

    rag_query_diversity_k_per_document: int = 2

    #######################################
    reranker_enabled: bool = False
    reranker_model: str = (
        "BAAI/bge-reranker-v2-m3"
    )
    reranker_device: str = "cuda"
    reranker_batch_size: int = 4
    reranker_max_length: int = 512
    ######################################

    ##################################################用于分析实验数据
    analysis_model: str = "deepseek-v4-flash"
    analysis_top_k_per_document: int = 8
    analysis_max_tables_per_document: int = 8
    analysis_max_output_tokens: int = 5000
    #################################################

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = PROJECT_ROOT / "data"
    max_upload_size_mb: int = 50






settings = Settings()