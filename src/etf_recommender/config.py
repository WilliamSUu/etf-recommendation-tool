from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    tencent_news_urls: list[str]
    news_keywords: list[str]
    timezone: str = "Asia/Shanghai"
    news_limit: int = 10
    etf_recommendation_limit: int = 10
    etf_candidate_limit: int = 80
    price_refresh_seconds: int = 5


DEFAULT_TENCENT_NEWS_URLS = [
    "tencent-feed://news_news_finance",
    "https://finance.qq.com/rss/stock.xml",
    "https://finance.qq.com/rss/money.xml",
    "https://finance.qq.com/rss/finance.xml",
    "https://news.qq.com/ch/finance/",
    "https://gu.qq.com/",
]


DEFAULT_NEWS_KEYWORDS = [
    "ETF",
    "基金",
    "A股",
    "港股",
    "美股",
    "债券",
    "黄金",
    "原油",
    "商品",
    "半导体",
    "人工智能",
    "新能源",
    "银行",
    "地产",
    "医药",
    "消费",
    "军工",
    "央行",
    "利率",
    "汇率",
    "经济",
    "政策",
    "指数",
    "股票",
    "市场",
    "沪指",
    "光刻机",
    "科技",
    "上市",
    "公司",
    "资金",
]


def _split_env_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
        tencent_news_urls=_split_env_list(os.getenv("TENCENT_NEWS_URLS"), DEFAULT_TENCENT_NEWS_URLS),
        news_keywords=_split_env_list(os.getenv("NEWS_KEYWORDS"), DEFAULT_NEWS_KEYWORDS),
    )
