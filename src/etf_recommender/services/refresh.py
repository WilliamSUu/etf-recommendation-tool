from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import traceback

from etf_recommender.ai.analyzer import (
    build_etf_candidates,
    hard_filter_etf_candidates,
    recommend_etfs,
    recommend_etfs_with_horizon_weighting,
    structure_news,
)
from etf_recommender.ai.deepseek_client import DeepSeekClient
from etf_recommender.config import get_settings
from etf_recommender.data.etf_market import fetch_all_etf_quotes
from etf_recommender.data.tencent_news import fetch_tencent_finance_news
from etf_recommender.services.cache import (
    NEWS_CACHE_FILE,
    RECOMMENDATIONS_CACHE_FILE,
    record_completed_slot,
    record_failed_slot,
    write_json,
)


class RefreshError(RuntimeError):
    """Raised when a full dashboard refresh fails."""


def refresh_dashboard_data(
    trigger: str,
    slot_key: str | None = None,
    investment_horizon: str = "短线",
) -> dict[str, object]:
    settings = get_settings()
    refreshed_at = datetime.now().isoformat(timespec="seconds")

    try:
        raw_news = fetch_tencent_finance_news(
            urls=settings.tencent_news_urls,
            keywords=settings.news_keywords,
            limit=settings.news_limit,
        )
        client = DeepSeekClient(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
        )
        structured_news = structure_news(client, raw_news)
        etf_quotes = fetch_all_etf_quotes()

        # 新增：硬性筛选（根据用户投资周期）
        hard_filtered = hard_filter_etf_candidates(etf_quotes, investment_horizon)

        # 原有逻辑：关键词匹配筛选
        candidates = build_etf_candidates(structured_news, hard_filtered, settings.etf_candidate_limit)

        # 第一阶段：基础 AI 评分
        recommendations = recommend_etfs(
            client=client,
            news=structured_news,
            candidates=candidates,
            limit=settings.etf_recommendation_limit,
        )

        # 第二阶段：根据用户投资周期进行加权调整
        recommendations = recommend_etfs_with_horizon_weighting(
            recommendations=recommendations,
            news=structured_news,
            investment_horizon=investment_horizon,
        )

        news_payload = {
            "refreshed_at": refreshed_at,
            "trigger": trigger,
            "investment_horizon": investment_horizon,
            "items": [item.to_dict() for item in structured_news],
        }
        recommendation_payload = {
            "refreshed_at": refreshed_at,
            "trigger": trigger,
            "investment_horizon": investment_horizon,
            "items": [item.to_dict() for item in recommendations],
            "candidate_count": int(len(candidates)),
        }
        write_json(NEWS_CACHE_FILE, news_payload)
        write_json(RECOMMENDATIONS_CACHE_FILE, recommendation_payload)
        if slot_key:
            record_completed_slot(slot_key)
        return {
            "news": news_payload,
            "recommendations": recommendation_payload,
        }
    except Exception as exc:
        if slot_key:
            record_failed_slot(slot_key, str(exc))
        raise RefreshError(f"刷新失败：{exc}") from exc


def refresh_dashboard_data_safely(
    trigger: str,
    slot_key: str | None = None,
    investment_horizon: str = "短线",
) -> None:
    try:
        refresh_dashboard_data(
            trigger=trigger,
            slot_key=slot_key,
            investment_horizon=investment_horizon,
        )
    except Exception:
        error_payload = {
            "failed_at": datetime.now().isoformat(timespec="seconds"),
            "trigger": trigger,
            "traceback": traceback.format_exc(),
        }
        write_json(RECOMMENDATIONS_CACHE_FILE.with_name("last_refresh_error.json"), error_payload)

