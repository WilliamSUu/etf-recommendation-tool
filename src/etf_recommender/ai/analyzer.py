from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math

import numpy as np
import pandas as pd

from etf_recommender.ai.deepseek_client import DeepSeekClient, DeepSeekError
from etf_recommender.data.tencent_news import NewsItem


class AnalysisError(RuntimeError):
    """Raised when AI analysis cannot produce usable output."""


@dataclass(frozen=True)
class StructuredNews:
    title: str
    summary: str
    key_points: list[str]
    assets: list[str]
    source: str
    published_at: str
    url: str
    keywords: list[str]
    duration_score: float = 5.0  # 资讯影响长度：0-10，默认5

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ETFRecommendation:
    code: str
    name: str
    relevance_score: float
    impact_score: float
    duration_score: float
    total_score: float
    reason: str
    related_news: list[str]
    investment_dimension: str = "综合"  # 投资价值维度：基本面/技术面/资金面/行业景气/综合

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


THEME_KEYWORDS: dict[str, list[str]] = {
    "黄金": ["黄金", "金", "贵金属", "有色"],
    "原油": ["原油", "石油", "油气", "能源", "商品"],
    "煤炭": ["煤炭", "能源", "资源"],
    "有色": ["有色", "铜", "铝", "稀土", "资源"],
    "半导体": ["半导体", "芯片", "集成电路", "科创", "电子"],
    "人工智能": ["人工智能", "AI", "软件", "计算机", "云计算", "通信"],
    "新能源": ["新能源", "电池", "光伏", "风电", "储能", "电力"],
    "汽车": ["汽车", "新能源车", "智能车"],
    "医药": ["医药", "医疗", "创新药", "生物"],
    "消费": ["消费", "食品", "酒", "旅游", "零售"],
    "银行": ["银行", "金融", "证券", "保险", "红利"],
    "地产": ["地产", "房地产", "基建", "建筑"],
    "军工": ["军工", "国防"],
    "农业": ["农业", "养殖", "粮食"],
    "港股": ["港股", "恒生", "H股", "中概", "互联网"],
    "美股": ["美股", "纳指", "标普", "纳斯达克"],
    "日经": ["日经", "日本"],
    "债券": ["债", "国债", "信用债", "可转债", "利率"],
    "货币": ["货币", "现金", "短融"],
    "红利": ["红利", "央企", "价值", "低波"],
    "宽基": ["沪深300", "中证500", "中证1000", "上证50", "创业板", "科创", "A500"],
}


# 投资周期配置
INVESTMENT_HORIZON_CONFIGS: dict[str, dict[str, object]] = {
    "超短线": {
        "days": 5,
        "price_min": 0.0,
        "price_max": 999999.0,
        "price_min_loss": -1.0,
        "turnover_min": 2.0,
        "turnover_max": 15.0,
        "duration_range": (0.0, 2.0),
        "relevance_boost": 0.30,
    },
    "短线": {
        "days": 15,
        "price_min": 0.0,
        "price_max": 999999.0,
        "price_min_loss": -2.0,
        "turnover_min": 1.0,
        "turnover_max": 8.0,
        "duration_range": (3.0, 7.0),
        "relevance_boost": 0.25,
    },
    "中线": {
        "days": 45,
        "price_min": 0.0,
        "price_max": 999999.0,
        "price_min_loss": -3.0,
        "turnover_min": 0.5,
        "turnover_max": 3.0,
        "duration_range": (8.0, 15.0),
        "relevance_boost": 0.20,
    },
    "长线": {
        "days": 200,
        "price_min": 0.0,
        "price_max": 999999.0,
        "price_min_loss": -5.0,
        "turnover_min": 0.5,
        "turnover_max": 2.0,
        "duration_range": (15.0, 60.0),
        "relevance_boost": 0.30,
    },
}


def structure_news(client: DeepSeekClient, news_items: list[NewsItem]) -> list[StructuredNews]:
    raw_items = [item.to_dict() for item in news_items]
    system_prompt = (
        "你是专业中文财经资讯编辑和 ETF 投资研究助理。请只输出 JSON，不要输出解释。"
        "你需要把腾讯财经资讯整理为投资研究面板可用的结构化数据，摘要要帮助用户理解事实背景和市场含义。"
    )
    user_prompt = f"""
请处理以下资讯列表，返回 JSON 对象，格式为：
{{
  "items": [
    {{
      "title": "重新提炼后的中文标题，简洁客观，不超过28个中文字符",
      "summary": "120-200字中文摘要，不重复标题，说明事件背景、核心事实、可能影响和后续观察点",
      "key_points": ["关键影响点1", "关键影响点2", "关键影响点3"],
      "assets": ["涉及资产/行业/主题"],
      "keywords": ["用于ETF匹配的关键词"],
      "duration_score": 5.0
    }}
  ]
}}

要求：
- items 数量必须与输入一致，顺序保持一致。
- title 和 summary 必须明显不同；summary 不得只是改写或复述标题。
- 摘要长度尽量在 120-200 个中文字符之间，不得超过 200 个中文字符。
- 摘要必须包含：发生了什么、为什么重要、可能影响哪些市场或行业、投资者后续应观察什么。
- 如果原始资讯信息很短，也要基于已给事实做谨慎扩展，但不能添加不存在的具体数字、政策结论或交易建议。
- key_points 至少 2 条，优先写对 ETF 主题、行业景气度、风险偏好、利率/汇率/商品价格/资金面的影响。
- 不要编造原文没有的信息。

duration_score 评分规则（0-10）：
- 0-2 分：1天内的突发消息、个股异动，市场反应快速衰减
- 3-5 分：1-2周的短期热点、公司公告、行业新闻
- 6-8 分：1个月左右的中期主题、产业方向变化、政策预期
- 8-10 分：3个月以上的长期趋势、重大政策变化、产业结构升级

输入资讯：
{json.dumps(raw_items, ensure_ascii=False)}
"""
    data = client.complete_json(system_prompt, user_prompt)
    ai_items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(ai_items, list) or len(ai_items) != len(news_items):
        raise AnalysisError("DeepSeek 新闻结构化结果数量不匹配。")

    structured: list[StructuredNews] = []
    for source_item, ai_item in zip(news_items, ai_items):
        structured.append(
            StructuredNews(
                title=str(ai_item.get("title", source_item.source_title)).strip(),
                summary=str(ai_item.get("summary", source_item.raw_summary)).strip()[:200],
                key_points=_string_list(ai_item.get("key_points"))[:5],
                assets=_string_list(ai_item.get("assets"))[:8],
                source=source_item.source,
                published_at=source_item.published_at,
                url=source_item.url,
                keywords=_string_list(ai_item.get("keywords"))[:12],
                duration_score=_score(ai_item.get("duration_score", 5.0)),
            )
        )
    return structured


def hard_filter_etf_candidates(
    etf_quotes: pd.DataFrame,
    investment_horizon: str = "短线",
) -> pd.DataFrame:
    """
    根据投资周期应用硬性筛选规则，筛选出横盘+活跃换手的 ETF。

    Args:
        etf_quotes: ETF 行情数据，需要包含以下列：
                   - code, name, price, change_pct, turnover
        investment_horizon: 投资周期，可选 "超短线", "短线", "中线", "长线"

    Returns:
        符合条件的 ETF DataFrame，已按候选得分排序
    """
    if etf_quotes.empty:
        return etf_quotes

    if investment_horizon not in INVESTMENT_HORIZON_CONFIGS:
        investment_horizon = "短线"

    config = INVESTMENT_HORIZON_CONFIGS[investment_horizon]
    df = etf_quotes.copy()

    # 必需的列检查
    required_cols = ["change_pct", "turnover"]
    if not all(col in df.columns for col in required_cols):
        return df

    # 应用硬性筛选
    # 1. 价格走势：在允许范围内
    price_loss_limit = float(config.get("price_min_loss", -2.0))
    price_max_limit = float(config.get("price_max", 5.0))
    price_mask = (df["change_pct"] >= price_loss_limit) & (df["change_pct"] <= price_max_limit)

    # 2. 换手率：在活跃范围内
    turnover_min = float(config.get("turnover_min", 0.5))
    turnover_max = float(config.get("turnover_max", 5.0))
    turnover_mask = (df["turnover"] >= turnover_min) & (df["turnover"] <= turnover_max)

    # 应用双重条件过滤
    filtered = df[price_mask & turnover_mask].copy()

    if filtered.empty:
        # 如果没有完全符合的，放宽条件返回价格符合的
        return df[price_mask].copy() if price_mask.any() else df.head(20)

    # 标记候选 ETF
    filtered["hard_filter_passed"] = True

    return filtered.reset_index(drop=True)


def build_etf_candidates(news: list[StructuredNews], etf_quotes: pd.DataFrame, limit: int) -> pd.DataFrame:
    context_terms = _context_terms(news)
    if etf_quotes.empty:
        return etf_quotes

    df = etf_quotes.copy()
    df["candidate_score"] = df["name"].apply(lambda name: _name_match_score(str(name), context_terms))

    if "turnover" in df.columns:
        turnover_rank = df["turnover"].fillna(0).rank(pct=True)
        df["candidate_score"] = df["candidate_score"] + turnover_rank * 0.8

    matched = df[df["candidate_score"] > 0].sort_values("candidate_score", ascending=False)
    if len(matched) < min(limit, 30):
        liquidity_top = df.sort_values("turnover" if "turnover" in df.columns else "price", ascending=False)
        matched = pd.concat([matched, liquidity_top]).drop_duplicates(subset=["code"])

    return matched.head(limit).reset_index(drop=True)


def recommend_etfs_with_horizon_weighting(
    recommendations: list[ETFRecommendation],
    news: list[StructuredNews],
    investment_horizon: str = "短线",
) -> list[ETFRecommendation]:
    """
    根据用户投资周期对 ETF 推荐进行加权调整。

    第二阶段评分逻辑：
    1. 如果资讯的 duration_score 符合用户周期偏好，提升相关 ETF 的 relevance_score
    2. 重新计算 total_score
    3. 重新排序

    Args:
        recommendations: 基础推荐列表（来自 recommend_etfs）
        news: 结构化资讯列表（含 duration_score）
        investment_horizon: 用户选择的投资周期

    Returns:
        加权后的推荐列表
    """
    if not recommendations or investment_horizon not in INVESTMENT_HORIZON_CONFIGS:
        return recommendations

    config = INVESTMENT_HORIZON_CONFIGS[investment_horizon]
    duration_range = tuple(config.get("duration_range", (3.0, 6.0)))
    relevance_boost = float(config.get("relevance_boost", 0.25))

    # 建立资讯标题 → duration_score 的映射
    news_duration_map = {n.title: n.duration_score for n in news}

    adjusted_recommendations = []

    for rec in recommendations:
        # 获取相关资讯的平均 duration_score
        related_durations = []
        for news_title in rec.related_news:
            if news_title in news_duration_map:
                related_durations.append(news_duration_map[news_title])

        # 判断是否需要提升 relevance_score
        if related_durations:
            avg_duration = sum(related_durations) / len(related_durations)
            # 如果平均 duration 在用户周期范围内，提升相关度
            if duration_range[0] <= avg_duration <= duration_range[1]:
                boosted_relevance = min(10.0, rec.relevance_score * (1 + relevance_boost))
            else:
                boosted_relevance = rec.relevance_score
        else:
            boosted_relevance = rec.relevance_score

        # 重新计算综合分
        new_total_score = round(
            0.4 * boosted_relevance + 0.35 * rec.impact_score + 0.25 * rec.duration_score,
            1,
        )

        adjusted_rec = ETFRecommendation(
            code=rec.code,
            name=rec.name,
            relevance_score=round(boosted_relevance, 1),
            impact_score=rec.impact_score,
            duration_score=rec.duration_score,
            total_score=new_total_score,
            reason=rec.reason,
            related_news=rec.related_news,
        )
        adjusted_recommendations.append(adjusted_rec)

    # 重新按 total_score 排序
    adjusted_recommendations.sort(key=lambda r: r.total_score, reverse=True)
    return adjusted_recommendations


def recommend_etfs(
    client: DeepSeekClient,
    news: list[StructuredNews],
    candidates: pd.DataFrame,
    limit: int,
) -> list[ETFRecommendation]:
    if candidates.empty:
        raise AnalysisError("没有可用于 AI 精排的 ETF 候选池。")

    candidate_records = []
    columns = ["code", "name", "price", "change_pct", "turnover", "candidate_score"]
    for _, row in candidates.iterrows():
        candidate_records.append(
            {column: _json_safe(row.get(column)) for column in columns if column in candidates.columns}
        )

    news_records = [item.to_dict() for item in news]
    system_prompt = (
        "你是中国 ETF 投资研究助手。请只输出 JSON，不要输出解释。"
        "你的任务是基于今日资讯环境，对候选 ETF 做相关性、影响力度、影响长度三维评分。"
        "重要：推荐理由必须明确标注投资价值维度标签。"
    )
    user_prompt = f"""
请基于当前资讯环境，从候选 ETF 中选出最值得关注的前 {limit} 只。

评分规则：
- relevance_score: 相关度，0-10，资讯与 ETF 标的/主题/资产类别的关联强度。
- impact_score: 影响力度，0-10，资讯可能造成的价格或资金关注变化强度。
- duration_score: 影响长度，0-10，影响可能持续的时间长度。
- total_score: 综合分，0-10，可按 0.4*相关度 + 0.35*影响力度 + 0.25*影响长度。
- reason: 推荐理由，80-120字中文，必须明确标注投资价值维度。
- related_news: 引用相关资讯标题，最多 3 条。

🔥 投资价值维度说明（必须在 reason 中明确标注）：
1. 【基本面驱动】：基于ETF持仓企业的盈利能力、估值、增长前景等基本数据
   示例：基本面: 产业链景气度上升，持仓公司净利润同比增长30%以上

2. 【技术面驱动】：基于价格趋势、支撑阻力、均线系统等技术指标
   示例：技术面: 已突破前期高点，均线系统多头排列，存在继续上涨空间

3. 【资金面/情绪驱动】：基于市场情绪、资金流向、融资数据等市场情绪因素
   示例：资金面: 北向资金持续净买入，机构持仓比例处于历史高位

4. 【行业景气驱动】：基于行业基本面数据，如订单、产能、政策等行业趋势
   示例：行业景气: 新能源装机容量快速增长，产业链订单饱满

硬性要求：
- 只能从候选 ETF 列表中选择，禁止编造代码或名称。
- 不要输出价格判断或收益承诺。
- 分数允许一位小数。
- 推荐理由必须包含以下结构（按顺序）：
  1. 主导维度标签（从上述4种维度中选1-2个为主）
  2. 该维度下的具体事实或数据
  3. 与候选ETF的对应关系

示例推荐理由：
"【基本面+行业景气】芯片产业链景气度向上，主要持仓企业ROE环比提升，
当前估值处于历史中位数水平。资金已形成底部堆积，换手率提升表明机构承接。"

- 返回 JSON 对象，格式：
{{
  "recommendations": [
    {{
      "code": "ETF代码",
      "name": "ETF名称",
      "relevance_score": 0,
      "impact_score": 0,
      "duration_score": 0,
      "total_score": 0,
      "reason": "【维度标签】具体推荐理由...",
      "related_news": ["资讯标题"]
    }}
  ]
}}

当前资讯环境（含资讯影响长度评分）：
{json.dumps(news_records, ensure_ascii=False)}

候选 ETF（已通过硬性筛选：横盘+活跃换手）：
{json.dumps(candidate_records, ensure_ascii=False)}
"""
    data = client.complete_json(system_prompt, user_prompt)
    raw_recommendations = data.get("recommendations") if isinstance(data, dict) else None
    if not isinstance(raw_recommendations, list):
        raise AnalysisError("DeepSeek ETF 推荐结果格式不正确。")

    valid_codes = set(candidates["code"].astype(str).str.zfill(6))
    recommendations: list[ETFRecommendation] = []
    for item in raw_recommendations:
        code = str(item.get("code", "")).zfill(6)
        if code not in valid_codes:
            continue
        reason = str(item.get("reason", "")).strip()[:100]
        investment_dimension = _extract_investment_dimension(reason)
        recommendations.append(
            ETFRecommendation(
                code=code,
                name=str(item.get("name", candidates.loc[candidates["code"] == code, "name"].iloc[0])),
                relevance_score=_score(item.get("relevance_score")),
                impact_score=_score(item.get("impact_score")),
                duration_score=_score(item.get("duration_score")),
                total_score=_score(item.get("total_score")),
                reason=reason,
                related_news=_string_list(item.get("related_news"))[:3],
                investment_dimension=investment_dimension,
            )
        )

    if not recommendations:
        raise AnalysisError("DeepSeek 没有返回任何有效 ETF 推荐。")

    recommendations.sort(key=lambda rec: rec.total_score, reverse=True)
    return recommendations[:limit]


def _context_terms(news: list[StructuredNews]) -> set[str]:
    terms: set[str] = set()
    for item in news:
        for value in [item.title, item.summary, *item.key_points, *item.assets, *item.keywords]:
            cleaned = str(value).strip()
            if cleaned:
                terms.add(cleaned)
            for theme, expansion in THEME_KEYWORDS.items():
                if theme.lower() in cleaned.lower() or any(token.lower() in cleaned.lower() for token in expansion):
                    terms.update(expansion)
                    terms.add(theme)
    return {term for term in terms if len(term) >= 2}


def _name_match_score(name: str, terms: set[str]) -> float:
    score = 0.0
    lower_name = name.lower()
    for term in terms:
        lower_term = term.lower()
        if lower_term and lower_term in lower_name:
            score += 3.0 if len(term) >= 3 else 1.5
    return score


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _score(value: object) -> float:
    try:
        number = float(value)
        if math.isnan(number):
            return 0.0
        return round(float(np.clip(number, 0, 10)), 1)
    except (TypeError, ValueError):
        return 0.0


def _json_safe(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_investment_dimension(reason: str) -> str:
    """
    从推荐理由中提取投资价值维度标签。

    Args:
        reason: 推荐理由字符串

    Returns:
        投资价值维度标签
    """
    reason_lower = reason.lower()

    # 检查各个维度的关键词
    if "【基本面" in reason or "基本面" in reason:
        return "基本面"
    elif "【技术面" in reason or "技术面" in reason or "趋势" in reason or "突破" in reason or "均线" in reason:
        return "技术面"
    elif "【资金面" in reason or "【情绪" in reason or "资金面" in reason or "北向" in reason or "融资" in reason or "持仓" in reason:
        return "资金面"
    elif "【行业景气" in reason or "【行业" in reason or "行业景气" in reason or "行业" in reason or "产业" in reason:
        return "行业景气"
    else:
        return "综合"
