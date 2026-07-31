from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import re
from typing import Iterable

from bs4 import BeautifulSoup
import feedparser
import requests


class NewsDataError(RuntimeError):
    """Raised when Tencent news data cannot be fetched or parsed."""


@dataclass(frozen=True)
class NewsItem:
    source_title: str
    url: str
    source: str
    published_at: str
    raw_summary: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def fetch_tencent_finance_news(
    urls: list[str],
    keywords: list[str],
    limit: int = 10,
    timeout: int = 12,
) -> list[NewsItem]:
    """Fetch latest Tencent finance/self-stock related news from configured public sources."""
    errors: list[str] = []
    collected: list[NewsItem] = []

    for url in urls:
        try:
            if url.startswith("tencent-feed://"):
                collected.extend(_parse_tencent_feed(url.replace("tencent-feed://", ""), timeout))
            elif _looks_like_rss(url):
                collected.extend(_parse_rss(url, timeout))
            else:
                collected.extend(_parse_web_page(url, timeout))
        except Exception as exc:  # pragma: no cover - network/data-source boundary
            errors.append(f"{url}: {exc}")

    deduped = _dedupe(collected)
    filtered = _keyword_filter(deduped, keywords)
    if len(filtered) < min(limit, len(deduped)):
        filtered = deduped
    filtered.sort(key=lambda item: item.published_at or "", reverse=True)

    if not filtered:
        error_detail = "；".join(errors) if errors else "腾讯源没有返回可用资讯"
        raise NewsDataError(f"腾讯资讯抓取失败或无匹配财经资讯：{error_detail}")

    return filtered[:limit]


def _parse_tencent_feed(channel_id: str, timeout: int) -> list[NewsItem]:
    payload = {
        "qimei36": "",
        "forward": "0",
        "base_req": {"from": "pc"},
        "flush_num": 0,
        "channel_id": channel_id,
        "device_id": "etf-recommendation-tool",
        "is_local_chlid": "",
    }
    response = requests.post(
        "https://r.inews.qq.com/web_feed/getPCList",
        headers={**REQUEST_HEADERS, "Referer": "https://news.qq.com/"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    records = data.get("data")
    if not isinstance(records, list) or not records:
        raise NewsDataError(f"腾讯频道接口返回空数据：{channel_id}")

    items: list[NewsItem] = []
    for record in records:
        title = _clean_text(str(record.get("title", "")))
        link_info = record.get("link_info") or {}
        share_info = record.get("share_info") or {}
        media_info = record.get("media_info") or {}
        link = link_info.get("url") or link_info.get("share_url") or link_info.get("short_url")
        source = (
            media_info.get("chlname")
            or media_info.get("media_name")
            or media_info.get("name")
            or "腾讯财经"
        )
        summary = (
            record.get("abstract")
            or record.get("intro")
            or share_info.get("share_title")
            or record.get("declare")
            or title
        )
        published_at = str(record.get("publish_time") or datetime.now().isoformat(timespec="seconds"))
        if title and link:
            items.append(
                NewsItem(
                    source_title=title,
                    url=str(link),
                    source=_clean_text(str(source)),
                    published_at=published_at,
                    raw_summary=_trim(str(summary), 280),
                )
            )
    return items


def _looks_like_rss(url: str) -> bool:
    return url.lower().endswith((".xml", ".rss")) or "rss" in url.lower()


def _parse_rss(url: str, timeout: int) -> list[NewsItem]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    if getattr(feed, "bozo", False) and not feed.entries:
        raise NewsDataError(f"RSS 解析失败：{url}")

    source = _clean_text(feed.feed.get("title", "腾讯财经"))
    items: list[NewsItem] = []
    for entry in feed.entries:
        title = _clean_text(entry.get("title", ""))
        link = entry.get("link", url)
        summary = _clean_html(entry.get("summary", "") or entry.get("description", ""))
        published = _entry_time(entry)
        if title and link:
            items.append(
                NewsItem(
                    source_title=title,
                    url=link,
                    source=source or "腾讯财经",
                    published_at=published,
                    raw_summary=summary,
                )
            )
    return items


def _parse_web_page(url: str, timeout: int) -> list[NewsItem]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    soup = BeautifulSoup(response.text, "html.parser")
    source = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "腾讯财经")
    items: list[NewsItem] = []

    for anchor in soup.find_all("a", href=True):
        title = _clean_text(anchor.get_text(" ", strip=True))
        href = anchor["href"]
        if not _is_likely_news_link(title, href):
            continue

        link = _normalize_url(href)
        parent_text = _clean_text(anchor.parent.get_text(" ", strip=True) if anchor.parent else "")
        items.append(
            NewsItem(
                source_title=title,
                url=link,
                source=source or "腾讯财经",
                published_at=_guess_datetime(parent_text),
                raw_summary=_trim(parent_text.replace(title, ""), 280),
            )
        )

    if not items:
        raise NewsDataError(f"页面没有解析出资讯链接：{url}")
    return items


def _entry_time(entry: object) -> str:
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None) or entry.get(attr) if hasattr(entry, "get") else None
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone().isoformat(timespec="seconds")
    for attr in ("published", "updated"):
        value = entry.get(attr, "") if hasattr(entry, "get") else ""
        if value:
            return str(value)
    return datetime.now().isoformat(timespec="seconds")


def _keyword_filter(items: Iterable[NewsItem], keywords: list[str]) -> list[NewsItem]:
    if not keywords:
        return list(items)
    normalized_keywords = [kw.lower() for kw in keywords if kw.strip()]
    filtered = []
    for item in items:
        haystack = f"{item.source_title} {item.raw_summary}".lower()
        if any(keyword.lower() in haystack for keyword in normalized_keywords):
            filtered.append(item)
    return filtered


def _dedupe(items: Iterable[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    result: list[NewsItem] = []
    for item in items:
        key = item.url or item.source_title
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _is_likely_news_link(title: str, href: str) -> bool:
    if len(title) < 8:
        return False
    if href.startswith("javascript:") or href.startswith("#"):
        return False
    return "qq.com" in href or href.startswith("/")


def _normalize_url(href: str) -> str:
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        return f"https://news.qq.com{href}"
    return href


def _guess_datetime(text: str) -> str:
    match = re.search(r"20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}", text)
    if match:
        return match.group(0).replace("年", "-").replace("月", "-").replace("/", "-").strip("-")
    return datetime.now().isoformat(timespec="seconds")


def _clean_html(value: str) -> str:
    return _clean_text(BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _trim(value: str, max_chars: int) -> str:
    value = _clean_text(value)
    return value[:max_chars]
