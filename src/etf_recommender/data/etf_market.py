from __future__ import annotations

from dataclasses import dataclass

import akshare as ak
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MarketDataError(RuntimeError):
    """Raised when the ETF market data source is unavailable."""


QUOTE_HOSTS = [
    "https://push2.eastmoney.com/api/qt/ulist.np/get",
    "https://push2delay.eastmoney.com/api/qt/ulist.np/get",
]


QUOTE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://quote.eastmoney.com/",
}


@dataclass(frozen=True)
class ETFQuote:
    code: str
    name: str
    price: float | None
    change_pct: float | None


def fetch_all_etf_quotes() -> pd.DataFrame:
    """Fetch all A-share exchange-traded ETF quotes from AkShare/Eastmoney."""
    try:
        raw = ak.fund_etf_spot_em()
    except Exception as exc:  # pragma: no cover - network/data-source boundary
        raise MarketDataError(f"ETF 实时行情源不可用：{exc}") from exc

    if raw is None or raw.empty:
        raise MarketDataError("ETF 实时行情源返回空数据。")

    df = raw.copy()
    rename_map = {
        "代码": "code",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "change_pct",
        "涨跌额": "change_amount",
        "成交量": "volume",
        "成交额": "turnover",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "昨收": "prev_close",
        "换手率": "turnover_rate",
        "流通市值": "market_cap",
        "总市值": "total_market_cap",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"code", "name", "price", "change_pct"}
    missing = required - set(df.columns)
    if missing:
        raise MarketDataError(f"ETF 行情字段缺失：{', '.join(sorted(missing))}")

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str)
    for column in ["price", "change_pct", "change_amount", "volume", "turnover", "open", "high", "low"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["code", "name"]).reset_index(drop=True)


def get_quote_map(quotes: pd.DataFrame) -> dict[str, ETFQuote]:
    quote_map: dict[str, ETFQuote] = {}
    for _, row in quotes.iterrows():
        code = str(row["code"]).zfill(6)
        quote_map[code] = ETFQuote(
            code=code,
            name=str(row["name"]),
            price=_safe_float(row.get("price")),
            change_pct=_safe_float(row.get("change_pct")),
        )
    return quote_map


def fetch_selected_etf_quotes(codes: list[str]) -> pd.DataFrame:
    """Fetch live quotes only for the selected ETF codes from Eastmoney."""
    normalized_codes = [str(code).zfill(6) for code in codes if str(code).strip()]
    if not normalized_codes:
        return pd.DataFrame(columns=["code", "name", "price", "change_pct"])

    secids = ",".join([f"{_eastmoney_market_prefix(code)}.{code}" for code in normalized_codes])
    params = {
        "fltt": "2",
        "invt": "2",
        "np": "1",
        "fields": "f12,f14,f2,f3",
        "secids": secids,
    }
    records = _request_selected_quote_records(params)

    rows = []
    for record in records:
        rows.append(
            {
                "code": str(record.get("f12", "")).zfill(6),
                "name": str(record.get("f14", "")),
                "price": _safe_float(record.get("f2")),
                "change_pct": _safe_float(record.get("f3")),
            }
        )

    if not rows:
        raise MarketDataError("ETF 快速行情源返回空数据。")

    return pd.DataFrame(rows).dropna(subset=["code", "name"]).reset_index(drop=True)


def _request_selected_quote_records(params: dict[str, str]) -> list[dict[str, object]]:
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.25,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    errors: list[str] = []
    for url in QUOTE_HOSTS:
        try:
            response = session.get(url, params=params, headers=QUOTE_HEADERS, timeout=8)
            response.raise_for_status()
            records = (response.json().get("data") or {}).get("diff") or []
            if records:
                return records
            errors.append(f"{url}: 返回空数据")
        except Exception as exc:  # pragma: no cover - network/data-source boundary
            errors.append(f"{url}: {exc}")

    raise MarketDataError(f"ETF 快速行情源不可用：{'；'.join(errors)}")


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _eastmoney_market_prefix(code: str) -> int:
    return 1 if code.startswith(("5", "6", "9")) else 0
