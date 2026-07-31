from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler

from etf_recommender.config import get_settings
from etf_recommender.services.cache import load_state
from etf_recommender.services.refresh import refresh_dashboard_data, refresh_dashboard_data_safely


_scheduler: BackgroundScheduler | None = None


SCHEDULED_TIMES = [time(9, 0), time(15, 0)]


def start_background_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    settings = get_settings()
    tz = ZoneInfo(settings.timezone)
    _scheduler = BackgroundScheduler(timezone=tz, daemon=True)
    for scheduled_time in SCHEDULED_TIMES:
        _scheduler.add_job(
            _run_scheduled_refresh,
            "cron",
            hour=scheduled_time.hour,
            minute=scheduled_time.minute,
            args=[scheduled_time.strftime("%H:%M")],
            id=f"refresh-{scheduled_time.strftime('%H%M')}",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    _scheduler.start()


def run_missed_refreshes() -> list[str]:
    settings = get_settings()
    tz = ZoneInfo(settings.timezone)
    now = datetime.now(tz)
    state = load_state()
    completed = set(state.get("completed_slots", []))
    failed_slots = set((state.get("failed_slots") or {}).keys())
    refreshed_slots: list[str] = []

    for scheduled_time in SCHEDULED_TIMES:
        slot = _slot_key(now, scheduled_time)
        if now.time() >= scheduled_time and slot not in completed and slot not in failed_slots:
            refresh_dashboard_data(
                trigger=f"missed:{scheduled_time.strftime('%H:%M')}",
                slot_key=slot,
                investment_horizon="短线",  # 自动刷新使用默认周期
            )
            refreshed_slots.append(slot)

    return refreshed_slots


def _slot_key(now: datetime, scheduled_time: time) -> str:
    return f"{now.date().isoformat()}T{scheduled_time.strftime('%H:%M')}"


def _run_scheduled_refresh(scheduled_time_text: str) -> None:
    settings = get_settings()
    tz = ZoneInfo(settings.timezone)
    hour, minute = [int(part) for part in scheduled_time_text.split(":")]
    scheduled_time = time(hour, minute)
    refresh_dashboard_data_safely(
        trigger=f"schedule:{scheduled_time_text}",
        slot_key=_slot_key(datetime.now(tz), scheduled_time),
        investment_horizon="短线",  # 自动刷新使用默认周期
    )
