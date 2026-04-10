"""Alert sender — webhook / Slack / LINE notifications."""

from __future__ import annotations

import json
import logging

import httpx

import config

logger = logging.getLogger(__name__)


async def send_alert(event: dict) -> None:
    """Send alert to all configured channels."""
    tasks = []
    if config.ALERT_WEBHOOK_URL:
        tasks.append(_send_webhook(event))
    if config.ALERT_SLACK_WEBHOOK:
        tasks.append(_send_slack(event))
    if config.ALERT_LINE_TOKEN:
        tasks.append(_send_line(event))

    if not tasks:
        logger.debug("No alert channels configured, skipping")
        return

    import asyncio
    await asyncio.gather(*tasks, return_exceptions=True)


async def _send_webhook(event: dict) -> None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                config.ALERT_WEBHOOK_URL,
                json=event,
                headers={"Content-Type": "application/json"},
            )
            logger.info("Webhook sent: %d", resp.status_code)
    except Exception as e:
        logger.error("Webhook failed: %s", e)


async def _send_slack(event: dict) -> None:
    cam = event.get("cam_id", "unknown")
    scenario = event.get("scenario", "unknown")
    confidence = event.get("confidence", 0)
    reason = event.get("reason", "")

    text = (
        f":rotating_light: *HiO Alert*\n"
        f"*Camera:* {cam}\n"
        f"*Scenario:* {scenario}\n"
        f"*Confidence:* {confidence:.1%}\n"
        f"*Reason:* {reason}"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                config.ALERT_SLACK_WEBHOOK,
                json={"text": text},
            )
            logger.info("Slack sent: %d", resp.status_code)
    except Exception as e:
        logger.error("Slack failed: %s", e)


async def _send_line(event: dict) -> None:
    cam = event.get("cam_id", "unknown")
    scenario = event.get("scenario", "unknown")
    confidence = event.get("confidence", 0)

    message = f"[HiO Alert] {cam} — {scenario} (conf: {confidence:.1%})"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {config.ALERT_LINE_TOKEN}"},
                data={"message": message},
            )
            logger.info("LINE sent: %d", resp.status_code)
    except Exception as e:
        logger.error("LINE failed: %s", e)
