"""AWS EC2 spot interruption watcher for HIO v3.

Polls the instance metadata service for ``spot/instance-action`` and signals
``SIGTERM`` to configured PIDs/systemd units so postprocess queue drains and
DB flush can run before the 2-minute termination window expires.

Usage::

    python tools/spot_interruption_watcher.py \
        --units vlm-model vlm-db vlm-frontend \
        --poll-sec 5 \
        --grace-sec 90

Run under systemd (see deploy/vlm-spot-watcher.service).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Iterable, Sequence

try:
    import requests  # noqa: F401
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
IMDS_ACTION_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
IMDS_TOKEN_TTL = "60"

logger = logging.getLogger("spot-watcher")


def _get_token(timeout_sec: float = 1.0) -> str:
    import urllib.request

    req = urllib.request.Request(
        IMDS_TOKEN_URL,
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": IMDS_TOKEN_TTL},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read().decode("utf-8", errors="ignore").strip()


def _check_action(token: str, timeout_sec: float = 1.0) -> str | None:
    """Return the action JSON string when an interruption is scheduled, else None."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        IMDS_ACTION_URL,
        headers={"X-aws-ec2-metadata-token": token},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="ignore").strip()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("IMDS action check transient error: %s", exc)
    return None


def _trigger_flush(flush_url: str, timeout_sec: float = 30.0) -> None:
    """POST to model_server /api/vlm/flush-now/ so today's events drain before stop."""
    if not flush_url:
        return
    import urllib.error
    import urllib.request

    req = urllib.request.Request(flush_url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            logger.info("Flush-now response %s: %s", resp.status, body[:200])
    except urllib.error.URLError as exc:
        logger.warning("Flush-now call failed (%s): %s", flush_url, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Flush-now unexpected error: %s", exc)


def _stop_units(units: Sequence[str]) -> None:
    for unit in units:
        try:
            subprocess.run(["systemctl", "stop", unit], check=False, timeout=60)
            logger.info("Stopped unit %s", unit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to stop unit %s: %s", unit, exc)


def _signal_pids(pids: Iterable[int]) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
            logger.info("SIGTERM -> pid %s", pid)
        except ProcessLookupError:
            logger.info("pid %s already exited", pid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to signal pid %s: %s", pid, exc)


def watch(
    *,
    units: Sequence[str],
    pids: Sequence[int],
    poll_sec: float,
    grace_sec: float,
    flush_url: str = "",
) -> int:
    logger.info(
        "Spot watcher started (units=%s pids=%s poll=%ss grace=%ss flush_url=%s)",
        list(units) or "-",
        list(pids) or "-",
        poll_sec,
        grace_sec,
        flush_url or "-",
    )
    token = ""
    last_token_refresh = 0.0
    while True:
        try:
            now = time.time()
            if not token or now - last_token_refresh > 30:
                token = _get_token()
                last_token_refresh = now
            action = _check_action(token) if token else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("IMDS polling transient error: %s", exc)
            action = None
        if action:
            logger.warning("Spot interruption scheduled: %s", action[:200])
            # Drain today's events BEFORE stopping services so DB has the latest.
            if flush_url:
                _trigger_flush(flush_url)
            if units:
                _stop_units(units)
            if pids:
                _signal_pids(pids)
            # Give graceful shutdown time, then exit so systemd can re-run us.
            time.sleep(max(1.0, float(grace_sec)))
            return 0
        time.sleep(max(1.0, float(poll_sec)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", nargs="*", default=[], help="systemd units to stop on interruption")
    parser.add_argument("--pid", nargs="*", type=int, default=[], help="PIDs to SIGTERM on interruption")
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--grace-sec", type=float, default=90.0)
    parser.add_argument(
        "--flush-url",
        type=str,
        default="http://127.0.0.1:8000/api/vlm/flush-now/",
        help="POST here before stopping services to drain today's events. "
        "Set empty string to skip.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    return watch(
        units=args.units,
        pids=args.pid,
        poll_sec=args.poll_sec,
        grace_sec=args.grace_sec,
        flush_url=args.flush_url,
    )


if __name__ == "__main__":
    sys.exit(main())
