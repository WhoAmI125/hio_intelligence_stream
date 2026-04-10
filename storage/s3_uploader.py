"""Optional S3 clip uploader."""

from __future__ import annotations

import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


async def upload_clip(local_path: str) -> str | None:
    """Upload clip to S3 and return the S3 URL. Returns None if S3 is disabled."""
    if not config.USE_S3:
        return None

    try:
        import boto3

        s3 = boto3.client("s3", region_name=config.AWS_REGION)
        key = f"clips/{Path(local_path).name}"
        s3.upload_file(local_path, config.AWS_BUCKET, key)
        url = f"https://{config.AWS_BUCKET}.s3.{config.AWS_REGION}.amazonaws.com/{key}"
        logger.info("Uploaded to S3: %s", url)
        return url
    except Exception as e:
        logger.error("S3 upload failed: %s", e)
        return None
