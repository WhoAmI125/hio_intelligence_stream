from django.db import models


class EpisodeReview(models.Model):
    POLICY_CHOICES = [
        ("CASH_TRANSACTION", "Cash Transaction"),
        ("THREAT", "Threat"),
        ("FIRE", "Fire"),
        ("THEFT", "Theft"),
        ("NONE", "None"),
    ]

    FOCUS_ZONE_CHOICES = [
        ("cashier", "Cashier"),
        ("drawer", "Drawer"),
        ("global", "Global"),
    ]

    REVIEW_STATUS_CHOICES = [
        ("queued", "Queued"),
        ("in_review", "In Review"),
        ("done", "Done"),
    ]

    episode_id = models.CharField(max_length=128, db_index=True)
    event_id = models.CharField(max_length=128, blank=True, default="", db_index=True)
    camera_id = models.CharField(max_length=64)
    event_type = models.CharField(max_length=32, blank=True, default="", db_index=True)
    final_policy = models.CharField(max_length=32, choices=POLICY_CHOICES)
    is_valid_event = models.BooleanField(default=False)
    evidence_span_start_sec = models.FloatField(null=True, blank=True)
    evidence_span_end_sec = models.FloatField(null=True, blank=True)
    focus_zone = models.CharField(max_length=16, choices=FOCUS_ZONE_CHOICES, default="global")
    note = models.TextField(blank=True, default="")
    review_status = models.CharField(max_length=16, choices=REVIEW_STATUS_CHOICES, default="done")
    reviewer = models.CharField(max_length=128, blank=True, default="")
    gemini_state = models.CharField(max_length=32, blank=True, default="")
    gemini_validated = models.BooleanField(null=True, blank=True)
    gemini_confidence = models.FloatField(null=True, blank=True)
    gemini_reason = models.TextField(blank=True, default="")
    validation_type = models.CharField(max_length=16, blank=True, default="")
    evidence_packet_id = models.CharField(max_length=128, blank=True, default="", db_index=True)
    input_mode = models.CharField(max_length=16, blank=True, default="")
    prompt_version = models.CharField(max_length=64, blank=True, default="")
    tier1_snapshot = models.JSONField(default=dict, blank=True)
    router_snapshot = models.JSONField(default=dict, blank=True)
    packet_summary = models.JSONField(default=dict, blank=True)
    florence_signals = models.JSONField(default=dict, blank=True)
    feedback_error_type = models.CharField(max_length=32, blank=True, default="")
    feedback_missed_focus = models.JSONField(default=list, blank=True)
    feedback_suggestion = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vlm_episode_reviews"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.episode_id} | {self.final_policy} | {self.review_status}"


class AdhocEvent(models.Model):
    """이벤트 영속성: worker가 재시작해도 기록을 유지."""
    event_id = models.CharField(max_length=128, unique=True, db_index=True)
    camera_id = models.CharField(max_length=64, db_index=True, blank=True, default="")
    rtsp_url = models.CharField(max_length=512, blank=True, default="")
    event_type = models.CharField(max_length=32, blank=True, default="", db_index=True)
    at = models.DateTimeField(db_index=True)
    gemini_validated = models.BooleanField(null=True, blank=True)
    gemini_confidence = models.FloatField(null=True, blank=True)
    gemini_reason = models.TextField(blank=True, default="")
    gemini_state = models.CharField(max_length=32, blank=True, default="")
    validation_clip_url = models.CharField(max_length=512, blank=True, default="")
    clip_url = models.CharField(max_length=512, blank=True, default="")
    human_feedback = models.JSONField(null=True, blank=True)
    event_data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vlm_adhoc_events"
        ordering = ["-at"]

    def __str__(self):
        return f"{self.event_id} | {self.event_type} | {self.at}"


class WorkerLease(models.Model):
    """Cross-process worker dedup: camera_id 단위 DB lease."""

    camera_id = models.CharField(max_length=128, unique=True, db_index=True)
    instance_id = models.UUIDField()
    pid = models.IntegerField(default=0)
    rtsp_url = models.CharField(max_length=512, blank=True, default="")
    acquired_at = models.DateTimeField(auto_now_add=True)
    last_heartbeat = models.DateTimeField(auto_now=True)
    lease_ttl_sec = models.IntegerField(default=60)

    class Meta:
        db_table = "vlm_worker_leases"

    def __str__(self):
        return f"{self.camera_id} | {self.instance_id} | pid={self.pid}"
