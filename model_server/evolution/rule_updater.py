"""
Rule Updater — Versioned prompt evolution with Gemini-based refinement.

Replaces the naive "append text" approach with:
1. Version-controlled prompt files (numbered snapshots)
2. Gemini Pro-powered prompt refinement when disagreement is high
3. Hot-reload support for runtime prompt updates

Directory structure:
    agents/prompts/
        cash.md          ← live (current) prompt
        fire.md
        violence.md
    data/rule_versions/
        cash/
            v001_20260227_130000.md
            v002_20260227_150000.md
            changelog.jsonl
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RuleUpdater:
    """
    Manages scenario prompt evolution with versioning and (optional) AI refinement.

    Usage:
        updater = RuleUpdater(prompts_dir="agents/prompts", versions_dir="data/rule_versions")

        # Called by ShadowAgent when disagreement rate is high
        updater.apply_feedback_to_rules("cash", "T1 missed drawer signals...")

        # Get version history
        history = updater.get_version_history("cash")
    """

    def __init__(
        self,
        prompts_dir: str = "agents/prompts",
        versions_dir: str = "data/rule_versions",
        gemini_api_key: Optional[str] = None,
        refine_model: str = "gemini-2.5-pro",
        max_versions: int = 50,
    ):
        self.prompts_dir = Path(prompts_dir)
        self.versions_dir = Path(versions_dir)
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.refine_model = refine_model
        self.max_versions = max_versions

        self.versions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def apply_feedback_to_rules(
        self,
        scenario_name: str,
        feedback_text: str,
        *,
        use_ai_refine: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply feedback to evolve a scenario's prompt.

        1. Snapshot current prompt as a new version
        2. (Optional) Use Gemini Pro to intelligently refine the prompt
        3. Write the updated prompt as the new live version
        4. Log the change

        Args:
            scenario_name: 'cash', 'fire', or 'violence'
            feedback_text: Summary of disagreements / issues
            use_ai_refine: If True, uses Gemini to generate refined prompt

        Returns:
            dict with version info and status
        """
        prompt_path = self.prompts_dir / f"{scenario_name}.md"
        if not prompt_path.exists():
            logger.error(f"[RuleUpdater] Prompt file not found: {prompt_path}")
            return {"status": "error", "message": f"Prompt not found: {prompt_path}"}

        # Read current prompt
        current_prompt = prompt_path.read_text(encoding="utf-8")

        # Snapshot current version
        version_num = self._next_version_number(scenario_name)
        snapshot_path = self._save_version(scenario_name, version_num, current_prompt)

        # Attempt AI-powered refinement
        if use_ai_refine and self.gemini_api_key:
            refined = self._refine_with_gemini(
                scenario_name, current_prompt, feedback_text
            )
            if refined:
                # Write refined prompt as new live version
                prompt_path.write_text(refined, encoding="utf-8")
                method = "gemini_refine"
                logger.info(
                    f"[RuleUpdater] {scenario_name} → v{version_num:03d} "
                    f"(Gemini-refined)"
                )
            else:
                # Fallback: append feedback as a training note
                self._append_feedback_note(prompt_path, feedback_text)
                method = "append_fallback"
                logger.warning(
                    f"[RuleUpdater] {scenario_name} → v{version_num:03d} "
                    f"(Gemini failed, appended feedback)"
                )
        else:
            # No AI: append feedback as structured note
            self._append_feedback_note(prompt_path, feedback_text)
            method = "manual_append"
            logger.info(
                f"[RuleUpdater] {scenario_name} → v{version_num:03d} "
                f"(manual append)"
            )

        # Log change
        self._log_change(scenario_name, version_num, method, feedback_text)

        # Cleanup old versions
        self._cleanup_old_versions(scenario_name)

        return {
            "status": "updated",
            "scenario": scenario_name,
            "version": version_num,
            "method": method,
            "snapshot": str(snapshot_path),
        }

    def get_current_prompt(self, scenario_name: str) -> Optional[str]:
        """Read the current live prompt for a scenario."""
        path = self.prompts_dir / f"{scenario_name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_version_history(self, scenario_name: str) -> List[Dict[str, Any]]:
        """Get the changelog for a scenario."""
        changelog = self.versions_dir / scenario_name / "changelog.jsonl"
        if not changelog.exists():
            return []

        entries = []
        with open(changelog, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def rollback(self, scenario_name: str, version: int) -> bool:
        """Rollback to a previous version."""
        scenario_dir = self.versions_dir / scenario_name
        ts_pattern = f"v{version:03d}_*.md"
        matches = list(scenario_dir.glob(ts_pattern))

        if not matches:
            logger.error(f"[RuleUpdater] Version v{version:03d} not found for {scenario_name}")
            return False

        old_prompt = matches[0].read_text(encoding="utf-8")

        # Snapshot current before rollback
        prompt_path = self.prompts_dir / f"{scenario_name}.md"
        if prompt_path.exists():
            current = prompt_path.read_text(encoding="utf-8")
            rb_version = self._next_version_number(scenario_name)
            self._save_version(scenario_name, rb_version, current)
            self._log_change(scenario_name, rb_version, "pre_rollback", f"Before rollback to v{version:03d}")

        # Write rolled-back prompt
        prompt_path.write_text(old_prompt, encoding="utf-8")

        new_version = self._next_version_number(scenario_name)
        self._log_change(scenario_name, new_version, "rollback", f"Rolled back to v{version:03d}")

        logger.info(f"[RuleUpdater] {scenario_name} rolled back to v{version:03d}")
        return True

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def _next_version_number(self, scenario_name: str) -> int:
        scenario_dir = self.versions_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        existing = list(scenario_dir.glob("v*_*.md"))
        if not existing:
            return 1
        nums = []
        for p in existing:
            try:
                nums.append(int(p.name.split("_")[0][1:]))
            except (ValueError, IndexError):
                continue
        return max(nums, default=0) + 1

    def _save_version(self, scenario_name: str, version: int, content: str) -> Path:
        scenario_dir = self.versions_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = scenario_dir / f"v{version:03d}_{ts}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def _log_change(
        self,
        scenario_name: str,
        version: int,
        method: str,
        feedback: str,
    ) -> None:
        changelog = self.versions_dir / scenario_name / "changelog.jsonl"
        entry = {
            "version": version,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "feedback_summary": feedback[:500],
        }
        with open(changelog, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _cleanup_old_versions(self, scenario_name: str) -> None:
        scenario_dir = self.versions_dir / scenario_name
        versions = sorted(scenario_dir.glob("v*_*.md"))
        if len(versions) > self.max_versions:
            for old in versions[:len(versions) - self.max_versions]:
                old.unlink()

    # ------------------------------------------------------------------
    # AI Refinement (Gemini Pro)
    # ------------------------------------------------------------------

    def _refine_with_gemini(
        self,
        scenario_name: str,
        current_prompt: str,
        feedback: str,
    ) -> Optional[str]:
        """
        Use Gemini Pro to intelligently refine a scenario prompt based on feedback.

        Returns:
            Refined prompt text, or None on failure.
        """
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)

            meta_prompt = f"""You are an AI prompt engineer for a CCTV detection system.

Current scenario: {scenario_name}

The current detection prompt is shown below. Based on the feedback provided,
refine and improve the prompt to reduce false positives/negatives.

Rules:
1. Keep the same overall structure and format
2. Add or modify keywords/rules based on the feedback
3. Do NOT remove existing critical rules unless feedback explicitly contradicts them
4. Output ONLY the refined prompt text (no explanations)

--- CURRENT PROMPT ---
{current_prompt[:3000]}
--- END CURRENT PROMPT ---

--- FEEDBACK ---
{feedback[:1000]}
--- END FEEDBACK ---

Output only the refined prompt (no markdown fences, no explanations):"""

            response = client.models.generate_content(
                model=self.refine_model,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=meta_prompt)]
                )],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=4000,
                ),
            )

            if response.text:
                refined = response.text.strip()
                # Sanity check: refined prompt should be substantial
                if len(refined) > 100:
                    return refined
                logger.warning("[RuleUpdater] Gemini returned too-short refinement, skipping.")
                return None

        except ImportError:
            logger.warning("[RuleUpdater] google-genai not installed.")
        except Exception as e:
            logger.error(f"[RuleUpdater] Gemini refinement failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Feedback append (fallback when AI is unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _append_feedback_note(prompt_path: Path, feedback: str) -> None:
        """Append a structured feedback note to the prompt file."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        note = f"""

---
## Learned Adjustments ({ts})
{feedback}
---
"""
        with open(prompt_path, 'a', encoding='utf-8') as f:
            f.write(note)
