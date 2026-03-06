"""
Florence-2 LoRA Dataset — PyTorch Dataset for fine-tuning.

Reads annotations.jsonl and provides (image, prompt, target_caption) tuples
suitable for Florence-2 captioning fine-tuning.

Usage:
    from model_server.lora.dataset import FlorenceLoRADataset

    ds = FlorenceLoRADataset("data/lora_training")
    train_ds, val_ds = ds.split(val_ratio=0.1)
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FlorenceLoRADataset:
    """
    PyTorch-compatible dataset for Florence-2 LoRA training.

    Each item returns a dict with:
        - image_path: absolute path to JPEG file
        - prefix: Florence-2 task token (e.g., "<MORE_DETAILED_CAPTION>")
        - suffix: target caption text
        - scenario: scenario name
        - label: "detected" / "normal" / "true_positive" / "false_positive"
    """

    def __init__(
        self,
        data_dir: str = "data/lora_training",
        filter_labels: Optional[List[str]] = None,
        filter_scenarios: Optional[List[str]] = None,
    ):
        """
        Args:
            data_dir: Path to training data directory.
            filter_labels: Only include samples with these labels.
            filter_scenarios: Only include samples from these scenarios.
        """
        self.data_dir = Path(data_dir)
        self.annotations_path = self.data_dir / "annotations.jsonl"
        self.records: List[Dict[str, Any]] = []

        self._load(filter_labels, filter_scenarios)

    def _load(
        self,
        filter_labels: Optional[List[str]] = None,
        filter_scenarios: Optional[List[str]] = None,
    ) -> None:
        """Load and filter annotations from JSONL file."""
        if not self.annotations_path.exists():
            logger.warning(
                f"[FlorenceLoRADataset] Annotations file not found: "
                f"{self.annotations_path}"
            )
            return

        with open(self.annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by label
                if filter_labels and record.get("label") not in filter_labels:
                    continue

                # Filter by scenario
                if filter_scenarios and record.get("scenario") not in filter_scenarios:
                    continue

                # Verify image exists
                image_path = self.data_dir / record.get("image", "")
                if not image_path.exists():
                    continue

                # Verify caption is non-empty
                if not record.get("suffix", "").strip():
                    continue

                self.records.append(record)

        logger.info(
            f"[FlorenceLoRADataset] Loaded {len(self.records)} samples "
            f"from {self.annotations_path}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image_path = str((self.data_dir / record["image"]).resolve())

        return {
            "image_path": image_path,
            "prefix": record.get("prefix", "<MORE_DETAILED_CAPTION>"),
            "suffix": record.get("suffix", ""),
            "scenario": record.get("scenario", "unknown"),
            "label": record.get("label", "unknown"),
        }

    def split(
        self,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple["FlorenceLoRADataset", "FlorenceLoRADataset"]:
        """
        Split into train and validation sets.

        Args:
            val_ratio: Fraction for validation (0.1 = 10%).
            seed: Random seed for reproducibility.

        Returns:
            (train_dataset, val_dataset)
        """
        rng = random.Random(seed)
        indices = list(range(len(self.records)))
        rng.shuffle(indices)

        val_size = max(1, int(len(indices) * val_ratio))
        val_indices = set(indices[:val_size])

        train_ds = FlorenceLoRADataset.__new__(FlorenceLoRADataset)
        train_ds.data_dir = self.data_dir
        train_ds.annotations_path = self.annotations_path
        train_ds.records = [
            r for i, r in enumerate(self.records) if i not in val_indices
        ]

        val_ds = FlorenceLoRADataset.__new__(FlorenceLoRADataset)
        val_ds.data_dir = self.data_dir
        val_ds.annotations_path = self.annotations_path
        val_ds.records = [
            r for i, r in enumerate(self.records) if i in val_indices
        ]

        logger.info(
            f"[FlorenceLoRADataset] Split: train={len(train_ds)}, val={len(val_ds)}"
        )

        return train_ds, val_ds

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the dataset contents."""
        label_counts: dict[str, int] = {}
        scenario_counts: dict[str, int] = {}

        for r in self.records:
            label = r.get("label", "unknown")
            scenario = r.get("scenario", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        return {
            "total": len(self.records),
            "by_label": label_counts,
            "by_scenario": scenario_counts,
        }


class FlorenceTrainCollate:
    """
    Collate function for Florence-2 training DataLoader.

    Converts raw dataset items into processor-ready batches.
    Requires the Florence-2 processor for tokenization.
    """

    def __init__(self, processor, max_length: int = 512):
        """
        Args:
            processor: Florence-2 AutoProcessor instance.
            max_length: Maximum token length for captions.
        """
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from PIL import Image

        images = []
        prefixes = []
        suffixes = []

        for item in batch:
            img = Image.open(item["image_path"]).convert("RGB")
            images.append(img)
            prefixes.append(item["prefix"])
            suffixes.append(item["suffix"])

        # Process inputs (image + task prefix)
        inputs = self.processor(
            text=prefixes,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Process target captions
        labels = self.processor.tokenizer(
            suffixes,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )

        inputs["labels"] = labels["input_ids"]

        return inputs
