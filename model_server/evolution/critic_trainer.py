"""
Critic Trainer — LightGBM-based false-alarm predictor.

Trains a binary classifier on shadow feedback data to predict whether
a Tier-1 detection is a true positive or false positive.

Used by:
- EvidenceRouter: to bias routing decisions (skip / escalate)
- ShadowAgent._on_batch_ready: triggered when batch threshold is met
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CriticTrainer:
    """
    Trains and manages a LightGBM critic model for false-alarm prediction.

    Features extracted from feedback records:
    - tier1_confidence
    - matched_keyword_count
    - agreement (label)
    """

    FEATURE_NAMES = [
        'tier1_confidence',
        'keyword_count',
        'object_hint_count',
    ]

    def __init__(
        self,
        model_dir: str = "data/critic_models",
        min_samples: int = 30,
    ):
        self.model_dir = model_dir
        self.min_samples = min_samples
        self.model = None
        self._training_count = 0

        os.makedirs(self.model_dir, exist_ok=True)

        # Try to load existing model
        self._load_latest_model()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, batch: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Train or incrementally update the critic model.

        Args:
            batch: list of feedback records from ShadowAgent.
                   Each record must have 'tier1_confidence', 'tier1_keywords',
                   and 'agreement' fields.

        Returns:
            True if training succeeded.
        """
        if batch is None:
            batch = []

        # Filter valid records (only those with a definitive agreement label)
        valid = [r for r in batch if r.get('agreement') is not None]

        if len(valid) < self.min_samples:
            logger.info(
                f"[CriticTrainer] Not enough samples ({len(valid)}/{self.min_samples}). "
                f"Skipping training."
            )
            return False

        try:
            import lightgbm as lgb
            import numpy as np

            X, y = self._prepare_features(valid)

            train_data = lgb.Dataset(X, label=y, feature_name=self.FEATURE_NAMES)

            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 15,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1,
            }

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(period=0)],  # suppress logging
            )

            # Save model
            self._training_count += 1
            model_path = os.path.join(
                self.model_dir,
                f"critic_v{self._training_count}.txt"
            )
            self.model.save_model(model_path)
            logger.info(
                f"[CriticTrainer] Trained on {len(valid)} samples → {model_path}"
            )
            return True

        except ImportError:
            logger.error("[CriticTrainer] lightgbm not installed. pip install lightgbm")
            return False
        except Exception as e:
            logger.error(f"[CriticTrainer] Training failed: {e}")
            return False

    def predict(self, tier1_result: Dict[str, Any]) -> float:
        """
        Predict the probability that a Tier-1 detection is a true positive.

        Returns:
            float in [0, 1], or -1.0 if model is not available.
        """
        if self.model is None:
            return -1.0

        try:
            import numpy as np

            features = self._extract_features(tier1_result)
            x = np.array([features])
            prob = self.model.predict(x)[0]
            return float(prob)
        except Exception as e:
            logger.error(f"[CriticTrainer] Predict failed: {e}")
            return -1.0

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _prepare_features(self, records: List[Dict[str, Any]]):
        import numpy as np

        X = []
        y = []
        for r in records:
            features = [
                float(r.get('tier1_confidence', 0.0)),
                len(r.get('tier1_keywords', [])),
                len(r.get('object_hints', [])),
            ]
            X.append(features)
            y.append(1.0 if r.get('agreement', False) else 0.0)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _extract_features(self, tier1_result: Dict[str, Any]) -> List[float]:
        return [
            float(tier1_result.get('confidence', 0.0)),
            len(tier1_result.get('matched_keywords', [])),
            len(tier1_result.get('object_hints', [])),
        ]

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _load_latest_model(self) -> None:
        """Load the most recent model file from model_dir."""
        try:
            import lightgbm as lgb

            model_files = sorted(Path(self.model_dir).glob("critic_v*.txt"))
            if model_files:
                latest = str(model_files[-1])
                self.model = lgb.Booster(model_file=latest)
                self._training_count = len(model_files)
                logger.info(f"[CriticTrainer] Loaded existing model: {latest}")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[CriticTrainer] Could not load model: {e}")
