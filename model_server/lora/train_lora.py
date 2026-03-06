"""
Florence-2 LoRA Fine-tuning Script.

Trains a LoRA adapter on collected CCTV data to improve Florence-2's
scene captioning for Cash/Fire/Violence detection scenarios.

Usage:
    # From project root:
    python -m model_server.lora.train_lora

    # With custom options:
    python -m model_server.lora.train_lora --epochs 5 --lr 5e-5 --batch-size 4

Requirements:
    pip install peft datasets accelerate
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Florence-2 with LoRA on CCTV training data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/lora_training",
        help="Path to training data directory (default: data/lora_training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lora_output",
        help="Path to save LoRA adapter weights (default: data/lora_output)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Florence-2-large",
        help="Base Florence-2 model name or path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8, higher=more capacity but more VRAM)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA scaling alpha (default: 16)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./model_cache",
        help="HuggingFace model cache directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length for captions (default: 512)",
    )
    return parser.parse_args()


def _check_data(data_dir: str) -> dict:
    """Verify training data exists and inspect it."""
    data_path = Path(data_dir)
    annotations = data_path / "annotations.jsonl"
    images_dir = data_path / "images"

    if not annotations.exists():
        return {"ready": False, "error": f"Annotations not found: {annotations}"}

    if not images_dir.exists():
        return {"ready": False, "error": f"Images dir not found: {images_dir}"}

    sample_count = 0
    labels: dict[str, int] = {}
    scenarios: dict[str, int] = {}

    with open(annotations, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                label = record.get("label", "unknown")
                scenario = record.get("scenario", "unknown")
                labels[label] = labels.get(label, 0) + 1
                scenarios[scenario] = scenarios.get(scenario, 0) + 1
                sample_count += 1
            except json.JSONDecodeError:
                continue

    if sample_count < 10:
        return {
            "ready": False,
            "error": f"Too few samples ({sample_count}). Need at least 10.",
            "count": sample_count,
        }

    return {
        "ready": True,
        "count": sample_count,
        "by_label": labels,
        "by_scenario": scenarios,
    }


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    print("=" * 60)
    print("Florence-2 LoRA Fine-tuning")
    print("=" * 60)

    # ─── Check data ───────────────────────────────────────────
    print("\n[1/6] Checking training data...")
    data_info = _check_data(args.data_dir)
    if not data_info["ready"]:
        print(f"ERROR: {data_info['error']}")
        print(
            "\nTo collect training data, run the system with "
            "LORA_DATA_COLLECTION=true and let it process RTSP streams."
        )
        sys.exit(1)

    print(f"  Found {data_info['count']} samples")
    print(f"  Labels: {data_info['by_label']}")
    print(f"  Scenarios: {data_info['by_scenario']}")

    # ─── Import heavy dependencies ────────────────────────────
    print("\n[2/6] Loading dependencies...")
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install peft datasets accelerate")
        sys.exit(1)

    # ─── Setup device ─────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"  Device: {device} (dtype: {dtype})")

    # ─── Set seed ─────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ─── Load model + processor ───────────────────────────────
    print(f"\n[3/6] Loading Florence-2 model: {args.model}...")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
    )

    # ─── Apply LoRA ───────────────────────────────────────────
    print(f"\n[4/6] Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")

    # Find target modules in Florence-2
    # Florence-2 uses an encoder-decoder architecture
    # We target the attention projection layers
    target_modules = []
    for name, _ in model.named_modules():
        if any(key in name for key in ["q_proj", "v_proj", "k_proj", "out_proj"]):
            # Get just the module path component
            target_modules.append(name.split(".")[-1])

    # Deduplicate
    target_modules = list(set(target_modules))
    if not target_modules:
        target_modules = ["q_proj", "v_proj"]

    print(f"  Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # ─── Load dataset ─────────────────────────────────────────
    print(f"\n[5/6] Loading dataset from {args.data_dir}...")

    # Use our custom dataset
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from model_server.lora.dataset import FlorenceLoRADataset, FlorenceTrainCollate

    full_dataset = FlorenceLoRADataset(data_dir=args.data_dir)
    train_dataset, val_dataset = full_dataset.split(val_ratio=args.val_ratio, seed=args.seed)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    if len(train_dataset) == 0:
        print("ERROR: No training samples available.")
        sys.exit(1)

    collate_fn = FlorenceTrainCollate(processor, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ─── Training loop ────────────────────────────────────────
    print(f"\n[6/6] Training for {args.epochs} epochs...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Simple linear warmup + decay scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = min(100, total_steps // 10)

    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

            total_loss += outputs.loss.item()

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Step {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f}"
                )

        avg_train_loss = total_loss / max(len(train_loader), 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"\n  Epoch {epoch+1}/{args.epochs} complete | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # ── Save best model ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(str(output_dir))
            processor.save_pretrained(str(output_dir))
            print(f"  ✓ Saved best model to {output_dir}")

    # ─── Final save ───────────────────────────────────────────
    # Always save the final version too
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    # Save training metadata
    metadata = {
        "base_model": args.model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_loss": best_val_loss,
        "target_modules": target_modules,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Adapter saved: {output_dir}")
    print(f"  To use: set LORA_ADAPTER_PATH={output_dir} in .env")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    train(args)
