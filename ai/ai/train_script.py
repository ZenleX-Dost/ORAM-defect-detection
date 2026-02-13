"""
ORAM – Training CLI Script
============================
Standalone entry point for training all agents.

Usage
-----
    python -m ai.train_script --mode synthetic --epochs 20
    python -m ai.train_script --mode kaggle --dataset railway_track_fault --epochs 50
    python -m ai.train_script --mode local --data-dir ./data/my_dataset --epochs 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from loguru import logger

# ── ensure parent is on sys.path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.training import (
    AgentConfig,
    AgentType,
    AgentTrainer,
    MultiAgentOrchestrator,
    get_orchestrator,
    MODEL_REGISTRY,
)
from ai.datasets import get_dataset_manager


# ---------------------------------------------------------------------------
# Default agent configurations
# ---------------------------------------------------------------------------

DEFAULT_AGENTS = [
    AgentConfig(
        agent_type=AgentType.CRACK,
        model_name="efficientnet_b0",
        dataset_key="surface_crack_kaggle",
        num_classes=2,
        class_names=["negative", "positive"],
        epochs=30,
        batch_size=32,
        learning_rate=1e-3,
    ),
    AgentConfig(
        agent_type=AgentType.CORROSION,
        model_name="mobilenet_v3",
        dataset_key="corrosion_mendeley",
        num_classes=2,
        class_names=["no_corrosion", "corrosion"],
        epochs=30,
        batch_size=32,
        learning_rate=1e-3,
    ),
    AgentConfig(
        agent_type=AgentType.LEAK,
        model_name="resnet18",
        dataset_key="synthetic",
        num_classes=2,
        class_names=["normal", "leak"],
        epochs=20,
        batch_size=32,
        learning_rate=1e-3,
    ),
    AgentConfig(
        agent_type=AgentType.GENERAL,
        model_name="efficientnet_b3",
        dataset_key="railway_track_fault",
        num_classes=2,
        class_names=["non-defective", "defective"],
        epochs=30,
        batch_size=16,
        learning_rate=5e-4,
        image_size=300,
    ),
]


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------


async def run_training(args: argparse.Namespace):
    """Main training coroutine."""
    dm = get_dataset_manager()
    orch = get_orchestrator()

    logger.info("=" * 60)
    logger.info("  ORAM – Multi-Agent Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Mode        : {args.mode}")
    logger.info(f"Epochs      : {args.epochs}")
    logger.info(f"Batch size  : {args.batch_size}")
    logger.info(f"Device      : auto (CUDA if available)")
    logger.info("=" * 60)

    # ── 1. Prepare data ──────────────────────────────────────
    if args.mode == "synthetic":
        logger.info("Generating synthetic dataset for quick prototyping ...")
        result = await dm.create_synthetic("synthetic", count=1000)
        data_root = Path(result["path"])
        logger.info(f"Synthetic dataset created: {result['samples']} images at {data_root}")

    elif args.mode == "kaggle":
        key = args.dataset or "railway_track_fault"
        logger.info(f"Downloading Kaggle dataset: {key}")
        result = await dm.download_kaggle(key)
        if result["status"] != "ok":
            logger.error(f"Download failed: {result['message']}")
            return
        data_root = Path(result["path"])
        logger.info(f"Dataset ready: {result['samples']} images at {data_root}")

    elif args.mode == "local":
        data_root = Path(args.data_dir)
        if not data_root.exists():
            logger.error(f"Data directory does not exist: {data_root}")
            return
        logger.info(f"Using local data: {data_root}")

    else:
        logger.error(f"Unknown mode: {args.mode}")
        return

    # ── 2. Configure agents ──────────────────────────────────
    agents = []
    if args.agent:
        # Train single agent
        agent_type = AgentType(args.agent)
        cfg = AgentConfig(
            agent_type=agent_type,
            model_name=args.model or "efficientnet_b0",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            image_size=args.image_size,
        )
        agents = [cfg]
    else:
        # Train all default agents
        agents = DEFAULT_AGENTS
        for a in agents:
            a.epochs = args.epochs
            a.batch_size = args.batch_size

    logger.info(f"Training {len(agents)} agents: {[a.agent_type.value for a in agents]}")

    # ── 3. Run training ──────────────────────────────────────
    start_time = time.time()

    job = orch.create_job(agents)
    logger.info(f"Training job created: {job.job_id}")

    job = await orch.run_job(job.job_id, data_root)

    elapsed = time.time() - start_time

    # ── 4. Report results ────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Status      : {job.status.value}")
    logger.info(f"Total time  : {elapsed:.1f}s")
    logger.info("")

    for agent_name, metrics in job.metrics.items():
        logger.info(f"  [{agent_name}]")
        if "error" in metrics:
            logger.error(f"    Error: {metrics['error']}")
        else:
            logger.info(f"    Model         : {metrics['model']}")
            logger.info(f"    Best val acc   : {metrics['best_val_acc']}%")
            logger.info(f"    Final epoch    : {metrics['final_epoch']}")
            logger.info(f"    Latency (CPU)  : {metrics['latency_ms']}ms")
            logger.info(f"    Model path     : {metrics['model_path']}")
        logger.info("")

    if job.error:
        logger.error(f"Job error: {job.error}")

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ORAM Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ai.train_script --mode synthetic --epochs 20
  python -m ai.train_script --mode kaggle --dataset railway_track_fault --epochs 50
  python -m ai.train_script --mode local --data-dir ./data/my_images --agent crack
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["synthetic", "kaggle", "local"],
        default="synthetic",
        help="Data source mode (default: synthetic)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Kaggle dataset key from catalogue (for --mode kaggle)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Local data directory (for --mode local)",
    )
    parser.add_argument(
        "--agent", type=str, default=None,
        choices=["crack", "corrosion", "leak", "general"],
        help="Train a single agent (default: train all)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture override",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")

    args = parser.parse_args()
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
