"""Command-line configuration for PSDCFN experiments."""
from __future__ import annotations
import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate PSDCFN for fasting blood glucose regression", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--label-path", default="data/ID-label-enhanced-92.csv")
    data.add_argument("--static-path", default="data/static.csv")
    data.add_argument("--nutrition-path", default="data/nutrition.csv")
    data.add_argument("--train-split", default="train")
    data.add_argument("--valid-split", default="test")
    data.add_argument("--sequence-length", type=int, default=16)
    model = parser.add_argument_group("model")
    model.add_argument("--static-dim", type=int, default=32)
    model.add_argument("--nutrition-dim", type=int, default=78, help="Nutrition columns excluding IDs and historical FBG")
    model.add_argument("--hidden-dim", type=int, default=64)
    model.add_argument("--num-heads", type=int, default=4)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.15)
    model.add_argument("--cf-lambda", type=float, default=0.05)
    model.add_argument("--perturb-scale", type=float, default=0.05)
    train = parser.add_argument_group("optimization")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    train.add_argument("--scheduler", choices=("cosine", "plateau", "none"), default="cosine")
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    output = parser.add_argument_group("output")
    output.add_argument("--result-path", default="results")
    output.add_argument("--run-name", default=None)
    output.add_argument("--checkpoint", default=None)
    output.add_argument("--resume", action="store_true")
    output.add_argument("--test-only", action="store_true")
    return parser

def parse_opts(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for name in ("label_path", "static_path", "nutrition_path", "result_path"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser()))
    return args
