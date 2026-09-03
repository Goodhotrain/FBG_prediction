"""Evaluate a trained PSDCFN checkpoint and export predictions."""
from __future__ import annotations
from pathlib import Path
import csv
import torch
from common.engine import load_checkpoint, resolve_device, run_epoch, seed_everything
from data.dataset import build_dataloaders
from models.model import FBGPredictor
from opt import parse_opts

def evaluate(args):
    if not args.checkpoint:
        raise ValueError("evaluation requires --checkpoint PATH")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    _, loader, _ = build_dataloaders(args)
    model = FBGPredictor.from_args(args).to(device)
    load_checkpoint(args.checkpoint, model, device)
    metrics = run_epoch(model, loader, device)
    print(f"MAE={metrics.mae:.4f} RMSE={metrics.rmse:.4f} MSE={metrics.mse:.4f} R2={metrics.r2:.4f}")
    output_path = Path(args.checkpoint).parent.parent / "predictions.csv"
    model.eval()
    rows = []
    with torch.inference_mode():
        for sample_id, static, nutrition, mask, target in loader:
            details = model(sample_id.to(device), static.to(device), nutrition.to(device), mask.to(device), return_details=True)
            for sid, truth, prediction, gate in zip(sample_id, target, details.prediction.cpu(), details.history_gate.cpu()):
                rows.append((sid.item(), truth.item(), prediction.item(), gate.item()))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("sample_id", "target", "prediction", "history_gate")); writer.writerows(rows)
    print(f"predictions={output_path}")
    return metrics

if __name__ == "__main__":
    evaluate(parse_opts())
