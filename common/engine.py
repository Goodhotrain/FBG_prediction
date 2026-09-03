"""Reusable training, evaluation, metrics, and checkpoint utilities."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch import nn

@dataclass
class RegressionMetrics:
    loss: float
    mae: float
    mse: float
    rmse: float
    r2: float
    counterfactual_loss: float = 0.0

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def _metrics(predictions: list[torch.Tensor], targets: list[torch.Tensor], loss: float, cf: float) -> RegressionMetrics:
    pred, true = torch.cat(predictions).double(), torch.cat(targets).double()
    residual = pred - true
    mse = residual.square().mean().item()
    total = (true - true.mean()).square().sum().item()
    r2 = 1.0 - residual.square().sum().item() / total if total > 0 else 0.0
    return RegressionMetrics(loss, residual.abs().mean().item(), mse, mse ** 0.5, r2, cf)

def run_epoch(model: nn.Module, loader, device: torch.device, optimizer=None, grad_clip: float = 1.0) -> RegressionMetrics:
    training = optimizer is not None
    model.train(training)
    predictions, targets, total_loss, total_cf, samples = [], [], 0.0, 0.0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for sample_id, static, nutrition, mask, target in loader:
            sample_id, static, nutrition, mask, target = [x.to(device, non_blocking=True) for x in (sample_id, static, nutrition, mask, target)]
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(sample_id, static, nutrition, mask, target, return_details=True)
            if training:
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch = target.numel()
            total_loss += output.loss.detach().item() * batch
            total_cf += output.counterfactual_loss.detach().item() * batch
            samples += batch
            predictions.append(output.prediction.detach().cpu())
            targets.append(target.detach().cpu())
    return _metrics(predictions, targets, total_loss / samples, total_cf / samples)

def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_mae: float, args, preprocessor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "best_mae": best_mae, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
                "config": vars(args), "preprocessor": preprocessor.state_dict()}, path)

def load_checkpoint(path: str | Path, model, device, optimizer=None, scheduler=None) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

def append_history(path: Path, epoch: int, train: RegressionMetrics, valid: RegressionMetrics, learning_rate: float) -> None:
    record = {"epoch": epoch, "learning_rate": learning_rate, "train": asdict(train), "valid": asdict(valid)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
