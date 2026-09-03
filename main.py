"""Training entry point: python main.py --help."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
from common.engine import append_history, load_checkpoint, resolve_device, run_epoch, save_checkpoint, seed_everything
from common.optimizer import get_optimizer, get_scheduler
from data.dataset import build_dataloaders
from models.model import FBGPredictor
from opt import parse_opts

def train(args) -> Path:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.result_path) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    train_loader, valid_loader, preprocessor = build_dataloaders(args)
    model = FBGPredictor.from_args(args).to(device)
    optimizer, scheduler = get_optimizer(args, model), None
    scheduler = get_scheduler(args, optimizer)
    start_epoch, best_mae, stale = 0, float("inf"), 0
    if args.resume:
        if not args.checkpoint:
            raise ValueError("--resume requires --checkpoint")
        state = load_checkpoint(args.checkpoint, model, device, optimizer, scheduler)
        start_epoch, best_mae = state["epoch"] + 1, state["best_mae"]
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} run={run_dir}")
    try:
        for epoch in range(start_epoch, args.epochs):
            train_metrics = run_epoch(model, train_loader, device, optimizer, args.grad_clip)
            valid_metrics = run_epoch(model, valid_loader, device)
            lr = optimizer.param_groups[0]["lr"]
            append_history(run_dir / "metrics.jsonl", epoch, train_metrics, valid_metrics, lr)
            for split, values in (("train", train_metrics), ("valid", valid_metrics)):
                for name, value in vars(values).items(): writer.add_scalar(f"{split}/{name}", value, epoch)
            print(f"epoch={epoch + 1:03d} train_loss={train_metrics.loss:.4f} valid_mae={valid_metrics.mae:.4f} valid_rmse={valid_metrics.rmse:.4f} r2={valid_metrics.r2:.4f}")
            improved = valid_metrics.mae < best_mae
            if improved:
                best_mae, stale = valid_metrics.mae, 0
                save_checkpoint(run_dir / "checkpoints/best.pt", model, optimizer, scheduler, epoch, best_mae, args, preprocessor)
            else:
                stale += 1
            save_checkpoint(run_dir / "checkpoints/last.pt", model, optimizer, scheduler, epoch, best_mae, args, preprocessor)
            if scheduler:
                scheduler.step(valid_metrics.mae) if args.scheduler == "plateau" else scheduler.step()
            if stale >= args.patience:
                print(f"early stopping after {args.patience} epochs without MAE improvement")
                break
    finally:
        writer.close()
    print(f"best validation MAE={best_mae:.4f}; checkpoint={run_dir / 'checkpoints/best.pt'}")
    return run_dir

if __name__ == "__main__":
    options = parse_opts()
    if options.test_only:
        from valid import evaluate
        evaluate(options)
    else:
        train(options)
