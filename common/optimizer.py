"""Optimizer and learning-rate scheduler factories."""
import torch

def get_optimizer(args, model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer: {args.optimizer}")

def get_scheduler(args, optimizer):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.01)
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=max(args.patience // 3, 1), factor=0.5)
    return None
