import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99)
    else:
        raise ValueError("Unknown optimizer")


def get_scheduler(args, optimizer):
    if args.scheduler == "step":
        return lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)
    elif args.scheduler == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        return None