import torch.optim as optim
from .build import SOLVER_REGISTRY


@SOLVER_REGISTRY.register()
def sgd(cfg, model):
    return optim.SGD(model.parameters(), lr=cfg.SOLVER.LR_INIT, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
