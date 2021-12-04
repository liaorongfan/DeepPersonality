import torch.optim as optim
from .build import SOLVER_REGISTRY


@SOLVER_REGISTRY.register()
def multi_step_scale(cfg, optimizer):
    return optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.SOLVER.FACTOR, milestones=cfg.SOLVER.MILESTONE)


@SOLVER_REGISTRY.register()
def crnet_multi_step_scale(cfg, optimizer):
    return optim.lr_scheduler.MultiStepLR(optimizer[1], gamma=cfg.SOLVER.FACTOR, milestones=cfg.SOLVER.MILESTONE)
